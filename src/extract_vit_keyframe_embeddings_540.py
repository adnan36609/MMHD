"""Extract cached ViT embeddings for MMHD's annotated keyframes.

Uses locally cached ViT weights and CPU inference only.
Processes two representative keyframes per annotated scene.
Saves progress to CSV so completed embeddings can be reused.

Run:
    python .\src\extract_vit_keyframe_embeddings_540.py
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = ROOT / "annotations" / "expanded_540" / "samples_540.json"
ANNOTATIONS_FILE = ROOT / "annotations" / "expanded_540" / "annotations_540.json"

OUTPUT_DIR = ROOT / "results" / "expanded_540" / "vit_keyframe_embeddings"
OUTPUT_FILE = OUTPUT_DIR / "embeddings.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"

MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 4

EMBEDDING_PREFIX = "embedding_"
METADATA_FIELDS = [
    "sample_id",
    "video_id",
    "humor_label",
    "frame_index",
    "image_path",
    "status",
    "error",
]


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def resolve_image_path(raw_path):
    if not raw_path:
        return None

    path = Path(str(raw_path))
    if not path.is_absolute():
        path = ROOT / path

    return path


def load_existing_embeddings():
    """Load prior results to allow interrupted runs to resume."""
    existing = {}

    if not OUTPUT_FILE.exists():
        return existing

    with OUTPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                key = (row["sample_id"], int(row["frame_index"]))
            except (KeyError, ValueError):
                continue

            existing[key] = row

    print(f"Loaded {len(existing)} cached keyframe records.")
    return existing


def image_to_tensor(image_path, image_size):
    """Apply standard ViT normalization without torchvision."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (image_size, image_size),
            Image.Resampling.BICUBIC,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0

    # Match the standard ViT processor normalization:
    # mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5].
    array = (array - 0.5) / 0.5

    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def make_empty_row(sample, label, frame_index, image_path, status, error=""):
    return {
        "sample_id": str(sample["sample_id"]),
        "video_id": str(sample.get("video_id", "")),
        "humor_label": str(label),
        "frame_index": str(frame_index),
        "image_path": str(image_path or ""),
        "status": status,
        "error": error,
    }


def save_embeddings(rows, samples, embedding_dim):
    """Write an ordered, recoverable CSV checkpoint."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = METADATA_FIELDS + [
        f"{EMBEDDING_PREFIX}{i:03d}"
        for i in range(embedding_dim)
    ]

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    with temporary_file.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for sample in samples:
            sample_id = str(sample["sample_id"])

            for frame_index in (1, 2):
                row = rows.get((sample_id, frame_index))

                if row is not None:
                    writer.writerow(row)

    temporary_file.replace(OUTPUT_FILE)


def main():
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = read_json(SAMPLES_FILE)
    annotations = read_json(ANNOTATIONS_FILE)

    if not isinstance(samples, list) or not isinstance(annotations, list):
        raise ValueError("Expected sample and annotation JSON lists.")

    sample_ids = [str(item["sample_id"]) for item in samples]
    annotation_ids = [str(item["sample_id"]) for item in annotations]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs found in the samples file.")

    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Duplicate sample IDs found in the annotations file.")

    label_by_id = {
        str(item["sample_id"]): item["humor_label"]
        for item in annotations
    }

    missing_annotations = set(sample_ids) - set(label_by_id)
    if missing_annotations:
        raise ValueError(
            f"{len(missing_annotations)} samples lack human annotations."
        )

    rows = load_existing_embeddings()

    print(f"Annotated scenes: {len(samples)}")
    print("Loading locally cached ViT weights on CPU...")

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    model.to("cpu")
    model.eval()

    image_size = int(model.config.image_size)
    embedding_dim = int(model.config.hidden_size)

    print(f"Model image size: {image_size} x {image_size}")
    print(f"Embedding dimensions: {embedding_dim}")
    print(f"Batch size: {BATCH_SIZE}")

    pending = []
    newly_encoded = 0
    new_errors = 0

    def flush_pending():
        nonlocal pending, newly_encoded, new_errors

        if not pending:
            return

        batch = torch.stack([item["tensor"] for item in pending])

        with torch.inference_mode():
            output = model(pixel_values=batch)
            embeddings = (
                output.last_hidden_state[:, 0, :]
                .detach()
                .cpu()
                .numpy()
            )

        for item, vector in zip(pending, embeddings):
            row = dict(item["metadata"])
            row["status"] = "ok"
            row["error"] = ""

            for index, value in enumerate(vector):
                row[f"{EMBEDDING_PREFIX}{index:03d}"] = str(
                    float(value)
                )

            rows[item["key"]] = row
            newly_encoded += 1

        pending = []

        # Preserve progress after every successful batch.
        save_embeddings(rows, samples, embedding_dim)

        print(
            f"New embeddings: {newly_encoded}; "
            f"new image errors: {new_errors}; "
            f"elapsed: {time.time() - started:.1f}s"
        )

    for sample_number, sample in enumerate(samples, start=1):
        sample_id = str(sample["sample_id"])
        label = label_by_id[sample_id]
        image_paths = sample.get("image_paths") or []

        for frame_index in (1, 2):
            raw_path = (
                image_paths[frame_index - 1]
                if len(image_paths) >= frame_index
                else None
            )

            relative_path = str(raw_path or "")
            key = (sample_id, frame_index)

            old_row = rows.get(key)
            input_matches = (
                old_row is not None
                and old_row.get("image_path", "") == relative_path
                and old_row.get("humor_label", "") == str(label)
                and old_row.get("video_id", "")
                == str(sample.get("video_id", ""))
            )

            if input_matches and old_row.get("status") == "ok":
                continue

            if (
                input_matches
                and old_row.get("status") == "missing_path"
            ):
                continue

            if not raw_path:
                rows[key] = make_empty_row(
                    sample,
                    label,
                    frame_index,
                    "",
                    "missing_path",
                )
                continue

            path = resolve_image_path(raw_path)

            if path is None or not path.is_file():
                rows[key] = make_empty_row(
                    sample,
                    label,
                    frame_index,
                    relative_path,
                    "missing_file",
                )
                new_errors += 1
                continue

            metadata = make_empty_row(
                sample,
                label,
                frame_index,
                relative_path,
                "pending",
            )

            try:
                tensor = image_to_tensor(path, image_size)
            except Exception as exc:
                rows[key] = make_empty_row(
                    sample,
                    label,
                    frame_index,
                    relative_path,
                    "image_error",
                    f"{type(exc).__name__}: {exc}",
                )
                new_errors += 1
                continue

            pending.append({
                "key": key,
                "metadata": metadata,
                "tensor": tensor,
            })

            if len(pending) >= BATCH_SIZE:
                flush_pending()

        if sample_number % 50 == 0:
            print(
                f"Inspected {sample_number}/{len(samples)} scenes."
            )

    flush_pending()
    save_embeddings(rows, samples, embedding_dim)

    ordered_rows = [
        rows[(str(sample["sample_id"]), frame_index)]
        for sample in samples
        for frame_index in (1, 2)
        if (str(sample["sample_id"]), frame_index) in rows
    ]

    status_counts = Counter(row["status"] for row in ordered_rows)

    summary = {
        "model": MODEL_NAME,
        "device": "cpu",
        "num_annotated_scenes": len(samples),
        "expected_keyframe_records": len(samples) * 2,
        "embedding_dimensions": embedding_dim,
        "batch_size": BATCH_SIZE,
        "status_counts": dict(status_counts),
        "successful_embeddings": status_counts.get("ok", 0),
        "new_embeddings_this_run": newly_encoded,
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
        "elapsed_seconds_this_run": round(time.time() - started, 2),
        "preprocessing": (
            "RGB resize to model image size; pixel range [0,1]; "
            "channel-wise normalization with mean and std 0.5."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nEXTRACTION SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()