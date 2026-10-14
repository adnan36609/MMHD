import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_clips"
OUTPUT_DIR = ROOT / "results"

PILOT_ANNOTATIONS = (
    ROOT
    / "annotations"
    / "pilot"
    / "pilot_annotations_second_completed.json"
)

MODALITIES = ["text", "audio", "visual", "ocr", "emoji"]


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def nonempty(value):
    """Check whether a field contains an actual non-empty payload."""
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, dict):
        return any(nonempty(item) for item in value.values())

    if isinstance(value, (list, tuple)):
        return any(nonempty(item) for item in value)

    return bool(value)


def resolve_path(value):
    if not value:
        return None

    path = Path(str(value).replace("\\", "/"))

    if not path.is_absolute():
        path = ROOT / path

    return path


def get_video_id(sample):
    return str(
        sample.get("source", {}).get("video_id")
        or sample.get("video_id")
        or ""
    )


def audit_sample(sample):
    modalities = sample.get("modalities", {})

    text_data = modalities.get("text", {})
    audio_data = modalities.get("audio", {})
    image_data = modalities.get("image", {})
    ocr_data = modalities.get("ocr", {})
    emoji_data = modalities.get("emoji", {})

    text_value = (
        text_data.get("value", "")
        if isinstance(text_data, dict)
        else ""
    )

    audio_path_value = (
        audio_data.get("path", "")
        if isinstance(audio_data, dict)
        else ""
    )

    image_paths = (
        image_data.get("paths", [])
        if isinstance(image_data, dict)
        else []
    )

    existing_images = [
        str(path)
        for path in image_paths
        if resolve_path(path) is not None
        and resolve_path(path).is_file()
    ]

    audio_path = resolve_path(audio_path_value)
    audio_exists = audio_path is not None and audio_path.is_file()

    ocr_value = (
        ocr_data.get("value", "")
        if isinstance(ocr_data, dict)
        else ""
    )

    emoji_value = (
        emoji_data.get("value", "")
        if isinstance(emoji_data, dict)
        else ""
    )

    signals = {
        "text": nonempty(text_value),
        "audio": audio_exists,
        "visual": len(existing_images) > 0,
        "ocr": nonempty(ocr_value),
        "emoji": nonempty(emoji_value),
    }

    status_fields = {
        "text": text_data,
        "audio": audio_data,
        "image": image_data,
        "ocr": ocr_data,
        "emoji": emoji_data,
    }

    statuses = {
        name: (
            str(value.get("status", "missing"))
            if isinstance(value, dict)
            else "missing"
        )
        for name, value in status_fields.items()
    }

    return {
        "sample_id": str(sample.get("sample_id", "")),
        "video_id": get_video_id(sample),
        "signals": signals,
        "statuses": statuses,
        "text": str(text_value or ""),
        "ocr": json.dumps(ocr_value, ensure_ascii=False)
        if not isinstance(ocr_value, str)
        else ocr_value,
        "emoji": json.dumps(emoji_value, ensure_ascii=False)
        if not isinstance(emoji_value, str)
        else emoji_value,
        "image_count": len(existing_images),
        "audio_exists": audio_exists,
    }


def main():
    dataset_files = sorted(
        SAMPLE_DIR.glob("*/dataset.json")
    )

    if not dataset_files:
        raise FileNotFoundError(
            f"No per-video dataset.json files found under {SAMPLE_DIR}"
        )

    records = []

    for path in dataset_files:
        dataset = read_json(path)

        if not isinstance(dataset, list):
            raise ValueError(
                f"Expected a JSON list in {path}"
            )

        records.extend(audit_sample(sample) for sample in dataset)

    sample_ids = [row["sample_id"] for row in records]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "Duplicate sample IDs detected across per-video datasets."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "corpus": {
            "dataset_files": len(dataset_files),
            "samples": len(records),
            "unique_sample_ids": len(set(sample_ids)),
        },
        "modalities": {},
        "pairwise_and_joint_availability": {},
        "per_video": {},
    }

    # Count both declared statuses and actual usable payloads.
    for modality in MODALITIES:
        status_key = "image" if modality == "visual" else modality

        status_counts = Counter(
            row["statuses"][status_key]
            for row in records
        )

        usable_count = sum(
            row["signals"][modality]
            for row in records
        )

        report["modalities"][modality] = {
            "samples_with_actual_content_or_file": int(usable_count),
            "samples_without_usable_content_or_file": (
                len(records) - int(usable_count)
            ),
            "status_counts": dict(status_counts),
        }

    # Count pairwise intersections and the full five-stream intersection.
    for size in (2, 3, 4, 5):
        for combination in combinations(MODALITIES, size):
            count = sum(
                all(row["signals"][name] for name in combination)
                for row in records
            )

            key = " + ".join(combination)
            report["pairwise_and_joint_availability"][key] = int(count)

    # Summarize availability by source video.
    video_rows = {}

    for row in records:
        video_id = row["video_id"] or "unknown"

        if video_id not in video_rows:
            video_rows[video_id] = {
                "samples": 0,
                **{name: 0 for name in MODALITIES},
            }

        video_rows[video_id]["samples"] += 1

        for modality in MODALITIES:
            video_rows[video_id][modality] += int(
                row["signals"][modality]
            )

    report["per_video"] = video_rows

    # Match the labeled pilot against the complete extracted corpus.
    corpus_by_id = {row["sample_id"]: row for row in records}

    if PILOT_ANNOTATIONS.is_file():
        annotations = read_json(PILOT_ANNOTATIONS)

        matched = [
            (annotation, corpus_by_id[annotation["sample_id"]])
            for annotation in annotations
            if annotation["sample_id"] in corpus_by_id
        ]

        pilot_report = {
            "annotation_records": len(annotations),
            "matched_to_corpus": len(matched),
            "unmatched_ids": len(annotations) - len(matched),
            "label_distribution": dict(
                Counter(
                    annotation.get("humor_label", "missing")
                    for annotation, _ in matched
                )
            ),
            "modality_availability": {},
        }

        for modality in MODALITIES:
            pilot_report["modality_availability"][modality] = sum(
                row["signals"][modality]
                for _, row in matched
            )

        report["labeled_pilot"] = pilot_report

    # Save examples so OCR and emoji payloads can be inspected manually.
    examples_path = OUTPUT_DIR / "modality_examples.csv"

    with examples_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fieldnames = [
            "sample_id",
            "video_id",
            "text_status",
            "text",
            "ocr_status",
            "ocr",
            "emoji_status",
            "emoji",
            "audio_exists",
            "image_count",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        examples_written = 0

        for row in records:
            if not row["signals"]["ocr"] and not row["signals"]["emoji"]:
                continue

            writer.writerow({
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "text_status": row["statuses"]["text"],
                "text": row["text"][:300],
                "ocr_status": row["statuses"]["ocr"],
                "ocr": row["ocr"][:300],
                "emoji_status": row["statuses"]["emoji"],
                "emoji": row["emoji"][:300],
                "audio_exists": row["audio_exists"],
                "image_count": row["image_count"],
            })

            examples_written += 1

    report["examples_file"] = str(examples_path.relative_to(ROOT))
    report["examples_written"] = examples_written

    report_path = OUTPUT_DIR / "modality_audit.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n--- Full MMHD Modality Audit ---")
    print(f"Dataset files: {len(dataset_files)}")
    print(f"Total samples: {len(records)}")
    print(f"Unique sample IDs: {len(set(sample_ids))}")

    print("\nActual content/file availability:")

    for modality, values in report["modalities"].items():
        print(
            f"{modality:>7}: "
            f"{values['samples_with_actual_content_or_file']} usable; "
            f"{values['samples_without_usable_content_or_file']} unavailable"
        )

    print("\nJoint availability:")

    for combination, count in (
        report["pairwise_and_joint_availability"].items()
    ):
        print(f"{combination}: {count}")

    if "labeled_pilot" in report:
        print("\nLabeled pilot:")
        print(json.dumps(report["labeled_pilot"], indent=2))

    print(f"\nSaved report: {report_path}")
    print(f"Saved content examples: {examples_path}")


if __name__ == "__main__":
    main()