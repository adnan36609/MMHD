import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np

from train_audio_baseline_540 import extract_audio_features


ROOT = Path(__file__).resolve().parents[1]

SAMPLE_ROOT = ROOT / "sample_clips"
ANNOTATIONS_FILE = ROOT / "annotations" / "expanded_540" / "annotations_540.json"

TEXT_MODEL_FILE = (
    ROOT / "results" / "expanded_540" / "text_baseline" / "model.joblib"
)
AUDIO_MODEL_FILE = (
    ROOT / "results" / "expanded_540" / "audio_baseline" / "model.joblib"
)

OUTPUT_DIR = ROOT / "results" / "full_corpus"
PREDICTIONS_FILE = OUTPUT_DIR / "humor_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "humor_predictions_summary.json"

AUDIO_FEATURES = [
    "duration_seconds",
    "rms_mean",
    "rms_std",
    "rms_p90",
    "zero_crossing_mean",
    "zero_crossing_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "silence_frame_ratio",
]

POSITIVE_LABEL = "humorous"


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def resolve_asset(value):
    if not value:
        return None

    candidate = Path(str(value))

    options = (
        [candidate]
        if candidate.is_absolute()
        else [ROOT / candidate, ROOT / "sample_clips" / candidate]
    )

    for path in options:
        if path.is_file():
            return path

    return None


def positive_probabilities(model, inputs):
    """Return the model's probability for the humorous class."""
    probabilities = model.predict_proba(inputs)
    classes = list(model.classes_)

    if POSITIVE_LABEL not in classes:
        raise ValueError(
            f"Model classes {classes} do not contain {POSITIVE_LABEL!r}."
        )

    return probabilities[:, classes.index(POSITIVE_LABEL)]


def load_corpus():
    records = []
    seen_ids = set()

    manifest_files = sorted(SAMPLE_ROOT.glob("*/dataset.json"))

    if not manifest_files:
        raise FileNotFoundError(
            f"No dataset manifests found under {SAMPLE_ROOT}"
        )

    for manifest_path in manifest_files:
        manifest = read_json(manifest_path)

        if not isinstance(manifest, list):
            raise ValueError(f"Unexpected format: {manifest_path}")

        for raw in manifest:
            sample_id = str(raw.get("sample_id") or "").strip()

            if not sample_id:
                continue

            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")

            seen_ids.add(sample_id)

            source = raw.get("source") or {}
            timing = raw.get("timing") or {}
            modalities = raw.get("modalities") or {}

            text_data = modalities.get("text") or {}
            audio_data = modalities.get("audio") or {}
            image_data = modalities.get("image") or {}
            ocr_data = modalities.get("ocr") or {}
            emoji_data = modalities.get("emoji") or {}

            text = " ".join(
                str(text_data.get("value") or "").split()
            )

            image_paths = image_data.get("paths") or []
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            records.append({
                "sample_id": sample_id,
                "video_id": str(source.get("video_id") or ""),
                "source_scene": str(source.get("source_scene") or ""),
                "start_seconds": timing.get("start_seconds"),
                "duration_seconds": timing.get("duration_seconds"),
                "text": text,
                "ocr": str(ocr_data.get("value") or "").strip(),
                "emoji": str(emoji_data.get("value") or "").strip(),
                "audio_path": str(audio_data.get("path") or ""),
                "image_count": len(image_paths),
                "has_text": bool(text),
                "has_audio_path": bool(audio_data.get("path")),
                "has_image": bool(image_paths),
                "has_ocr": bool(str(ocr_data.get("value") or "").strip()),
                "has_emoji": bool(str(emoji_data.get("value") or "").strip()),
            })

    return records


def main():
    start_time = time.perf_counter()

    for path in [TEXT_MODEL_FILE, AUDIO_MODEL_FILE, ANNOTATIONS_FILE]:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading full corpus...")
    records = load_corpus()

    annotations = read_json(ANNOTATIONS_FILE)
    labels_by_id = {
        row["sample_id"]: row["humor_label"]
        for row in annotations
    }

    for record in records:
        record["human_label"] = labels_by_id.get(
            record["sample_id"], "not_annotated"
        )

    print(f"Accepted corpus records: {len(records):,}")
    print(
        f"Records with transcripts: "
        f"{sum(r['has_text'] for r in records):,}"
    )
    print(
        f"Records with recorded audio paths: "
        f"{sum(r['has_audio_path'] for r in records):,}"
    )

    print("\nLoading saved humor classifiers...")
    text_model = joblib.load(TEXT_MODEL_FILE)
    audio_model = joblib.load(AUDIO_MODEL_FILE)

    # Generate text predictions in a batch.
    text_indices = [
        i for i, record in enumerate(records)
        if record["text"]
    ]

    text_probabilities = {}

    if text_indices:
        print(f"Predicting humor from {len(text_indices):,} transcripts...")

        texts = [records[i]["text"] for i in text_indices]
        probabilities = positive_probabilities(text_model, texts)

        text_probabilities = {
            index: float(probability)
            for index, probability in zip(text_indices, probabilities)
        }

    # Reuse the existing acoustic feature extractor for audio.
    audio_probabilities = {}
    audio_errors = {}

    audio_indices = [
        i for i, record in enumerate(records)
        if record["audio_path"]
    ]

    print(f"\nProcessing {len(audio_indices):,} audio paths...")
    audio_start = time.perf_counter()

    for count, index in enumerate(audio_indices, start=1):
        record = records[index]
        sample_id = record["sample_id"]

        audio_path = resolve_asset(record["audio_path"])

        if audio_path is None:
            audio_errors[sample_id] = "audio_file_not_found"
            continue

        try:
            features = extract_audio_features(audio_path)

            vector = np.asarray(
                [[float(features[name]) for name in AUDIO_FEATURES]],
                dtype=np.float64,
            )

            if not np.isfinite(vector).all():
                raise ValueError("Non-finite audio features")

            probability = positive_probabilities(
                audio_model,
                vector,
            )[0]

            audio_probabilities[index] = float(probability)

        except Exception as exc:
            audio_errors[sample_id] = (
                f"{type(exc).__name__}: {str(exc)[:180]}"
            )

        if count % 250 == 0 or count == len(audio_indices):
            print(
                f"Audio processed: {count:,}/{len(audio_indices):,} | "
                f"usable predictions: {len(audio_probabilities):,} | "
                f"elapsed: {time.perf_counter() - audio_start:.1f}s"
            )

    # Use fixed equal-weight late fusion when BOTH models are available.
    # Otherwise, fall back to the available model and record its identity.
    output_rows = []

    method_counts = {
        "text_audio_late_fusion": 0,
        "text_only_fallback": 0,
        "audio_only_fallback": 0,
        "no_available_model": 0,
    }

    for index, record in enumerate(records):
        text_probability = text_probabilities.get(index)
        audio_probability = audio_probabilities.get(index)

        if (
            text_probability is not None
            and audio_probability is not None
        ):
            probability = (
                text_probability + audio_probability
            ) / 2.0

            method = "text_audio_late_fusion"
            disagreement = abs(
                text_probability - audio_probability
            )

        elif text_probability is not None:
            probability = text_probability
            method = "text_only_fallback"
            disagreement = None

        elif audio_probability is not None:
            probability = audio_probability
            method = "audio_only_fallback"
            disagreement = None

        else:
            probability = None
            method = "no_available_model"
            disagreement = None

        method_counts[method] += 1

        if probability is None:
            prediction = "unavailable"
            score_margin = None
        else:
            prediction = (
                "humorous"
                if probability >= 0.5
                else "non_humorous"
            )

            # Distance from the decision boundary, NOT calibrated confidence.
            score_margin = abs(probability - 0.5) * 2.0

        if record["human_label"] != "not_annotated":
            interpretation_status = (
                "annotated_sample_direct_inference_not_evaluation"
            )
        else:
            interpretation_status = "unlabelled_corpus_inference"

        output_rows.append({
            **record,
            "text_humorous_probability": text_probability,
            "audio_humorous_probability": audio_probability,
            "humor_probability": probability,
            "humor_prediction": prediction,
            "prediction_method": method,
            "text_audio_probability_disagreement": disagreement,
            "decision_boundary_margin": score_margin,
            "interpretation_status": interpretation_status,
            "audio_processing_error": audio_errors.get(
                record["sample_id"], ""
            ),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = list(output_rows[0].keys()) if output_rows else []

    with PREDICTIONS_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "experiment": "Full-corpus humor inference",
        "corpus_records": len(records),
        "records_with_text": len(text_indices),
        "audio_paths_attempted": len(audio_indices),
        "usable_audio_predictions": len(audio_probabilities),
        "usable_text_predictions": len(text_probabilities),
        "prediction_method_counts": method_counts,
        "annotated_records": sum(
            r["human_label"] != "not_annotated"
            for r in records
        ),
        "audio_processing_errors": len(audio_errors),
        "audio_error_examples": dict(list(audio_errors.items())[:20]),
        "fusion_rule": (
            "Equal-weight average of text and audio humorous probabilities "
            "when both exist; otherwise use the available model."
        ),
        "decision_threshold": 0.5,
        "confidence_note": (
            "Model probabilities are not necessarily calibrated. "
            "decision_boundary_margin is not a calibrated confidence score."
        ),
        "evaluation_note": (
            "These batch predictions are not out-of-fold evaluation results. "
            "Predictions for annotated samples may be in-sample. Use saved "
            "out-of-fold results for reported predictive performance."
        ),
        "runtime_seconds": round(time.perf_counter() - start_time, 2),
        "prediction_file": str(PREDICTIONS_FILE.relative_to(ROOT)),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== FULL-CORPUS HUMOR INFERENCE SUMMARY ===")
    print(f"Corpus records: {len(records):,}")
    print(f"Text predictions: {len(text_probabilities):,}")
    print(f"Audio predictions: {len(audio_probabilities):,}")

    for method, count in method_counts.items():
        print(f"{method}: {count:,}")

    print(f"Audio errors: {len(audio_errors):,}")
    print(f"Runtime: {summary['runtime_seconds']:.1f}s")
    print(f"Predictions saved: {PREDICTIONS_FILE.relative_to(ROOT)}")
    print(f"Summary saved: {SUMMARY_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()