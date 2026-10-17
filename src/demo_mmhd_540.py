import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = ROOT / "results" / "expanded_540" / "audio_baseline" / "audio_features.csv"
MODEL_FILE = ROOT / "results" / "expanded_540" / "audio_baseline" / "model.joblib"
TEXT_MODEL_FILE = ROOT / "results" / "expanded_540" / "text_baseline" / "model.joblib"
METRICS_FILE = ROOT / "results" / "expanded_540" / "audio_baseline" / "metrics.json"
SAMPLES_FILE = ROOT / "annotations" / "expanded_540" / "samples_540.json"
ANNOTATIONS_FILE = ROOT / "annotations" / "expanded_540" / "annotations_540.json"
MODALITY_FILE = ROOT / "results" / "modality_examples.csv"

FEATURE_NAMES = [
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


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def describe_asset(value):
    if not value:
        return "[not recorded]"
    asset_path = Path(str(value))
    if not asset_path.is_absolute():
        asset_path = ROOT / asset_path
    return f"{value} | exists={asset_path.is_file()}"

def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate the MMHD audio humor classifier."
    )
    parser.add_argument(
        "--sample-id",
        help="Annotated sample ID from the expanded dataset to demonstrate. Defaults to the first audio sample.",
    )
    args = parser.parse_args()

    for path in [
        FEATURES_FILE,
        MODEL_FILE,
        TEXT_MODEL_FILE,
        METRICS_FILE,
        SAMPLES_FILE,
        ANNOTATIONS_FILE,
        MODALITY_FILE,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    feature_rows = read_csv(FEATURES_FILE)
    annotations = read_json(ANNOTATIONS_FILE)
    samples = read_json(SAMPLES_FILE)
    metrics = read_json(METRICS_FILE)
    modality_rows = read_csv(MODALITY_FILE)
    modalities_by_id = {
        row["sample_id"]: row for row in modality_rows
    }

    features_by_id = {
        row["sample_id"]: row for row in feature_rows
    }
    labels_by_id = {
        row["sample_id"]: row["humor_label"] for row in annotations
    }
    samples_by_id = {
        row["sample_id"]: row for row in samples
    }

    if args.sample_id:
        sample_id = args.sample_id
    else:
        sample_id = feature_rows[0]["sample_id"]

    if sample_id not in features_by_id:
        raise ValueError(
            f"No usable audio features found for {sample_id!r}. "
            "Choose a sample present in audio_features.csv."
        )

    row = features_by_id[sample_id]

    X = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES]],
        dtype=np.float64,
    )

    model = joblib.load(MODEL_FILE)
    text_model = joblib.load(TEXT_MODEL_FILE)
    classifier = model.named_steps["classifier"]
    scaler = model.named_steps["scaler"]

    prediction = str(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    class_names = list(classifier.classes_)
    humorous_index = class_names.index("humorous")
    humorous_probability = float(probabilities[humorous_index])

    # Contributions to the linear decision score oriented toward
    # the humorous class. These are model-level feature explanations.
    direction = (
        1.0 if class_names[1] == "humorous" else -1.0
    )

    coefficients = direction * classifier.coef_[0]
    standardized_features = scaler.transform(X)[0]
    contributions = standardized_features * coefficients

    supporting = sorted(
        [
            (FEATURE_NAMES[i], float(contributions[i]))
            for i in range(len(FEATURE_NAMES))
            if contributions[i] > 0
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    opposing = sorted(
        [
            (FEATURE_NAMES[i], float(contributions[i]))
            for i in range(len(FEATURE_NAMES))
            if contributions[i] < 0
        ],
        key=lambda item: item[1],
    )[:3]

    sample = samples_by_id.get(sample_id, {})
    modality = modalities_by_id.get(sample_id, {})
    true_label = labels_by_id.get(sample_id, "unknown")
    transcript = " ".join(str(sample.get("text") or "").split())
    text_prediction = None
    text_probability = None
    if transcript:
        text_prediction = str(text_model.predict([transcript])[0])
        text_classes = list(text_model.named_steps["classifier"].classes_)
        text_probs = text_model.predict_proba([transcript])[0]
        text_probability = float(text_probs[text_classes.index("humorous")])

    cv_metrics = metrics.get("evaluation", {}).get(
        "out_of_fold_metrics", {}
    )
    late_fusion_metrics_path = (
        ROOT / "results" / "expanded_540" / "late_fusion" / "metrics.json"
    )
    late_fusion_metrics = read_json(late_fusion_metrics_path)
    late_fusion_cv_metrics = late_fusion_metrics["models"][
        "Equal-weight Late Fusion (Text + Audio)"
    ]

    print("\n========== MMHD DEMONSTRATION ==========")
    print(f"Sample ID: {sample_id}")
    print(f"Source video: {sample.get('video_id', 'unknown')}")
    print("\n--- Available modalities ---")
    print(f"Text ({modality.get('text_status', 'not indexed')}): "
          f"{sample.get('text') or '[No transcript]'}")
    print(f"Audio: {describe_asset(sample.get('audio_path'))}")
    image_paths = sample.get("image_paths") or []
    print(f"Images available: {len(image_paths)}")
    for index, image_path in enumerate(image_paths, start=1):
        print(f"  Image {index}: {describe_asset(image_path)}")
    print(f"OCR ({modality.get('ocr_status', 'not indexed')}): "
          f"{modality.get('ocr') or '[No OCR text]'}")
    print(f"Emoji ({modality.get('emoji_status', 'not indexed')}): "
          f"{modality.get('emoji') or '[No emoji recorded]'}")
    print("\nText and audio probabilities are combined using a fixed 50:50 late-fusion average when both are available.")

    print("\n--- Separate model predictions ---")
    print("Audio-only classifier:")
    print(f"  Predicted label: {prediction}")
    print(f"  Humorous probability: {humorous_probability:.1%}")
    print("Text-only classifier:")
    if text_prediction is not None:
        print(f"  Predicted label: {text_prediction}")
        print(f"  Humorous probability: {text_probability:.1%}")
    else:
        print("  Not run: no transcript available.")

    if text_probability is not None:
        late_fusion_probability = (
            humorous_probability + text_probability
        ) / 2.0
        late_fusion_prediction = (
            "humorous"
            if late_fusion_probability >= 0.5
            else "non_humorous"
        )
        print("\n--- Equal-weight text-audio late fusion ---")
        print(f"  Predicted label: {late_fusion_prediction}")
        print(f"  Humorous probability: {late_fusion_probability:.1%}")
        print("  This is an in-sample demo prediction, not a held-out result.")
    else:
        print("\nLate fusion not run: transcript unavailable.")

    print(f"\nExisting annotation: {true_label}")

    print("\n--- Features pushing toward humorous ---")
    for name, value in supporting:
        print(f"{name}: {value:+.3f}")

    print("\n--- Features pushing away from humorous ---")
    for name, value in opposing:
        print(f"{name}: {value:+.3f}")

    print("\n--- Audio-only cross-validation results ---")
    for name in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "f1_humorous",
        "roc_auc_humorous",
    ]:
        if name in cv_metrics:
            print(f"{name}: {cv_metrics[name]:.4f}")

    print("\n--- Late-fusion cross-validation results ---")
    for name in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "f1_humorous",
        "roc_auc_humorous",
    ]:
        if name in late_fusion_cv_metrics:
            print(f"{name}: {late_fusion_cv_metrics[name]:.4f}")

    print("\n--- Interpretation limits ---")
    print(
        "The probability is the classifier's output, not a guarantee "
        "that the sample is humorous."
    )
    print(
        "Feature contributions explain the classifier's decision score; "
        "they do not establish why a human finds the clip funny."
    )
    print(
        "The demonstration sample belongs to the expanded dataset used to fit the "
        "saved model. Use cross-validation results to assess performance."
    )
    print("\n========================================")


if __name__ == "__main__":
    main()
