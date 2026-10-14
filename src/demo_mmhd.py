import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = ROOT / "results" / "audio_baseline" / "audio_features.csv"
MODEL_FILE = ROOT / "results" / "audio_baseline" / "model.joblib"
METRICS_FILE = ROOT / "results" / "audio_baseline" / "metrics.json"
SAMPLES_FILE = ROOT / "annotations" / "pilot" / "pilot_samples.json"
ANNOTATIONS_FILE = (
    ROOT / "annotations" / "pilot"
    / "pilot_annotations_second_completed.json"
)

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


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate the MMHD audio humor classifier."
    )
    parser.add_argument(
        "--sample-id",
        help="Pilot sample ID to demonstrate. Defaults to the first audio sample.",
    )
    args = parser.parse_args()

    for path in [
        FEATURES_FILE,
        MODEL_FILE,
        METRICS_FILE,
        SAMPLES_FILE,
        ANNOTATIONS_FILE,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    feature_rows = read_csv(FEATURES_FILE)
    annotations = read_json(ANNOTATIONS_FILE)
    samples = read_json(SAMPLES_FILE)
    metrics = read_json(METRICS_FILE)

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
    true_label = labels_by_id.get(sample_id, "unknown")

    cv_metrics = metrics.get("evaluation", {}).get(
        "out_of_fold_metrics", {}
    )

    print("\n========== MMHD DEMONSTRATION ==========")
    print(f"Sample ID: {sample_id}")
    print(f"Source video: {sample.get('video_id', 'unknown')}")
    print(f"Transcript: {sample.get('text') or '[No transcript]'}")
    print("Model input: Acoustic features extracted from WAV audio")

    print("\n--- Prediction ---")
    print(f"Predicted label: {prediction}")
    print(f"Humorous probability: {humorous_probability:.1%}")
    print(f"Existing annotation: {true_label}")

    print("\n--- Features pushing toward humorous ---")
    for name, value in supporting:
        print(f"{name}: {value:+.3f}")

    print("\n--- Features pushing away from humorous ---")
    for name, value in opposing:
        print(f"{name}: {value:+.3f}")

    print("\n--- Cross-validation results ---")
    for name in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "f1_humorous",
        "roc_auc_humorous",
    ]:
        if name in cv_metrics:
            print(f"{name}: {cv_metrics[name]:.4f}")

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
        "The demonstration sample belongs to the pilot used to fit the "
        "saved model. Use cross-validation results to assess performance."
    )
    print("\n========================================")


if __name__ == "__main__":
    main()