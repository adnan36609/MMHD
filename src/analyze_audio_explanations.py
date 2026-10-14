import csv
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    ROOT / "results" / "audio_baseline" / "audio_features.csv"
)
MODEL_FILE = (
    ROOT / "results" / "audio_baseline" / "model.joblib"
)
ANNOTATIONS_FILE = (
    ROOT
    / "annotations"
    / "pilot"
    / "pilot_annotations_second_completed.json"
)

OUTPUT_DIR = ROOT / "results" / "audio_baseline"

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


def main():
    if not FEATURES_FILE.is_file():
        raise FileNotFoundError(FEATURES_FILE)

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(MODEL_FILE)

    if not ANNOTATIONS_FILE.is_file():
        raise FileNotFoundError(ANNOTATIONS_FILE)

    feature_rows = read_csv(FEATURES_FILE)

    with ANNOTATIONS_FILE.open(
        "r", encoding="utf-8-sig"
    ) as file:
        annotations = json.load(file)

    labels_by_id = {
        row["sample_id"]: row["humor_label"]
        for row in annotations
    }

    records = []

    for row in feature_rows:
        sample_id = row["sample_id"]

        if sample_id not in labels_by_id:
            raise ValueError(
                f"Missing label for sample {sample_id}"
            )

        records.append({
            **row,
            "true_label": labels_by_id[sample_id],
        })

    if not records:
        raise ValueError("No audio feature records found.")

    X = np.asarray([
        [float(row[name]) for name in FEATURE_NAMES]
        for row in records
    ], dtype=np.float64)

    model = joblib.load(MODEL_FILE)

    # This saved pipeline uses StandardScaler followed by LogisticRegression.
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]

    if list(classifier.classes_).count("humorous") != 1:
        raise ValueError("Expected a binary model with a humorous class.")

    # Convert the classifier's linear coefficients into the direction
    # associated with the humorous class.
    if classifier.classes_[1] == "humorous":
        direction = 1.0
    else:
        direction = -1.0

    coefficients = direction * classifier.coef_[0]
    intercept = direction * classifier.intercept_[0]

    # Features are standardized using statistics learned during training.
    X_standardized = scaler.transform(X)

    # Each contribution shows how one standardized feature contributes
    # to the model's humorous-versus-non-humorous decision score.
    contributions = X_standardized * coefficients

    probabilities = model.predict_proba(X)
    humorous_index = list(classifier.classes_).index("humorous")
    predictions = model.predict(X)

    # Export global model coefficients.
    effect_rows = []

    for index, feature_name in enumerate(FEATURE_NAMES):
        coefficient = float(coefficients[index])

        effect_rows.append({
            "feature": feature_name,
            "coefficient_toward_humorous": coefficient,
            "interpretation": (
                "Higher values push toward humorous"
                if coefficient > 0
                else "Higher values push away from humorous"
                if coefficient < 0
                else "No linear contribution"
            ),
        })

    effect_rows.sort(
        key=lambda row: abs(
            row["coefficient_toward_humorous"]
        ),
        reverse=True,
    )

    with (
        OUTPUT_DIR / "feature_effects.csv"
    ).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "feature",
                "coefficient_toward_humorous",
                "interpretation",
            ],
        )
        writer.writeheader()
        writer.writerows(effect_rows)

    # Export per-sample contributions for inspection.
    sample_rows = []

    for row_index, record in enumerate(records):
        contribution = contributions[row_index]

        positive_indices = sorted(
            [
                i for i, value in enumerate(contribution)
                if value > 0
            ],
            key=lambda i: contribution[i],
            reverse=True,
        )[:3]

        negative_indices = sorted(
            [
                i for i, value in enumerate(contribution)
                if value < 0
            ],
            key=lambda i: contribution[i],
        )[:3]

        supports = "; ".join(
            f"{FEATURE_NAMES[i]} ({contribution[i]:+.3f})"
            for i in positive_indices
        )

        opposes = "; ".join(
            f"{FEATURE_NAMES[i]} ({contribution[i]:+.3f})"
            for i in negative_indices
        )

        sample_rows.append({
            "sample_id": record["sample_id"],
            "video_id": record["video_id"],
            "true_label": record["true_label"],
            "model_prediction": str(predictions[row_index]),
            "humorous_probability": round(
                float(probabilities[row_index, humorous_index]), 6
            ),
            "features_supporting_humorous": supports,
            "features_opposing_humorous": opposes,
        })

    with (
        OUTPUT_DIR / "sample_explanations.csv"
    ).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "video_id",
                "true_label",
                "model_prediction",
                "humorous_probability",
                "features_supporting_humorous",
                "features_opposing_humorous",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    summary = {
        "method": (
            "Coefficient-based interpretation of the fitted "
            "audio Logistic Regression classifier"
        ),
        "model": str(MODEL_FILE.relative_to(ROOT)),
        "sample_count": len(records),
        "feature_count": len(FEATURE_NAMES),
        "top_absolute_coefficients": effect_rows[:5],
        "limitations": [
            "Feature contributions explain the fitted classifier, "
            "not the true cause of humor.",
            "The acoustic features are low-level signal statistics.",
            "Per-sample predictions are from the final model fitted "
            "on the complete audio pilot and are not held-out "
            "evaluation predictions.",
            "This analysis does not provide commonsense reasoning "
            "or natural-language explanations.",
        ],
    }

    with (
        OUTPUT_DIR / "interpretability_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n--- MMHD Audio Model Interpretability ---")
    print(f"Samples analyzed: {len(records)}")
    print("\nFeatures with the largest absolute coefficients:")

    for row in effect_rows[:5]:
        print(
            f"{row['feature']}: "
            f"{row['coefficient_toward_humorous']:+.4f} "
            f"({row['interpretation']})"
        )

    print("\nSaved files:")
    print(f"  {OUTPUT_DIR / 'feature_effects.csv'}")
    print(f"  {OUTPUT_DIR / 'sample_explanations.csv'}")
    print(f"  {OUTPUT_DIR / 'interpretability_summary.json'}")

    print(
        "\nNote: These are model-based feature contributions, "
        "not proof that any acoustic feature causes humor."
    )


if __name__ == "__main__":
    main()