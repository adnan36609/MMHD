import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    ROOT / "results" / "expanded_540" / "audio_baseline"
    / "audio_features.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "expanded_540"
    / "audio_interpretability"
)

N_SPLITS = 5
RANDOM_STATE = 42

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


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def main():
    if not FEATURES_FILE.is_file():
        raise FileNotFoundError(FEATURES_FILE)

    with FEATURES_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        records = list(csv.DictReader(file))

    if not records:
        raise ValueError("No audio feature records found.")

    required_columns = {
        "sample_id", "video_id", "label", *FEATURE_NAMES
    }

    missing_columns = required_columns - set(records[0])

    if missing_columns:
        raise ValueError(
            f"Missing columns in feature CSV: {sorted(missing_columns)}"
        )

    sample_ids = [row["sample_id"] for row in records]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs found.")

    if any(not row["video_id"].strip() for row in records):
        raise ValueError("Some records have missing video IDs.")

    X = np.asarray(
        [
            [float(row[name]) for name in FEATURE_NAMES]
            for row in records
        ],
        dtype=np.float64,
    )

    y = [row["label"] for row in records]
    groups = [row["video_id"] for row in records]

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains non-finite values.")

    if set(y) != {"humorous", "non_humorous"}:
        raise ValueError("Expected both humor labels.")

    if len(set(groups)) < N_SPLITS:
        raise ValueError("Not enough source videos for five-fold CV.")

    splitter = GroupKFold(n_splits=N_SPLITS)
    fold_rows = []
    fold_coefficients = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        train_videos = {groups[i] for i in train_idx}
        test_videos = {groups[i] for i in test_idx}

        if train_videos & test_videos:
            raise RuntimeError("Source-video leakage detected.")

        model = build_model()
        model.fit(X[train_idx], [y[i] for i in train_idx])

        classifier = model.named_steps["classifier"]
        classes = list(classifier.classes_)

        if "humorous" not in classes:
            raise ValueError("Humorous class missing from model.")

        # Orient every coefficient toward the humorous class.
        direction = 1.0 if classes[1] == "humorous" else -1.0
        coefficients = direction * classifier.coef_[0]

        ranks = np.empty(len(FEATURE_NAMES), dtype=int)

        ranked_indices = sorted(
            range(len(FEATURE_NAMES)),
            key=lambda i: abs(coefficients[i]),
            reverse=True,
        )

        for rank, index in enumerate(ranked_indices, start=1):
            ranks[index] = rank

        for index, feature in enumerate(FEATURE_NAMES):
            coefficient = float(coefficients[index])

            fold_coefficients[feature].append({
                "fold": fold,
                "coefficient_toward_humorous": coefficient,
                "absolute_coefficient": abs(coefficient),
                "absolute_rank": int(ranks[index]),
            })

            fold_rows.append({
                "fold": fold,
                "training_samples": len(train_idx),
                "validation_samples": len(test_idx),
                "training_videos": len(train_videos),
                "validation_videos": len(test_videos),
                "feature": feature,
                "coefficient_toward_humorous": coefficient,
                "absolute_coefficient": abs(coefficient),
                "absolute_rank": int(ranks[index]),
            })

        print(
            f"Fold {fold}/{N_SPLITS}: "
            f"train={len(train_idx)}, validation={len(test_idx)}, "
            f"training videos={len(train_videos)}, "
            f"validation videos={len(test_videos)}"
        )

    # Summarize coefficient magnitude and sign consistency.
    stability_rows = []

    for feature in FEATURE_NAMES:
        values = [
            item["coefficient_toward_humorous"]
            for item in fold_coefficients[feature]
        ]

        ranks = [
            item["absolute_rank"]
            for item in fold_coefficients[feature]
        ]

        positive = sum(value > 1e-12 for value in values)
        negative = sum(value < -1e-12 for value in values)
        zero = N_SPLITS - positive - negative

        if positive > negative:
            dominant_direction = "toward humorous"
        elif negative > positive:
            dominant_direction = "away from humorous"
        else:
            dominant_direction = "mixed"

        stability_rows.append({
            "feature": feature,
            "mean_coefficient": float(np.mean(values)),
            "std_coefficient": float(np.std(values)),
            "mean_absolute_coefficient": float(
                np.mean(np.abs(values))
            ),
            "positive_folds": positive,
            "negative_folds": negative,
            "near_zero_folds": zero,
            "direction_consistency": round(
                max(positive, negative) / N_SPLITS, 4
            ),
            "mean_absolute_rank": round(float(np.mean(ranks)), 2),
            "dominant_direction": dominant_direction,
        })

    stability_rows.sort(
        key=lambda row: row["mean_absolute_coefficient"],
        reverse=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "fold_coefficients.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fold_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(fold_rows)

    with (OUTPUT_DIR / "coefficient_stability.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(stability_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(stability_rows)

    summary = {
        "method": (
            "Five-fold GroupKFold coefficient stability analysis "
            "using source-video groups"
        ),
        "sample_count": len(records),
        "source_video_count": len(set(groups)),
        "fold_count": N_SPLITS,
        "feature_count": len(FEATURE_NAMES),
        "model": (
            "StandardScaler + LogisticRegression "
            "(class_weight=balanced, max_iter=3000)"
        ),
        "top_features_by_mean_absolute_coefficient": (
            stability_rows[:5]
        ),
        "limitations": [
            "Five folds provide only a preliminary stability assessment.",
            "Coefficients describe model associations, not causal effects.",
            "Correlated acoustic features can influence coefficient signs "
            "and magnitudes.",
            "Consistent coefficients do not establish generalization "
            "to unseen datasets or source domains.",
        ],
    }

    with (OUTPUT_DIR / "coefficient_stability_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n--- Audio Coefficient Stability ---")
    print(f"Samples: {len(records)}")
    print(f"Source videos: {len(set(groups))}")

    print("\nTop features by mean absolute coefficient:")

    for row in stability_rows[:5]:
        print(
            f"{row['feature']}: "
            f"mean={row['mean_coefficient']:+.4f}, "
            f"std={row['std_coefficient']:.4f}, "
            f"direction={row['dominant_direction']}, "
            f"consistency={row['direction_consistency']:.0%}, "
            f"mean rank={row['mean_absolute_rank']:.2f}"
        )

    print("\nSaved results to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()