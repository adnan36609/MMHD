"""Evaluate fixed-weight late fusion using existing out-of-fold predictions."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results" / "expanded_540"
    / "multimodal_comparison" / "oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "results" / "expanded_540" / "late_fusion"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "oof_predictions.csv"

TEXT_COL = "text_only_humorous_probability"
AUDIO_COL = "audio_only_humorous_probability"
FEATURE_FUSION_COL = "text_audio_humorous_probability"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"OOF predictions not found: {INPUT_FILE}")

    with INPUT_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("OOF prediction file is empty.")

    required = {
        "sample_id",
        "video_id",
        "true_label",
        TEXT_COL,
        AUDIO_COL,
        FEATURE_FUSION_COL,
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs found in OOF predictions.")

    labels = [row["true_label"] for row in rows]
    if not set(labels).issubset({"humorous", "non_humorous"}):
        raise ValueError(f"Unexpected labels found: {set(labels)}")

    y_true = np.array(
        [1 if label == "humorous" else 0 for label in labels],
        dtype=int,
    )

    def probabilities(column):
        values = np.array([float(row[column]) for row in rows])
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite scores in {column}")
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"Scores outside [0, 1] in {column}")
        return values

    text_prob = probabilities(TEXT_COL)
    audio_prob = probabilities(AUDIO_COL)
    feature_fusion_prob = probabilities(FEATURE_FUSION_COL)

    # Fixed 50:50 average; no weights are fitted or tuned on these labels.
    late_fusion_prob = (text_prob + audio_prob) / 2.0

    score_sets = {
        "Text-only": text_prob,
        "Audio-only": audio_prob,
        "Feature-level Text + Audio": feature_fusion_prob,
        "Equal-weight Late Fusion (Text + Audio)": late_fusion_prob,
    }

    model_metrics = {}
    for name, scores in score_sets.items():
        y_pred = (scores >= 0.5).astype(int)

        model_metrics[name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "precision_humorous": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "recall_humorous": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "f1_humorous": float(
                f1_score(y_true, y_pred, zero_division=0)
            ),
            "roc_auc_humorous": float(
                roc_auc_score(y_true, scores)
            ),
            "confusion_matrix_label_order": [
                "non_humorous", "humorous"
            ],
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).tolist(),
        }

    counts = Counter(labels)
    metrics = {
        "experiment": "Fixed equal-weight text-audio late fusion",
        "evaluation_method": (
            "Deterministic 50:50 average of text-only and audio-only "
            "out-of-fold humorous-class probabilities from the existing "
            "five-fold GroupKFold-by-source-video evaluation. "
            "Fusion weights were not tuned on evaluation labels."
        ),
        "positive_class": "humorous",
        "decision_threshold": 0.5,
        "samples": len(rows),
        "distinct_source_videos": len(
            {row["video_id"] for row in rows}
        ),
        "label_distribution": dict(counts),
        "models": model_metrics,
        "limitations": [
            "This is a fixed-weight fusion experiment, not a learned fusion model.",
            "Input probabilities have not been calibrated.",
            "Results apply to this common subset and these cross-validation folds.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with PREDICTIONS_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "sample_id",
            "video_id",
            "true_label",
            "text_humorous_probability",
            "audio_humorous_probability",
            "feature_fusion_humorous_probability",
            "late_fusion_humorous_probability",
            "late_fusion_prediction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            writer.writerow({
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "true_label": labels[i],
                "text_humorous_probability": f"{text_prob[i]:.6f}",
                "audio_humorous_probability": f"{audio_prob[i]:.6f}",
                "feature_fusion_humorous_probability": (
                    f"{feature_fusion_prob[i]:.6f}"
                ),
                "late_fusion_humorous_probability": (
                    f"{late_fusion_prob[i]:.6f}"
                ),
                "late_fusion_prediction": (
                    "humorous"
                    if late_fusion_prob[i] >= 0.5
                    else "non_humorous"
                ),
            })

    print("\n--- MMHD Late-Fusion Experiment ---")
    print(f"Samples: {len(rows)}")
    print(f"Distinct source videos: {metrics['distinct_source_videos']}")
    print(f"Label counts: {dict(counts)}")
    print("\nModel                              Accuracy  Balanced Acc.  Macro F1  Humor F1  ROC-AUC")

    for name, result in model_metrics.items():
        print(
            f"{name:35} "
            f"{result['accuracy']:.4f}    "
            f"{result['balanced_accuracy']:.4f}        "
            f"{result['macro_f1']:.4f}    "
            f"{result['f1_humorous']:.4f}    "
            f"{result['roc_auc_humorous']:.4f}"
        )

    print(f"\nMetrics: {METRICS_FILE}")
    print(f"Predictions: {PREDICTIONS_FILE}")

if __name__ == "__main__":
    main()
