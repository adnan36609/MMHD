"""Exploratory test of visual probabilities added to MMHD late fusion.

Uses saved out-of-fold predictions from the same 391-sample evaluation.
Does not retrain models or overwrite baseline results.
The visual configuration was selected after inspecting earlier results,
so these comparisons are exploratory, not independent confirmation.
"""

import csv
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

BASELINE_FILE = (
    ROOT / "results" / "expanded_540"
    / "late_fusion" / "oof_predictions.csv"
)

VISUAL_FILE = (
    ROOT / "results" / "expanded_540"
    / "cross_frame_vit" / "oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "results" / "expanded_540" / "visual_late_fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "oof_predictions.csv"
OUTPUT_JSON = OUTPUT_DIR / "metrics.json"

VISUAL_PROBABILITY_COLUMN = (
    "absolute_frame_difference_humorous_probability"
)

# Predeclared diagnostic weights: proportion assigned to visual probability.
VISUAL_WEIGHTS = [0.10, 0.20, 0.30, 0.50]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def metrics(y_true, probability):
    predictions = [int(p >= 0.5) for p in probability]

    return {
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def main():
    baseline_rows = read_csv(BASELINE_FILE)
    visual_rows = read_csv(VISUAL_FILE)

    if not baseline_rows or not visual_rows:
        raise ValueError("One or both prediction files are empty.")

    baseline_by_id = {r["sample_id"]: r for r in baseline_rows}
    visual_by_id = {r["sample_id"]: r for r in visual_rows}

    if len(baseline_by_id) != len(baseline_rows):
        raise ValueError("Duplicate sample IDs in baseline predictions.")

    if len(visual_by_id) != len(visual_rows):
        raise ValueError("Duplicate sample IDs in visual predictions.")

    if set(baseline_by_id) != set(visual_by_id):
        raise ValueError("Baseline and visual files have different sample IDs.")

    sample_ids = [r["sample_id"] for r in baseline_rows]

    y_true = []
    baseline_probabilities = []
    visual_probabilities = []

    for sample_id in sample_ids:
        base = baseline_by_id[sample_id]
        visual = visual_by_id[sample_id]

        if base["video_id"] != visual["video_id"]:
            raise ValueError(f"Video ID mismatch: {sample_id}")

        if base["true_label"] != visual["humor_label"]:
            raise ValueError(f"Label mismatch: {sample_id}")

        label = base["true_label"]
        if label not in {"humorous", "non_humorous"}:
            raise ValueError(f"Unexpected label for {sample_id}: {label}")

        y_true.append(int(label == "humorous"))
        baseline_probabilities.append(
            float(base["late_fusion_humorous_probability"])
        )
        visual_probabilities.append(
            float(visual[VISUAL_PROBABILITY_COLUMN])
        )

    if not all(
        0.0 <= p <= 1.0
        for p in baseline_probabilities + visual_probabilities
    ):
        raise ValueError("A probability is outside [0, 1].")

    results = {
        "text_audio_late_fusion_baseline": metrics(
            y_true, baseline_probabilities
        )
    }

    prediction_rows = []

    for i, sample_id in enumerate(sample_ids):
        prediction_rows.append({
            "sample_id": sample_id,
            "video_id": baseline_by_id[sample_id]["video_id"],
            "true_label": baseline_by_id[sample_id]["true_label"],
            "text_audio_probability": baseline_probabilities[i],
            "visual_probability": visual_probabilities[i],
        })

    for visual_weight in VISUAL_WEIGHTS:
        combined = [
            (1.0 - visual_weight) * baseline_probabilities[i]
            + visual_weight * visual_probabilities[i]
            for i in range(len(sample_ids))
        ]

        name = f"visual_weight_{visual_weight:.2f}"
        results[name] = metrics(y_true, combined)

        for i, row in enumerate(prediction_rows):
            row[f"{name}_probability"] = combined[i]
            row[f"{name}_prediction"] = (
                "humorous" if combined[i] >= 0.5 else "non_humorous"
            )

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(prediction_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    output = {
        "experiment": "Exploratory visual probability contribution to late fusion",
        "samples": len(sample_ids),
        "visual_configuration": "absolute_frame_difference",
        "visual_weights": VISUAL_WEIGHTS,
        "threshold": 0.5,
        "baseline_and_visual_samples_aligned": True,
        "limitations": [
            "Uses existing out-of-fold predictions; no independent test set.",
            "The visual configuration was selected after inspecting its results.",
            "Trying several weights introduces selection bias if the best score is chosen.",
            "A promising result requires a properly grouped nested evaluation.",
            "Probability averaging is late fusion, not learned cross-modal reasoning.",
        ],
        "results": results,
        "predictions_file": str(OUTPUT_CSV.relative_to(ROOT)),
    }

    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("VISUAL–TEXT/AUDIO FUSION RESULTS")
    print(json.dumps(output, indent=2))
    print(f"\nSaved metrics: {OUTPUT_JSON}")
    print(f"Saved predictions: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()