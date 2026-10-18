"""Compare MMHD models on the same 391 samples.

Uses saved out-of-fold predictions only.
Computes paired, source-video-clustered bootstrap intervals.
Does not retrain models or modify existing results.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

BASELINE_FILE = (
    ROOT / "results/expanded_540/late_fusion/oof_predictions.csv"
)
CONTEXT_FILE = (
    ROOT / "results/expanded_540/context_experiment/oof_predictions.csv"
)
ATTENTION_FILE = (
    ROOT / "results/expanded_540/cross_modal_reasoning/oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "results/expanded_540/model_comparison_391"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPLICATES = 1000
SEED = 42


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def compute_metrics(y, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "macro_f1": float(
            f1_score(y, predictions, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def main():
    baseline_rows = read_csv(BASELINE_FILE)
    context_rows = read_csv(CONTEXT_FILE)
    attention_rows = read_csv(ATTENTION_FILE)

    baseline = {r["sample_id"]: r for r in baseline_rows}
    context = {r["sample_id"]: r for r in context_rows}
    attention = {r["sample_id"]: r for r in attention_rows}

    if (
        len(baseline) != len(baseline_rows)
        or len(context) != len(context_rows)
        or len(attention) != len(attention_rows)
    ):
        raise ValueError("Duplicate sample IDs detected.")

    common_ids = sorted(
        set(baseline) & set(context) & set(attention)
    )

    if len(common_ids) != 391:
        raise ValueError(
            f"Expected 391 common samples; found {len(common_ids)}."
        )

    y = []
    groups = []
    probabilities = {
        "text_audio_late_fusion": [],
        "neighbor_context_text": [],
        "text_visual_attention": [],
    }

    output_rows = []

    for sample_id in common_ids:
        b = baseline[sample_id]
        c = context[sample_id]
        a = attention[sample_id]

        labels = {
            b["true_label"],
            c["true_label"],
            a["true_label"],
        }
        videos = {
            b["video_id"],
            c["video_id"],
            a["video_id"],
        }

        if len(labels) != 1:
            raise ValueError(f"Label mismatch: {sample_id}")
        if len(videos) != 1:
            raise ValueError(f"Video mismatch: {sample_id}")

        label = labels.pop()
        if label not in {"humorous", "non_humorous"}:
            raise ValueError(f"Unexpected label: {label}")

        y.append(int(label == "humorous"))
        groups.append(videos.pop())

        row = {
            "sample_id": sample_id,
            "video_id": groups[-1],
            "true_label": label,
        }

        column_map = {
            "text_audio_late_fusion":
                float(b["late_fusion_humorous_probability"]),
            "neighbor_context_text":
                float(c["context_model_humorous_probability"]),
            "text_visual_attention":
                float(a["cross_modal_probability"]),
        }

        for name, probability in column_map.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"Invalid probability for {sample_id}: {name}"
                )

            probabilities[name].append(probability)
            row[f"{name}_probability"] = probability
            row[f"{name}_prediction"] = (
                "humorous" if probability >= 0.5 else "non_humorous"
            )

        output_rows.append(row)

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    probabilities = {
        name: np.asarray(values, dtype=float)
        for name, values in probabilities.items()
    }

    if len(set(groups)) != 24:
        raise ValueError(
            f"Expected 24 source videos; found {len(set(groups))}."
        )

    metrics = {
        name: compute_metrics(y, values)
        for name, values in probabilities.items()
    }

    # Cluster bootstrap: resample entire source videos, not individual rows.
    rng = np.random.default_rng(SEED)
    unique_groups = np.unique(groups)
    group_indices = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }

    bootstrap_differences = {
        name: [] for name in probabilities
        if name != "text_audio_late_fusion"
    }

    for _ in range(BOOTSTRAP_REPLICATES):
        selected_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )

        indices = np.concatenate([
            group_indices[group] for group in selected_groups
        ])

        if len(np.unique(y[indices])) < 2:
            continue

        baseline_score = f1_score(
            y[indices],
            (probabilities["text_audio_late_fusion"][indices] >= 0.5)
            .astype(int),
            average="macro",
            zero_division=0,
        )

        for name in bootstrap_differences:
            model_score = f1_score(
                y[indices],
                (probabilities[name][indices] >= 0.5).astype(int),
                average="macro",
                zero_division=0,
            )
            bootstrap_differences[name].append(
                float(model_score - baseline_score)
            )

    comparisons = {}

    for name, differences in bootstrap_differences.items():
        if not differences:
            raise RuntimeError(f"No valid bootstrap replicates for {name}.")

        comparisons[name] = {
            "macro_f1_difference_vs_text_audio": float(
                metrics[name]["macro_f1"]
                - metrics["text_audio_late_fusion"]["macro_f1"]
            ),
            "cluster_bootstrap_95_percent_interval": [
                float(np.percentile(differences, 2.5)),
                float(np.percentile(differences, 97.5)),
            ],
            "valid_bootstrap_replicates": len(differences),
        }

    with (OUTPUT_DIR / "oof_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(output_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(output_rows)

    result = {
        "experiment": "Paired comparison of saved OOF predictions",
        "samples": len(common_ids),
        "source_videos": len(set(groups)),
        "bootstrap_unit": "source video",
        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
        "metrics": metrics,
        "comparisons_vs_text_audio": comparisons,
        "limitations": [
            "Small evaluation subset from 24 source videos.",
            "Predictions were generated by independently evaluated pipelines.",
            "Context model and other models may have different fold assignments.",
            "Bootstrap intervals quantify uncertainty on these saved predictions; "
            "they do not establish external generalization.",
            "No hyperparameter selection or fusion weights are tuned here.",
        ],
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("PAIRED 391-SAMPLE COMPARISON")
    print(json.dumps(result, indent=2))
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()