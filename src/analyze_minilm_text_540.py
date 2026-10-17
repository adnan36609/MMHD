import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "expanded_540"
    / "semantic_text_comparison"
    / "oof_predictions.csv"
)

OUTPUT_DIR = INPUT_FILE.parent
BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 42

LABELS = ["non_humorous", "humorous"]
POSITIVE_LABEL = "humorous"

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision_humorous",
    "recall_humorous",
    "f1_humorous",
    "roc_auc_humorous",
]


def calculate_metrics(y_true, predictions, probabilities):
    binary_true = np.asarray(
        [1 if label == POSITIVE_LABEL else 0 for label in y_true]
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_humorous": float(
            precision_score(
                y_true,
                predictions,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "recall_humorous": float(
            recall_score(
                y_true,
                predictions,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "f1_humorous": float(
            f1_score(
                y_true,
                predictions,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "roc_auc_humorous": float(
            roc_auc_score(binary_true, probabilities)
        ),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = {
        "video_id",
        "true_label",
        "tfidf_prediction",
        "tfidf_probability_humorous",
        "minilm_prediction",
        "minilm_probability_humorous",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    if not set(df["true_label"]).issubset(LABELS):
        raise ValueError("Unexpected true labels.")

    video_ids = df["video_id"].astype(str).to_numpy()
    y_true = df["true_label"].to_numpy()

    tfidf_pred = df["tfidf_prediction"].to_numpy()
    minilm_pred = df["minilm_prediction"].to_numpy()

    tfidf_prob = df[
        "tfidf_probability_humorous"
    ].to_numpy(dtype=float)

    minilm_prob = df[
        "minilm_probability_humorous"
    ].to_numpy(dtype=float)

    if not np.isfinite(tfidf_prob).all():
        raise ValueError("Invalid TF-IDF probabilities.")

    if not np.isfinite(minilm_prob).all():
        raise ValueError("Invalid MiniLM probabilities.")

    videos = np.unique(video_ids)

    if len(videos) < 2:
        raise ValueError("At least two source videos are required.")

    # Point estimates using all out-of-fold predictions.
    tfidf_metrics = calculate_metrics(
        y_true, tfidf_pred, tfidf_prob
    )
    minilm_metrics = calculate_metrics(
        y_true, minilm_pred, minilm_prob
    )

    # Keep every source video's samples together during resampling.
    video_indices = {
        video: np.flatnonzero(video_ids == video)
        for video in videos
    }

    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_deltas = {metric: [] for metric in METRICS}

    valid = 0
    attempts = 0
    max_attempts = BOOTSTRAP_ITERATIONS * 10

    while valid < BOOTSTRAP_ITERATIONS:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Too many bootstrap draws lacked both classes."
            )

        # Draw the same number of videos, with replacement.
        sampled_videos = rng.choice(
            videos,
            size=len(videos),
            replace=True,
        )

        # Repeatedly drawn videos contribute their samples repeatedly.
        indices = np.concatenate(
            [video_indices[video] for video in sampled_videos]
        )

        y_boot = y_true[indices]

        # ROC-AUC requires both classes. Reject a draw lacking either.
        if len(np.unique(y_boot)) != 2:
            continue

        tfidf_boot = calculate_metrics(
            y_boot,
            tfidf_pred[indices],
            tfidf_prob[indices],
        )

        minilm_boot = calculate_metrics(
            y_boot,
            minilm_pred[indices],
            minilm_prob[indices],
        )

        for metric in METRICS:
            bootstrap_deltas[metric].append(
                minilm_boot[metric] - tfidf_boot[metric]
            )

        valid += 1

    summary = {
        "experiment": "Paired source-video cluster bootstrap",
        "input": str(INPUT_FILE.relative_to(ROOT)),
        "samples": int(len(df)),
        "source_videos": int(len(videos)),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "valid_iterations": valid,
        "seed": RANDOM_SEED,
        "resampling_unit": "source video",
        "confidence_interval": "95% percentile interval",
        "tfidf_point_estimates": tfidf_metrics,
        "minilm_point_estimates": minilm_metrics,
        "differences_minilm_minus_tfidf": {},
    }

    print("\n=== Paired Video-Cluster Bootstrap ===")
    print(f"Samples: {len(df)}")
    print(f"Source videos: {len(videos)}")
    print(f"Valid bootstrap draws: {valid}")

    print(
        f"\n{'Metric':<24}"
        f"{'TF-IDF':>10}"
        f"{'MiniLM':>10}"
        f"{'Delta':>10}"
        f"{'95% CI':>25}"
    )
    print("-" * 79)

    for metric in METRICS:
        deltas = np.asarray(bootstrap_deltas[metric])

        low, high = np.percentile(deltas, [2.5, 97.5])
        point_delta = minilm_metrics[metric] - tfidf_metrics[metric]
        fraction_positive = float(np.mean(deltas > 0))

        summary["differences_minilm_minus_tfidf"][metric] = {
            "point_difference": float(point_delta),
            "ci_95_lower": float(low),
            "ci_95_upper": float(high),
            "fraction_bootstrap_deltas_positive": fraction_positive,
            "bootstrap_median_difference": float(np.median(deltas)),
        }

        interval = f"[{low:+.4f}, {high:+.4f}]"

        print(
            f"{metric:<24}"
            f"{tfidf_metrics[metric]:>10.4f}"
            f"{minilm_metrics[metric]:>10.4f}"
            f"{point_delta:>+10.4f}"
            f"{interval:>25}"
        )

    output_path = OUTPUT_DIR / "bootstrap_analysis.json"

    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {output_path.relative_to(ROOT)}")
    print(
        "\nInterpretation: an interval containing zero means "
        "the direction of the difference remains uncertain "
        "under this bootstrap analysis."
    )
    print(
        "The bootstrap is an uncertainty analysis, not proof "
        "that one model is universally superior."
    )


if __name__ == "__main__":
    main()