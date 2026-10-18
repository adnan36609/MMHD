"""Compare single-frame and cross-frame ViT representations for MMHD.

Uses frozen, locally extracted ViT embeddings.
Evaluation is restricted to the existing 391-sample primary subset.
All configurations use identical source-video-grouped folds.

This is a cross-frame feature experiment inspired by multimodal
reasoning research; it is not a reproduction of MMSD3.0's CIRM.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = ROOT / "annotations" / "expanded_540" / "samples_540.json"
ANNOTATIONS_FILE = ROOT / "annotations" / "expanded_540" / "annotations_540.json"

EMBEDDINGS_FILE = (
    ROOT / "results" / "expanded_540"
    / "vit_keyframe_embeddings" / "embeddings.csv"
)

EVALUATION_FILE = (
    ROOT / "results" / "expanded_540"
    / "multimodal_comparison" / "oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "results" / "expanded_540" / "cross_frame_vit"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "oof_predictions.csv"

N_SPLITS = 5
LOGISTIC_REGRESSION_C = 0.1


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_embeddings():
    rows = read_csv(EMBEDDINGS_FILE)
    embedding_columns = sorted(
        [c for c in rows[0] if c.startswith("embedding_")]
    )

    embeddings = {}

    for row in rows:
        if row["status"] != "ok":
            continue

        key = (row["sample_id"], int(row["frame_index"]))
        vector = np.asarray(
            [float(row[c]) for c in embedding_columns],
            dtype=np.float32,
        )

        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite embedding: {key}")

        if key in embeddings:
            raise ValueError(f"Duplicate embedding record: {key}")

        embeddings[key] = vector

    return embeddings, len(embedding_columns)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = read_json(SAMPLES_FILE)
    annotations = read_json(ANNOTATIONS_FILE)
    evaluation_rows = read_csv(EVALUATION_FILE)
    embeddings, embedding_dim = read_embeddings()

    sample_by_id = {s["sample_id"]: s for s in samples}
    label_by_id = {
        a["sample_id"]: a["humor_label"] for a in annotations
    }

    evaluation_ids = [r["sample_id"] for r in evaluation_rows]

    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Primary evaluation contains duplicate sample IDs.")

    missing_metadata = [
        sid for sid in evaluation_ids
        if sid not in sample_by_id or sid not in label_by_id
    ]
    if missing_metadata:
        raise ValueError(
            f"Missing metadata or labels for {len(missing_metadata)} samples."
        )

    missing_frames = [
        (sid, frame)
        for sid in evaluation_ids
        for frame in (1, 2)
        if (sid, frame) not in embeddings
    ]
    if missing_frames:
        raise ValueError(
            f"{len(missing_frames)} evaluation keyframes lack embeddings. "
            f"Examples: {missing_frames[:10]}"
        )

    labels_raw = [label_by_id[sid] for sid in evaluation_ids]
    allowed_labels = {"humorous", "non_humorous"}

    unexpected = set(labels_raw) - allowed_labels
    if unexpected:
        raise ValueError(f"Unexpected humor labels: {unexpected}")

    y = np.asarray(
        [1 if label == "humorous" else 0 for label in labels_raw],
        dtype=np.int32,
    )

    groups = np.asarray([
        sample_by_id[sid]["video_id"] for sid in evaluation_ids
    ])

    frame1 = np.stack([
        embeddings[(sid, 1)] for sid in evaluation_ids
    ])
    frame2 = np.stack([
        embeddings[(sid, 2)] for sid in evaluation_ids
    ])

    mean_frame = (frame1 + frame2) / 2.0
    absolute_difference = np.abs(frame2 - frame1)
    elementwise_product = frame1 * frame2

    configurations = {
        "frame_1_only": frame1,
        "frame_2_only": frame2,
        "mean_pooled_frames": mean_frame,
        "absolute_frame_difference": absolute_difference,
        "ordered_frame_pair": np.concatenate(
            [frame1, frame2], axis=1
        ),
        "cross_frame_relations": np.concatenate(
            [
                mean_frame,
                absolute_difference,
                elementwise_product,
            ],
            axis=1,
        ),
    }

    # Construct folds once and reuse the identical folds for every model.
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(frame1, y, groups))

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        if train_groups & test_groups:
            raise RuntimeError(
                f"Source-video leakage detected in fold {fold_number}."
            )

    results = {}
    prediction_columns = {}
    fold_assignment = np.full(len(evaluation_ids), -1, dtype=int)

    for fold_number, (_, test_idx) in enumerate(splits, start=1):
        fold_assignment[test_idx] = fold_number

    for name, features in configurations.items():
        print(
            f"Evaluating {name}: "
            f"{features.shape[0]} samples x {features.shape[1]} features"
        )

        oof_probability = np.full(len(y), np.nan, dtype=np.float64)

        for train_idx, test_idx in splits:
            # Scaling is fitted using the training fold only.
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=LOGISTIC_REGRESSION_C,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42,
                ),
            )

            classifier.fit(features[train_idx], y[train_idx])
            oof_probability[test_idx] = classifier.predict_proba(
                features[test_idx]
            )[:, 1]

        if not np.isfinite(oof_probability).all():
            raise RuntimeError(f"Incomplete OOF predictions for {name}.")

        oof_prediction = (oof_probability >= 0.5).astype(int)

        metrics = {
            "macro_f1": float(f1_score(
                y, oof_prediction, average="macro", zero_division=0
            )),
            "accuracy": float(accuracy_score(y, oof_prediction)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y, oof_prediction)
            ),
            "roc_auc_humorous": float(roc_auc_score(y, oof_probability)),
            "feature_dimensions": int(features.shape[1]),
        }

        results[name] = metrics
        prediction_columns[name] = {
            "prediction": oof_prediction,
            "probability": oof_probability,
        }

    output_rows = []

    for index, sample_id in enumerate(evaluation_ids):
        row = {
            "sample_id": sample_id,
            "video_id": groups[index],
            "humor_label": labels_raw[index],
            "fold": int(fold_assignment[index]),
        }

        for name, values in prediction_columns.items():
            row[f"{name}_prediction"] = (
                "humorous"
                if values["prediction"][index] == 1
                else "non_humorous"
            )
            row[f"{name}_humorous_probability"] = float(
                values["probability"][index]
            )

        output_rows.append(row)

    with PREDICTIONS_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    metrics_output = {
        "experiment": "Frozen ViT single-frame versus cross-frame features",
        "inspiration": (
            "Cross-image reasoning direction from MMSD3.0; "
            "not a CIRM reproduction."
        ),
        "samples": len(evaluation_ids),
        "source_videos": len(set(groups)),
        "class_counts": dict(Counter(labels_raw)),
        "embedding_model": "google/vit-base-patch16-224-in21k",
        "embedding_dimensions": embedding_dim,
        "folds": N_SPLITS,
        "split_method": "GroupKFold grouped by source video",
        "folds_share_no_source_videos": True,
        "classifier": "StandardScaler + balanced LogisticRegression",
        "regularization_C": LOGISTIC_REGRESSION_C,
        "threshold": 0.5,
        "hyperparameter_tuning": False,
        "results": results,
        "interpretation_warning": (
            "Exploratory experiment on a small labeled subset. "
            "Cross-frame feature engineering is not learned cross-image "
            "attention. Results do not establish causal reasoning or "
            "reproduce the MMSD3.0 CIRM architecture."
        ),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics_output, indent=2),
        encoding="utf-8",
    )

    print("\nCROSS-FRAME EXPERIMENT RESULTS")
    print(json.dumps(metrics_output, indent=2))
    print(f"\nSaved metrics: {METRICS_FILE}")
    print(f"Saved OOF predictions: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()