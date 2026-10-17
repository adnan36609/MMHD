import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = ROOT / "annotations" / "expanded_540" / "samples_540.json"
ANNOTATIONS_FILE = ROOT / "annotations" / "expanded_540" / "annotations_540.json"
SAMPLE_ROOT = ROOT / "sample_clips"

OUTPUT_DIR = (
    ROOT / "results" / "expanded_540" / "semi_supervised_minilm"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

N_SPLITS = 5
RANDOM_STATE = 42
BATCH_SIZE = 32
MAX_LENGTH = 256

CONFIDENCE_THRESHOLD = 0.90
MAX_PSEUDO_MULTIPLIER = 2

LABEL_TO_ID = {
    "non_humorous": 0,
    "humorous": 1,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def normalize_text(text):
    return " ".join(str(text or "").split())


def text_key(text):
    return normalize_text(text).casefold()


def load_labeled_records():
    samples = read_json(SAMPLES_FILE)
    annotations = read_json(ANNOTATIONS_FILE)

    samples_by_id = {
        str(sample["sample_id"]): sample
        for sample in samples
    }

    annotated_ids = set()
    records = []
    seen_annotations = set()

    # Preserve the existing experiment's annotation ordering.
    for annotation in annotations:
        sample_id = str(annotation.get("sample_id") or "").strip()
        label = str(annotation.get("humor_label") or "").strip()

        if not sample_id:
            raise ValueError("Annotation missing sample_id.")

        if sample_id in seen_annotations:
            raise ValueError(f"Duplicate annotation: {sample_id}")

        seen_annotations.add(sample_id)
        annotated_ids.add(sample_id)

        if label not in LABEL_TO_ID:
            raise ValueError(
                f"Unexpected label {label!r} for {sample_id}"
            )

        if sample_id not in samples_by_id:
            raise ValueError(f"Missing sample record: {sample_id}")

        sample = samples_by_id[sample_id]
        text = normalize_text(sample.get("text"))
        video_id = str(sample.get("video_id") or "").strip()

        if not text:
            continue

        if not video_id:
            raise ValueError(f"Missing video_id for {sample_id}")

        records.append({
            "sample_id": sample_id,
            "video_id": video_id,
            "text": text,
            "label": label,
            "y": LABEL_TO_ID[label],
        })

    if len(records) < N_SPLITS:
        raise ValueError("Not enough labelled text samples.")

    if len(set(r["y"] for r in records)) != 2:
        raise ValueError("Both classes must be represented.")

    if len(set(r["video_id"] for r in records)) < N_SPLITS:
        raise ValueError("At least five source videos are required.")

    return records, annotated_ids, len(samples), len(annotations)


def load_unlabeled_corpus(annotated_ids):
    """Read transcripts only from the canonical dataset manifests."""
    records = []
    seen_ids = set()
    manifest_count = 0

    for path in sorted(SAMPLE_ROOT.glob("*/dataset.json")):
        data = read_json(path)

        if not isinstance(data, list):
            raise ValueError(f"Unexpected dataset format: {path}")

        manifest_count += len(data)

        for sample in data:
            sample_id = str(sample.get("sample_id") or "").strip()

            if not sample_id:
                continue

            if sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate corpus sample_id: {sample_id}"
                )

            seen_ids.add(sample_id)

            # Never turn any human-annotated sample into an
            # unlabeled pseudo-training example.
            if sample_id in annotated_ids:
                continue

            source = sample.get("source") or {}
            modalities = sample.get("modalities") or {}
            text_data = modalities.get("text") or {}

            text = normalize_text(text_data.get("value"))
            video_id = str(source.get("video_id") or "").strip()

            if not text or not video_id:
                continue

            records.append({
                "sample_id": sample_id,
                "video_id": video_id,
                "text": text,
            })

    if not records:
        raise ValueError(
            "No unlabeled transcripts found in dataset manifests."
        )

    return records, manifest_count


def encode_texts(texts):
    """Generate frozen MiniLM embeddings on the CPU."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    model.to("cpu")
    model.eval()

    output_vectors = []

    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            result = model(**encoded)
            token_embeddings = result.last_hidden_state

            mask = encoded["attention_mask"].unsqueeze(-1).to(
                token_embeddings.dtype
            )

            pooled = (token_embeddings * mask).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, p=2, dim=1)

            output_vectors.append(pooled.cpu().numpy())

    del model
    del tokenizer

    return np.concatenate(output_vectors, axis=0)


def build_classifier():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )


def positive_probability(model, features):
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(features)[:, positive_index]


def calculate_metrics(y_true, predictions, probabilities):
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
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "precision_humorous": float(
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_humorous": float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_humorous": float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "roc_auc_humorous": float(
            roc_auc_score(y_true, probabilities)
        ),
    }


def main():
    start_time = time.perf_counter()

    labeled, annotated_ids, total_samples, total_annotations = (
        load_labeled_records()
    )

    unlabeled, manifest_count = load_unlabeled_corpus(
        annotated_ids
    )

    texts = [row["text"] for row in labeled]
    labels = np.asarray(
        [row["y"] for row in labeled],
        dtype=int,
    )
    groups = np.asarray(
        [row["video_id"] for row in labeled],
        dtype=object,
    )

    unlabeled_texts = [row["text"] for row in unlabeled]
    unlabeled_groups = np.asarray(
        [row["video_id"] for row in unlabeled],
        dtype=object,
    )

    print("\n=== MMHD Semi-Supervised MiniLM ===")
    print(f"Manifest records: {manifest_count}")
    print(f"Annotation samples: {total_annotations}")
    print(f"Samples in annotation file: {total_samples}")
    print(f"Labelled samples with text: {len(labeled)}")
    print(f"Unlabelled transcripts available: {len(unlabeled)}")
    print(f"Label counts: {dict(Counter(labels))}")
    print(f"Source videos: {len(set(groups))}")
    print(f"Evaluation: {N_SPLITS}-fold GroupKFold by video")
    print(f"Pseudo-label confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(
        "Maximum pseudo-labels per class: "
        f"{MAX_PSEUDO_MULTIPLIER}x that class's labelled training count"
    )

    # Embeddings are generated once. Labels and pseudo-labels are
    # still restricted separately within each cross-validation fold.
    print("\nGenerating labelled and unlabelled embeddings...")
    embedding_start = time.perf_counter()

    all_embeddings = encode_texts(texts + unlabeled_texts)

    labeled_embeddings = all_embeddings[:len(labeled)]
    unlabeled_embeddings = all_embeddings[len(labeled):]

    del all_embeddings

    print(f"Embedding time: {time.perf_counter() - embedding_start:.2f}s")
    print(f"Labeled embedding shape: {labeled_embeddings.shape}")
    print(f"Unlabeled embedding shape: {unlabeled_embeddings.shape}")

    count = len(labeled)

    supervised_predictions = np.full(count, -1, dtype=int)
    supervised_probabilities = np.full(count, np.nan)

    semi_predictions = np.full(count, -1, dtype=int)
    semi_probabilities = np.full(count, np.nan)

    folds = np.zeros(count, dtype=int)
    fold_results = []

    splitter = GroupKFold(n_splits=N_SPLITS)

    print("\nRunning video-grouped cross-validation...")

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(labeled_embeddings, labels, groups),
        start=1,
    ):
        train_videos = set(groups[train_idx])
        test_videos = set(groups[test_idx])

        if train_videos & test_videos:
            raise RuntimeError(f"Video leakage in fold {fold}.")

        x_train = labeled_embeddings[train_idx]
        y_train = labels[train_idx]
        x_test = labeled_embeddings[test_idx]
        y_test = labels[test_idx]

        if set(y_train) != {0, 1}:
            raise ValueError(f"Training fold {fold} lacks a class.")

        # Supervised baseline.
        supervised = build_classifier()
        supervised.fit(x_train, y_train)

        baseline_pred = supervised.predict(x_test)
        baseline_prob = positive_probability(supervised, x_test)

        # Use unlabeled transcripts from training videos only.
        # Also exclude exact-normalized transcript duplicates of
        # this fold's held-out samples from the pseudo-label pool.
        test_text_keys = {
            text_key(texts[i])
            for i in test_idx
        }

        pool_idx = [
            i for i, row in enumerate(unlabeled)
            if unlabeled_groups[i] in train_videos
            and text_key(row["text"]) not in test_text_keys
        ]

        if pool_idx:
            x_pool = unlabeled_embeddings[pool_idx]

            if len(set(unlabeled_groups[pool_idx]) & test_videos):
                raise RuntimeError(
                    f"Unlabeled video leakage in fold {fold}."
                )

            pool_probabilities = supervised.predict_proba(x_pool)
            pool_predictions = supervised.predict(x_pool)
            pool_confidence = pool_probabilities.max(axis=1)

            selected_pool_positions = []
            selected_pseudo_labels = []

            # Confidence filtering and per-class caps prevent the
            # unlabelled corpus from overwhelming human annotations.
            for class_id in (0, 1):
                true_class_count = int(
                    np.sum(y_train == class_id)
                )
                cap = MAX_PSEUDO_MULTIPLIER * true_class_count

                candidates = np.flatnonzero(
                    (pool_predictions == class_id)
                    & (pool_confidence >= CONFIDENCE_THRESHOLD)
                )

                # Keep the highest-confidence candidates for this class.
                ranked = candidates[
                    np.argsort(pool_confidence[candidates])[::-1]
                ]

                selected = ranked[:cap]

                selected_pool_positions.extend(selected.tolist())
                selected_pseudo_labels.extend(
                    [class_id] * len(selected)
                )

            selected_pool_positions = np.asarray(
                selected_pool_positions,
                dtype=int,
            )
            selected_pseudo_labels = np.asarray(
                selected_pseudo_labels,
                dtype=int,
            )

        else:
            x_pool = np.empty((0, labeled_embeddings.shape[1]))
            selected_pool_positions = np.asarray([], dtype=int)
            selected_pseudo_labels = np.asarray([], dtype=int)

        # One round of confidence-threshold pseudo-labelling.
        if len(selected_pool_positions):
            x_augmented = np.vstack([
                x_train,
                x_pool[selected_pool_positions],
            ])
            y_augmented = np.concatenate([
                y_train,
                selected_pseudo_labels,
            ])
        else:
            x_augmented = x_train
            y_augmented = y_train

        semi_supervised = build_classifier()
        semi_supervised.fit(x_augmented, y_augmented)

        semi_pred = semi_supervised.predict(x_test)
        semi_prob = positive_probability(
            semi_supervised,
            x_test,
        )

        supervised_predictions[test_idx] = baseline_pred
        supervised_probabilities[test_idx] = baseline_prob

        semi_predictions[test_idx] = semi_pred
        semi_probabilities[test_idx] = semi_prob

        folds[test_idx] = fold

        baseline_metrics = calculate_metrics(
            y_test,
            baseline_pred,
            baseline_prob,
        )
        semi_metrics = calculate_metrics(
            y_test,
            semi_pred,
            semi_prob,
        )

        pseudo_counts = {
            "non_humorous": int(np.sum(selected_pseudo_labels == 0)),
            "humorous": int(np.sum(selected_pseudo_labels == 1)),
        }

        fold_results.append({
            "fold": fold,
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "train_videos": int(len(train_videos)),
            "test_videos": int(len(test_videos)),
            "unlabeled_pool_size": int(len(pool_idx)),
            "pseudo_labels_selected": int(len(selected_pool_positions)),
            "pseudo_label_counts": pseudo_counts,
            "supervised_minilm": baseline_metrics,
            "semi_supervised_minilm": semi_metrics,
        })

        print(
            f"Fold {fold}/{N_SPLITS} | "
            f"training videos={len(train_videos)} | "
            f"unlabeled pool={len(pool_idx)} | "
            f"pseudo-labels={len(selected_pool_positions)} "
            f"(non-humorous={pseudo_counts['non_humorous']}, "
            f"humorous={pseudo_counts['humorous']}) | "
            f"supervised macro-F1={baseline_metrics['macro_f1']:.4f} | "
            f"semi-supervised macro-F1={semi_metrics['macro_f1']:.4f}"
        )

    if (
        np.any(supervised_predictions < 0)
        or np.any(semi_predictions < 0)
        or np.isnan(supervised_probabilities).any()
        or np.isnan(semi_probabilities).any()
        or np.any(folds == 0)
    ):
        raise RuntimeError("Incomplete out-of-fold predictions.")

    supervised_metrics = calculate_metrics(
        labels,
        supervised_predictions,
        supervised_probabilities,
    )
    semi_metrics = calculate_metrics(
        labels,
        semi_predictions,
        semi_probabilities,
    )

    deltas = {
        metric: float(semi_metrics[metric] - supervised_metrics[metric])
        for metric in supervised_metrics
    }

    print("\n=== OUT-OF-FOLD RESULTS ===")
    print(
        f"{'Metric':<24}{'Supervised':>12}"
        f"{'Semi-supervised':>17}{'Delta':>10}"
    )
    print("-" * 63)

    for metric in supervised_metrics:
        print(
            f"{metric:<24}"
            f"{supervised_metrics[metric]:>12.4f}"
            f"{semi_metrics[metric]:>17.4f}"
            f"{deltas[metric]:>+10.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame({
        "sample_id": [row["sample_id"] for row in labeled],
        "video_id": groups,
        "true_label": [row["label"] for row in labeled],
        "fold": folds,
        "supervised_minilm_prediction": [
            ID_TO_LABEL[v] for v in supervised_predictions
        ],
        "supervised_minilm_probability_humorous": (
            supervised_probabilities
        ),
        "semi_supervised_minilm_prediction": [
            ID_TO_LABEL[v] for v in semi_predictions
        ],
        "semi_supervised_minilm_probability_humorous": (
            semi_probabilities
        ),
    })

    predictions_path = OUTPUT_DIR / "oof_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    results = {
        "experiment": "One-round confidence-threshold pseudo-labeling",
        "encoder": MODEL_NAME,
        "encoder_trainable": False,
        "classifier": "class-weighted LogisticRegression",
        "annotation_samples": int(total_annotations),
        "available_sample_records": int(total_samples),
        "manifest_records": int(manifest_count),
        "labeled_text_samples": int(len(labeled)),
        "unlabeled_transcripts_available": int(len(unlabeled)),
        "distinct_labeled_videos": int(len(set(groups))),
        "evaluation": f"{N_SPLITS}-fold GroupKFold by source video",
        "video_leakage_check": "passed",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "maximum_pseudo_label_multiplier": MAX_PSEUDO_MULTIPLIER,
        "pseudo_label_policy": (
            "Within each fold, use only unlabeled records from training "
            "videos; exclude exact-normalized held-out transcript matches; "
            "select probabilities above the fixed threshold; cap selected "
            "pseudo-labels per class at twice its human-labelled training "
            "count."
        ),
        "supervised_minilm_oof_metrics": supervised_metrics,
        "semi_supervised_minilm_oof_metrics": semi_metrics,
        "delta_semi_minus_supervised": deltas,
        "fold_results": fold_results,
        "runtime_seconds": round(time.perf_counter() - start_time, 2),
    }

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved metrics: {metrics_path.relative_to(ROOT)}")
    print(f"Saved predictions: {predictions_path.relative_to(ROOT)}")
    print(f"Total runtime: {results['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()