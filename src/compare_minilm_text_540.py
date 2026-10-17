import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = ROOT / "annotations" / "expanded_540" / "samples_540.json"
ANNOTATIONS_FILE = ROOT / "annotations" / "expanded_540" / "annotations_540.json"

OUTPUT_DIR = ROOT / "results" / "expanded_540" / "semantic_text_comparison"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

N_SPLITS = 5
RANDOM_STATE = 42
BATCH_SIZE = 32
MAX_LENGTH = 256

LABELS = ("non_humorous", "humorous")
POSITIVE_LABEL = "humorous"


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_dataset():
    samples = read_json(SAMPLES_FILE)
    annotations = read_json(ANNOTATIONS_FILE)

    samples_by_id = {}

    for sample in samples:
        sample_id = str(sample.get("sample_id") or "").strip()

        if not sample_id:
            raise ValueError("A sample is missing its sample_id.")

        if sample_id in samples_by_id:
            raise ValueError(f"Duplicate sample_id: {sample_id}")

        samples_by_id[sample_id] = sample

    records = []
    seen_annotations = set()
    missing_samples = []
    unusable_labels = []

    # Preserve annotation order to match the existing experiment's
    # likely GroupKFold input ordering.
    for annotation in annotations:
        sample_id = str(annotation.get("sample_id") or "").strip()

        if not sample_id:
            raise ValueError("An annotation is missing its sample_id.")

        if sample_id in seen_annotations:
            raise ValueError(f"Duplicate annotation: {sample_id}")

        seen_annotations.add(sample_id)

        if sample_id not in samples_by_id:
            missing_samples.append(sample_id)
            continue

        label = str(annotation.get("humor_label") or "").strip()

        if label not in LABELS:
            unusable_labels.append((sample_id, label))
            continue

        sample = samples_by_id[sample_id]

        text = " ".join(str(sample.get("text") or "").split())
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
        })

    if missing_samples:
        raise ValueError(
            f"{len(missing_samples)} annotations have no matching sample."
        )

    if unusable_labels:
        raise ValueError(
            f"Unexpected annotation labels: {unusable_labels[:10]}"
        )

    if len(records) < N_SPLITS:
        raise ValueError("Not enough text samples for cross-validation.")

    if set(row["label"] for row in records) != set(LABELS):
        raise ValueError("The usable dataset must contain both classes.")

    if len(set(row["video_id"] for row in records)) < N_SPLITS:
        raise ValueError("At least five distinct source videos are required.")

    return records, len(samples), len(annotations)


def build_tfidf_model():
    """Use the existing TF-IDF baseline configuration."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def encode_texts(texts, tokenizer, model):
    """Generate frozen, mean-pooled MiniLM sentence embeddings."""
    embeddings = []

    model.eval()

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

            output = model(**encoded)
            token_embeddings = output.last_hidden_state

            mask = (
                encoded["attention_mask"]
                .unsqueeze(-1)
                .to(token_embeddings.dtype)
            )

            pooled = (token_embeddings * mask).sum(dim=1)
            pooled /= mask.sum(dim=1).clamp(min=1)

            # Normalize each sentence vector before classification.
            pooled = F.normalize(pooled, p=2, dim=1)

            embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def calculate_metrics(y_true, y_pred, probabilities):
    """Calculate metrics with humorous as the positive class."""
    binary_true = [
        1 if label == POSITIVE_LABEL else 0
        for label in y_true
    ]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_humorous": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "recall_humorous": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "f1_humorous": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "roc_auc_humorous": float(
            roc_auc_score(binary_true, probabilities)
        ),
    }


def positive_probabilities(classifier, features):
    """Extract probabilities for the humorous class."""
    positive_index = list(classifier.classes_).index(POSITIVE_LABEL)
    return classifier.predict_proba(features)[:, positive_index]


def main():
    start_time = time.perf_counter()

    records, total_samples, total_annotations = load_dataset()

    texts = [row["text"] for row in records]
    labels = np.asarray([row["label"] for row in records], dtype=object)
    groups = np.asarray([row["video_id"] for row in records], dtype=object)

    print("\n=== MMHD: TF-IDF vs MiniLM ===")
    print(f"Annotation samples: {total_annotations}")
    print(f"Available sample records: {total_samples}")
    print(f"Samples with usable text: {len(records)}")
    print(f"Excluded for empty text: {total_annotations - len(records)}")
    print(f"Distinct source videos: {len(set(groups))}")
    print(f"Label counts: {dict(Counter(labels))}")
    print(f"Evaluation: {N_SPLITS}-fold GroupKFold by video_id")

    print("\nLoading cached MiniLM model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    encoder = AutoModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    encoder.to("cpu")
    encoder.eval()

    print("Generating frozen MiniLM embeddings...")
    embedding_start = time.perf_counter()
    embeddings = encode_texts(texts, tokenizer, encoder)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding time: {time.perf_counter() - embedding_start:.2f}s")

    del encoder
    del tokenizer

    n = len(records)

    tfidf_predictions = np.empty(n, dtype=object)
    minilm_predictions = np.empty(n, dtype=object)

    tfidf_probabilities = np.full(n, np.nan, dtype=float)
    minilm_probabilities = np.full(n, np.nan, dtype=float)
    oof_folds = np.zeros(n, dtype=int)

    fold_results = []
    splitter = GroupKFold(n_splits=N_SPLITS)

    print("\nRunning grouped cross-validation...")

    for fold_number, (train_idx, test_idx) in enumerate(
        splitter.split(texts, labels, groups),
        start=1,
    ):
        train_videos = set(groups[train_idx])
        test_videos = set(groups[test_idx])

        if train_videos & test_videos:
            raise RuntimeError(
                f"Video leakage detected in fold {fold_number}."
            )

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        if set(y_train) != set(LABELS):
            raise ValueError(
                f"Training fold {fold_number} is missing a class."
            )

        # Model 1: existing TF-IDF baseline.
        tfidf_model = build_tfidf_model()

        tfidf_model.fit(
            [texts[i] for i in train_idx],
            y_train,
        )

        tfidf_pred = tfidf_model.predict(
            [texts[i] for i in test_idx]
        )

        tfidf_prob = positive_probabilities(
            tfidf_model.named_steps["classifier"],
            tfidf_model.named_steps["tfidf"].transform(
                [texts[i] for i in test_idx]
            ),
        )

        # Model 2: frozen MiniLM embeddings with Logistic Regression.
        minilm_model = LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )

        minilm_model.fit(
            embeddings[train_idx],
            y_train,
        )

        minilm_pred = minilm_model.predict(embeddings[test_idx])

        minilm_prob = positive_probabilities(
            minilm_model,
            embeddings[test_idx],
        )

        tfidf_predictions[test_idx] = tfidf_pred
        minilm_predictions[test_idx] = minilm_pred

        tfidf_probabilities[test_idx] = tfidf_prob
        minilm_probabilities[test_idx] = minilm_prob
        oof_folds[test_idx] = fold_number

        tfidf_fold_metrics = calculate_metrics(
            y_test,
            tfidf_pred,
            tfidf_prob,
        )

        minilm_fold_metrics = calculate_metrics(
            y_test,
            minilm_pred,
            minilm_prob,
        )

        fold_results.append({
            "fold": fold_number,
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "train_videos": int(len(train_videos)),
            "test_videos": int(len(test_videos)),
            "tfidf": tfidf_fold_metrics,
            "minilm": minilm_fold_metrics,
        })

        print(
            f"Fold {fold_number}/{N_SPLITS} | "
            f"test={len(test_idx)} | "
            f"TF-IDF macro-F1={tfidf_fold_metrics['macro_f1']:.4f} | "
            f"MiniLM macro-F1={minilm_fold_metrics['macro_f1']:.4f}"
        )

    if np.isnan(tfidf_probabilities).any():
        raise RuntimeError("Some TF-IDF out-of-fold predictions are missing.")

    if np.isnan(minilm_probabilities).any():
        raise RuntimeError("Some MiniLM out-of-fold predictions are missing.")

    if np.any(oof_folds == 0):
        raise RuntimeError("Some samples were not assigned a test fold.")

    tfidf_metrics = calculate_metrics(
        labels,
        tfidf_predictions,
        tfidf_probabilities,
    )

    minilm_metrics = calculate_metrics(
        labels,
        minilm_predictions,
        minilm_probabilities,
    )

    delta = {
        key: float(minilm_metrics[key] - tfidf_metrics[key])
        for key in tfidf_metrics
    }

    print("\n=== OUT-OF-FOLD RESULTS ===")
    print(f"{'Metric':<24} {'TF-IDF':>10} {'MiniLM':>10} {'Delta':>10}")
    print("-" * 58)

    for metric in tfidf_metrics:
        print(
            f"{metric:<24} "
            f"{tfidf_metrics[metric]:>10.4f} "
            f"{minilm_metrics[metric]:>10.4f} "
            f"{delta[metric]:>+10.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame({
        "sample_id": [row["sample_id"] for row in records],
        "video_id": groups,
        "true_label": labels,
        "fold": oof_folds,
        "tfidf_prediction": tfidf_predictions,
        "tfidf_probability_humorous": tfidf_probabilities,
        "minilm_prediction": minilm_predictions,
        "minilm_probability_humorous": minilm_probabilities,
    })

    predictions_path = OUTPUT_DIR / "oof_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    results = {
        "experiment": "TF-IDF vs frozen MiniLM sentence embeddings",
        "samples": int(n),
        "annotated_samples": int(total_annotations),
        "available_sample_records": int(total_samples),
        "empty_text_excluded": int(total_annotations - n),
        "distinct_videos": int(len(set(groups))),
        "label_counts": dict(Counter(labels)),
        "evaluation": f"{N_SPLITS}-fold GroupKFold by source video",
        "video_leakage_check": "passed",
        "encoder": MODEL_NAME,
        "encoder_hidden_size": int(embeddings.shape[1]),
        "encoder_trainable": False,
        "max_token_length": MAX_LENGTH,
        "classifier": "class-weighted LogisticRegression",
        "tfidf_settings": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "min_df": 1,
            "sublinear_tf": True,
            "class_weight": "balanced",
            "max_iter": 3000,
        },
        "tfidf_oof_metrics": tfidf_metrics,
        "minilm_oof_metrics": minilm_metrics,
        "delta_minilm_minus_tfidf": delta,
        "fold_results": fold_results,
        "runtime_seconds": round(time.perf_counter() - start_time, 2),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "predictions_file": str(predictions_path.relative_to(ROOT)),
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    metrics_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved metrics: {metrics_path.relative_to(ROOT)}")
    print(f"Saved predictions: {predictions_path.relative_to(ROOT)}")
    print(f"Total runtime: {results['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()