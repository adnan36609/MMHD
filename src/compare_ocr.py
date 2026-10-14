import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

EXAMPLES_FILE = ROOT / "results" / "modality_examples.csv"
ANNOTATIONS_FILE = (
    ROOT
    / "annotations"
    / "pilot"
    / "pilot_annotations_second_completed.json"
)

OUTPUT_DIR = ROOT / "results" / "ocr_comparison"

LABELS = ["non_humorous", "humorous"]
N_SPLITS = 5
RANDOM_STATE = 42


def read_csv(path):
    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def build_models():
    text_model = Pipeline([
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

    ocr_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(3, 5),
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

    text_ocr_features = ColumnTransformer([
        (
            "spoken_text",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
            0,
        ),
        (
            "ocr_text",
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
            ),
            1,
        ),
    ])

    text_ocr_model = Pipeline([
        ("features", text_ocr_features),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    return {
        "Text-only": text_model,
        "OCR-only": ocr_model,
        "Text + OCR": text_ocr_model,
    }


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
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_humorous": float(
            precision_score(
                y_true,
                predictions,
                pos_label="humorous",
                zero_division=0,
            )
        ),
        "recall_humorous": float(
            recall_score(
                y_true,
                predictions,
                pos_label="humorous",
                zero_division=0,
            )
        ),
        "f1_humorous": float(
            f1_score(
                y_true,
                predictions,
                pos_label="humorous",
                zero_division=0,
            )
        ),
        "roc_auc_humorous": float(
            roc_auc_score(
                [
                    1 if label == "humorous" else 0
                    for label in y_true
                ],
                probabilities,
            )
        ),
    }


def evaluate_model(model_template, X, y, groups, splits):
    predictions = [None] * len(y)
    probabilities = [None] * len(y)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splits, start=1
    ):
        train_videos = {groups[i] for i in train_idx}
        test_videos = {groups[i] for i in test_idx}

        if train_videos & test_videos:
            raise RuntimeError(
                f"Source-video leakage detected in fold {fold}."
            )

        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        if set(y_train) != set(LABELS):
            raise ValueError(
                f"Training fold {fold} lacks one of the classes."
            )

        model = clone(model_template)
        model.fit(X[train_idx], y_train)

        fold_predictions = model.predict(X[test_idx])
        fold_probabilities = model.predict_proba(X[test_idx])

        class_order = list(model.named_steps["classifier"].classes_)
        humorous_index = class_order.index("humorous")

        for local_index, original_index in enumerate(test_idx):
            predictions[original_index] = str(
                fold_predictions[local_index]
            )
            probabilities[original_index] = float(
                fold_probabilities[local_index, humorous_index]
            )

        fold_accuracy = accuracy_score(
            y_test, fold_predictions
        )

        fold_results.append({
            "fold": fold,
            "training_samples": len(train_idx),
            "validation_samples": len(test_idx),
            "training_videos": len(train_videos),
            "validation_videos": len(test_videos),
            "accuracy": float(fold_accuracy),
        })

        print(
            f"    Fold {fold}/{N_SPLITS}: "
            f"accuracy={fold_accuracy:.3f}"
        )

    if any(value is None for value in predictions):
        raise RuntimeError("Some samples lack predictions.")

    metrics = calculate_metrics(y, predictions, probabilities)

    return (
        metrics,
        predictions,
        probabilities,
        confusion_matrix(y, predictions, labels=LABELS).tolist(),
        fold_results,
    )


def main():
    for path in [EXAMPLES_FILE, ANNOTATIONS_FILE]:
        if not path.is_file():
            raise FileNotFoundError(path)

    examples = read_csv(EXAMPLES_FILE)
    annotations = read_json(ANNOTATIONS_FILE)

    labels_by_id = {
        row["sample_id"]: row["humor_label"]
        for row in annotations
    }

    # The examples file contains samples with OCR or emoji content.
    by_id = {}

    for row in examples:
        sample_id = row["sample_id"]

        if sample_id in by_id:
            raise ValueError(
                f"Duplicate sample in modality audit: {sample_id}"
            )

        by_id[sample_id] = row

    # Keep only samples with actual non-empty OCR content.
    # Empty spoken transcripts are retained as empty strings so
    # every modality comparison uses the same eligible sample set.
    records = []

    for sample_id, row in by_id.items():
        if sample_id not in labels_by_id:
            continue

        ocr = str(row.get("ocr") or "").strip()

        if not ocr:
            continue

        records.append({
            "sample_id": sample_id,
            "video_id": str(row.get("video_id") or "").strip(),
            "text": str(row.get("text") or "").strip(),
            "ocr": ocr,
            "emoji": str(row.get("emoji") or "").strip(),
            "label": labels_by_id[sample_id],
        })

    if not records:
        raise ValueError(
            "No labeled pilot samples with usable OCR were found."
        )

    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Duplicate sample IDs in OCR experiment.")

    if any(not row["video_id"] for row in records):
        raise ValueError("Some OCR samples have no video_id.")

    y = [row["label"] for row in records]
    groups = [row["video_id"] for row in records]
    texts = [row["text"] for row in records]
    ocr_texts = [row["ocr"] for row in records]

    if set(y) != set(LABELS):
        raise ValueError("Both humor labels are required.")

    if len(set(groups)) < N_SPLITS:
        raise ValueError("Not enough videos for GroupKFold.")

    X_by_model = {
        "Text-only": np.asarray(texts, dtype=object),
        "OCR-only": np.asarray(ocr_texts, dtype=object),
        "Text + OCR": np.column_stack([texts, ocr_texts]),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(
        splitter.split(
            np.zeros((len(records), 1)),
            y,
            groups,
        )
    )

    print("\n--- MMHD OCR Experiment ---")
    print(f"Samples with usable OCR: {len(records)}")
    print(f"Source videos: {len(set(groups))}")
    print(f"Samples with non-empty spoken text: {sum(bool(t) for t in texts)}")
    print(f"Label distribution: {dict(Counter(y))}")
    print("Evaluation: 5-fold GroupKFold by source video")
    print("All models use the same samples and folds.")

    # Emoji is analyzed descriptively; 13 pilot examples are too
    # few for a credible standalone supervised model.
    emoji_records = [
        row for row in records if row["emoji"]
    ]

    emoji_label_counts = Counter(
        row["label"] for row in emoji_records
    )
    emoji_value_counts = Counter(
        row["emoji"] for row in emoji_records
    )

    print("\n--- Emoji coverage among OCR-eligible pilot samples ---")
    print(f"Samples with emoji content: {len(emoji_records)}")
    print(f"Label distribution among emoji samples: {dict(emoji_label_counts)}")
    print(f"Emoji values: {dict(emoji_value_counts)}")

    with (OUTPUT_DIR / "emoji_examples.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fields = [
            "sample_id", "video_id", "true_label",
            "text", "ocr", "emoji"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for row in emoji_records:
            writer.writerow({
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "true_label": row["label"],
                "text": row["text"],
                "ocr": row["ocr"],
                "emoji": row["emoji"],
            })

    model_templates = build_models()
    results = {}
    all_predictions = {}

    for name, template in model_templates.items():
        print(f"\n--- {name} ---")

        (
            metrics,
            predictions,
            probabilities,
            matrix,
            fold_results,
        ) = evaluate_model(
            template,
            X_by_model[name],
            y,
            groups,
            splits,
        )

        results[name] = {
            "out_of_fold_metrics": metrics,
            "confusion_matrix_label_order": LABELS,
            "confusion_matrix": matrix,
            "folds": fold_results,
        }

        all_predictions[name] = {
            "predictions": predictions,
            "probabilities": probabilities,
        }

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Majority-class baseline on the same grouped folds.
    majority_predictions = [None] * len(records)

    for train_idx, test_idx in splits:
        majority_label = Counter(
            y[i] for i in train_idx
        ).most_common(1)[0][0]

        for index in test_idx:
            majority_predictions[index] = majority_label

    majority_metrics = {
        "accuracy": float(
            accuracy_score(y, majority_predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, majority_predictions)
        ),
        "macro_f1": float(
            f1_score(
                y,
                majority_predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }

    # Save out-of-fold predictions for error analysis.
    prediction_path = OUTPUT_DIR / "oof_predictions.csv"

    with prediction_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fields = [
            "sample_id", "video_id", "true_label",
            "text", "ocr", "emoji",
            "text_prediction", "text_probability",
            "ocr_prediction", "ocr_probability",
            "text_ocr_prediction", "text_ocr_probability",
        ]

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for index, row in enumerate(records):
            writer.writerow({
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "true_label": y[index],
                "text": row["text"],
                "ocr": row["ocr"],
                "emoji": row["emoji"],
                "text_prediction": all_predictions["Text-only"]["predictions"][index],
                "text_probability": round(
                    all_predictions["Text-only"]["probabilities"][index], 6
                ),
                "ocr_prediction": all_predictions["OCR-only"]["predictions"][index],
                "ocr_probability": round(
                    all_predictions["OCR-only"]["probabilities"][index], 6
                ),
                "text_ocr_prediction": all_predictions["Text + OCR"]["predictions"][index],
                "text_ocr_probability": round(
                    all_predictions["Text + OCR"]["probabilities"][index], 6
                ),
            })

    output = {
        "experiment": "OCR-aware humor classification",
        "evaluation_method": (
            "Five-fold GroupKFold by source video; "
            "all models use the same OCR-available samples and folds."
        ),
        "sample_count": len(records),
        "source_video_count": len(set(groups)),
        "label_distribution": dict(Counter(y)),
        "samples_with_spoken_text": sum(bool(t) for t in texts),
        "samples_with_emoji": len(emoji_records),
        "emoji_label_distribution": dict(emoji_label_counts),
        "emoji_value_distribution": dict(emoji_value_counts),
        "majority_class_baseline": majority_metrics,
        "models": results,
        "software": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "OCR content includes recognition errors and may include on-screen branding.",
            "Sparse emoji examples are reported descriptively, not treated as a reliable standalone evaluation.",
            "The study uses a small labeled pilot rather than the full 7,289-sample corpus.",
        ],
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n--- Summary ---")
    print(
        f"{'Model':<15} {'Accuracy':>10} "
        f"{'Balanced Acc.':>14} {'Macro F1':>10} {'Humor F1':>10}"
    )

    for name, result in results.items():
        metric = result["out_of_fold_metrics"]

        print(
            f"{name:<15} "
            f"{metric['accuracy']:>10.4f} "
            f"{metric['balanced_accuracy']:>14.4f} "
            f"{metric['macro_f1']:>10.4f} "
            f"{metric['f1_humorous']:>10.4f}"
        )

    print("\nMajority-class baseline:")
    for metric, value in majority_metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nMetrics: {metrics_path}")
    print(f"Predictions: {prediction_path}")
    print(f"Emoji examples: {OUTPUT_DIR / 'emoji_examples.csv'}")


if __name__ == "__main__":
    main()