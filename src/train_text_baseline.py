import csv
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = ROOT / "annotations" / "pilot" / "pilot_samples.json"
ANNOTATIONS_FILE = (
    ROOT
    / "annotations"
    / "pilot"
    / "pilot_annotations_second_completed.json"
)

OUTPUT_DIR = ROOT / "results" / "text_baseline"

LABELS = ["non_humorous", "humorous"]
N_SPLITS = 5
RANDOM_STATE = 42


def load_dataset():
    """Load annotations and match them to their corresponding pilot samples."""
    if not SAMPLES_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {SAMPLES_FILE}")

    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing completed annotations: {ANNOTATIONS_FILE}"
        )

    with SAMPLES_FILE.open("r", encoding="utf-8-sig") as file:
        samples = json.load(file)

    with ANNOTATIONS_FILE.open("r", encoding="utf-8-sig") as file:
        annotations = json.load(file)

    sample_ids = [sample["sample_id"] for sample in samples]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs found in pilot_samples.json.")

    annotation_ids = [row["sample_id"] for row in annotations]

    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Duplicate sample IDs found in the annotation file.")

    samples_by_id = {
        sample["sample_id"]: sample for sample in samples
    }

    if set(annotation_ids) != set(sample_ids):
        missing_samples = set(annotation_ids) - set(sample_ids)
        missing_annotations = set(sample_ids) - set(annotation_ids)

        raise ValueError(
            "Sample and annotation IDs do not match.\n"
            f"Annotation IDs without samples: {len(missing_samples)}\n"
            f"Samples without annotations: {len(missing_annotations)}"
        )

    records = []

    for annotation in annotations:
        sample_id = annotation["sample_id"]
        label = annotation.get("humor_label")

        if label not in LABELS:
            raise ValueError(
                f"Invalid label for {sample_id}: {label!r}"
            )

        sample = samples_by_id[sample_id]
        text = " ".join(str(sample.get("text") or "").split())
        video_id = str(sample.get("video_id") or "").strip()

        if not text:
            continue

        if not video_id:
            raise ValueError(f"Missing video_id for {sample_id}")

        records.append(
            {
                "sample_id": sample_id,
                "video_id": video_id,
                "text": text,
                "label": label,
            }
        )

    if len(records) < N_SPLITS:
        raise ValueError("Not enough text samples for cross-validation.")

    if len({row["label"] for row in records}) != 2:
        raise ValueError("The usable dataset must contain both classes.")

    if len({row["video_id"] for row in records}) < N_SPLITS:
        raise ValueError(
            f"At least {N_SPLITS} distinct source videos are required."
        )

    return records, len(samples), len(annotations)


def build_model():
    """Construct the text-classification pipeline."""
    return Pipeline(
        [
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
        ]
    )


def calculate_metrics(y_true, y_pred, probabilities):
    """Calculate metrics using humorous as the positive class."""
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
                pos_label="humorous",
                zero_division=0,
            )
        ),
        "recall_humorous": float(
            recall_score(
                y_true,
                y_pred,
                pos_label="humorous",
                zero_division=0,
            )
        ),
        "f1_humorous": float(
            f1_score(
                y_true,
                y_pred,
                pos_label="humorous",
                zero_division=0,
            )
        ),
        "roc_auc_humorous": float(
            roc_auc_score(
                [1 if label == "humorous" else 0 for label in y_true],
                probabilities,
            )
        ),
    }


def main():
    records, total_samples, total_annotations = load_dataset()

    texts = [row["text"] for row in records]
    labels = [row["label"] for row in records]
    groups = [row["video_id"] for row in records]

    group_count = len(set(groups))

    print("\n--- MMHD Text Baseline ---")
    print(f"Pilot samples: {total_samples}")
    print(f"Annotations: {total_annotations}")
    print(f"Samples used for training/evaluation: {len(records)}")
    print(f"Samples excluded due to empty text: {total_annotations - len(records)}")
    print(f"Distinct source videos: {group_count}")
    print(f"Label counts: {dict(Counter(labels))}")
    print(f"Cross-validation: {N_SPLITS}-fold GroupKFold by video_id")

    # Each sample receives a prediction only when its source video
    # is held out from the corresponding fold's training data.
    oof_predictions = [None] * len(records)
    oof_probabilities = [None] * len(records)
    oof_folds = [None] * len(records)
    fold_results = []

    splitter = GroupKFold(n_splits=N_SPLITS)

    for fold_number, (train_idx, test_idx) in enumerate(
        splitter.split(texts, labels, groups),
        start=1,
    ):
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]

        if set(train_labels) != set(LABELS):
            raise ValueError(
                f"Fold {fold_number} training data does not contain both classes."
            )

        train_videos = {groups[i] for i in train_idx}
        test_videos = {groups[i] for i in test_idx}

        if train_videos & test_videos:
            raise RuntimeError(
                f"Video leakage detected in fold {fold_number}."
            )

        model = build_model()

        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        model.fit(train_texts, train_labels)

        predictions = model.predict(test_texts)

        classifier = model.named_steps["classifier"]
        humorous_index = list(classifier.classes_).index("humorous")

        probabilities = model.predict_proba(test_texts)[:, humorous_index]

        for local_index, original_index in enumerate(test_idx):
            oof_predictions[original_index] = str(
                predictions[local_index]
            )
            oof_probabilities[original_index] = float(
                probabilities[local_index]
            )
            oof_folds[original_index] = fold_number

        fold_results.append(
            {
                "fold": fold_number,
                "training_samples": len(train_idx),
                "validation_samples": len(test_idx),
                "training_videos": len(train_videos),
                "validation_videos": len(test_videos),
                "accuracy": float(
                    accuracy_score(test_labels, predictions)
                ),
            }
        )

        print(
            f"Fold {fold_number}/{N_SPLITS}: "
            f"train={len(train_idx)}, validation={len(test_idx)}, "
            f"accuracy={accuracy_score(test_labels, predictions):.3f}"
        )

    if any(value is None for value in oof_predictions):
        raise RuntimeError("Some samples did not receive an out-of-fold prediction.")

    if any(value is None for value in oof_probabilities):
        raise RuntimeError("Some samples did not receive an out-of-fold probability.")

    y_pred = oof_predictions
    probabilities = oof_probabilities

    overall_metrics = calculate_metrics(labels, y_pred, probabilities)

    report = classification_report(
        labels,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(labels, y_pred, labels=LABELS)

    # Compare with a majority-class baseline evaluated using the same folds.
    majority_predictions = [None] * len(records)

    for train_idx, test_idx in splitter.split(texts, labels, groups):
        majority_label = Counter(
            labels[i] for i in train_idx
        ).most_common(1)[0][0]

        for original_index in test_idx:
            majority_predictions[original_index] = majority_label

    majority_metrics = {
        "accuracy": float(
            accuracy_score(labels, majority_predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, majority_predictions)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                majority_predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save out-of-fold predictions for transparent error analysis.
    predictions_path = OUTPUT_DIR / "oof_predictions.csv"

    with predictions_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "video_id",
                "text",
                "true_label",
                "predicted_label",
                "humorous_probability",
                "fold",
            ],
        )

        writer.writeheader()

        for index, row in enumerate(records):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "video_id": row["video_id"],
                    "text": row["text"],
                    "true_label": labels[index],
                    "predicted_label": y_pred[index],
                    "humorous_probability": round(probabilities[index], 6),
                    "fold": oof_folds[index],
                }
            )

    # Save the confusion matrix as an image.
    figure, axis = plt.subplots(figsize=(6, 5))

    image = axis.imshow(matrix)

    axis.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title="MMHD Text Baseline — Out-of-Fold Confusion Matrix",
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=180,
    )
    plt.close(figure)

    metrics = {
        "experiment": "TF-IDF + Logistic Regression",
        "task": "Binary text-based humor classification",
        "positive_class": "humorous",
        "dataset": {
            "pilot_samples": total_samples,
            "annotation_records": total_annotations,
            "samples_used": len(records),
            "samples_excluded_empty_text": total_annotations - len(records),
            "source_videos": group_count,
            "label_distribution": dict(Counter(labels)),
        },
        "evaluation": {
            "method": f"{N_SPLITS}-fold GroupKFold grouped by video_id",
            "metrics_are_out_of_fold": True,
            "out_of_fold_metrics": overall_metrics,
            "majority_class_baseline": majority_metrics,
            "classification_report": report,
            "confusion_matrix_label_order": LABELS,
            "confusion_matrix": matrix.tolist(),
            "folds": fold_results,
        },
        "software": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
        },
    }

    with (OUTPUT_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    # Fit a final model on all eligible pilot examples for demonstration use.
    final_model = build_model()
    final_model.fit(texts, labels)

    joblib.dump(final_model, OUTPUT_DIR / "model.joblib")

    print("\n--- Out-of-fold evaluation results ---")

    for name, value in overall_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n--- Majority-class baseline ---")

    for name, value in majority_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nSaved files:")
    print(f"  Results directory: {OUTPUT_DIR}")
    print(f"  Metrics: {OUTPUT_DIR / 'metrics.json'}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"  Final demonstration model: {OUTPUT_DIR / 'model.joblib'}")

    print(
        "\nImportant: use the out-of-fold metrics for reporting. "
        "The saved demonstration model was subsequently fitted on all "
        "eligible pilot samples."
    )


if __name__ == "__main__":
    main()