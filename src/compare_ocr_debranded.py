import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
import sys

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
OUTPUT_DIR = ROOT / "results" / "ocr_debranding"

LABELS = ["non_humorous", "humorous"]
N_SPLITS = 5
RANDOM_STATE = 42


# These rules target visible source branding and promotional overlays
# observed during the OCR audit. They do not use the humor labels.
DEBRANDING_PATTERNS = [
    r"\bJ?\s*DRY\s*BAR(?:\s+COMEDY)?\b",
    r"\bCONAN\s+O['’]?\s*BRIEN\s+NEEDS\s+A\s+FRIEND\b",
    r"\bCONAN\b",
    r"\bCLUELESS\s+GAMER\b",
    r"\bCASSIE\s+CAGE\b",
    r"\bNBNBC\b",
    r"\bPASSWORD\b",
    r"\bWATCH\s+MORE\b",
    r"\bSUBSCRIBE(?:\s+NOW)?\b",
    r"\bWATCH\s+(?:FULL\s+)?EPISODES?\b",
    r"\bLIKE\s+FOLLOW\b",
    r"\bMORECLIPS\b",
    r"\bDOWNLOAD\s+ON\s+THE\b",
    r"\bGET\s+IT\s+ON\b",
    r"\bAPP\s+STORE\b",
    r"\bGOOGLE\s+PLAY\b",
]


def debrand_ocr(text):
    """Remove predefined branding and promotional phrases."""
    cleaned = str(text or "")

    for pattern in DEBRANDING_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    # Normalize spacing while preserving the remaining OCR words.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def read_csv(path):
    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def build_text_model():
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


def build_ocr_model():
    return Pipeline([
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


def build_text_ocr_model():
    features = ColumnTransformer([
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

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


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

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
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

        class_order = list(
            model.named_steps["classifier"].classes_
        )
        humorous_index = class_order.index("humorous")

        for local_index, original_index in enumerate(test_idx):
            predictions[original_index] = str(
                fold_predictions[local_index]
            )
            probabilities[original_index] = float(
                fold_probabilities[local_index, humorous_index]
            )

        fold_accuracy = accuracy_score(y_test, fold_predictions)

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
        confusion_matrix(
            y, predictions, labels=LABELS
        ).tolist(),
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

    by_id = {}

    for row in examples:
        sample_id = row["sample_id"]

        if sample_id in by_id:
            raise ValueError(
                f"Duplicate sample in modality audit: {sample_id}"
            )

        by_id[sample_id] = row

    # Preserve the same OCR-eligible sample-selection rule as
    # compare_ocr.py. Samples remain included after cleaning even
    # if their cleaned OCR becomes empty.
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
            "ocr_raw": ocr,
            "ocr_debranded": debrand_ocr(ocr),
            "label": labels_by_id[sample_id],
        })

    if not records:
        raise ValueError("No labeled samples with usable OCR found.")

    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate sample IDs in the experiment.")

    if any(not r["video_id"] for r in records):
        raise ValueError("Some samples have no video_id.")

    y = [r["label"] for r in records]
    groups = [r["video_id"] for r in records]

    texts = [r["text"] for r in records]
    raw_ocr = [r["ocr_raw"] for r in records]
    clean_ocr = [r["ocr_debranded"] for r in records]

    if set(y) != set(LABELS):
        raise ValueError("Both humor labels are required.")

    if len(set(groups)) < N_SPLITS:
        raise ValueError("Not enough source videos for GroupKFold.")

    X_by_model = {
        "Text-only": np.asarray(texts, dtype=object),
        "Raw OCR-only": np.asarray(raw_ocr, dtype=object),
        "De-branded OCR-only": np.asarray(clean_ocr, dtype=object),
        "Text + raw OCR": np.column_stack([texts, raw_ocr]),
        "Text + de-branded OCR": np.column_stack(
            [texts, clean_ocr]
        ),
    }

    model_templates = {
        "Text-only": build_text_model(),
        "Raw OCR-only": build_ocr_model(),
        "De-branded OCR-only": build_ocr_model(),
        "Text + raw OCR": build_text_ocr_model(),
        "Text + de-branded OCR": build_text_ocr_model(),
    }

    splitter = GroupKFold(n_splits=N_SPLITS)

    splits = list(
        splitter.split(
            np.zeros((len(records), 1)),
            y,
            groups,
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- MMHD OCR De-branding Experiment ---")
    print(f"Samples: {len(records)}")
    print(f"Source videos: {len(set(groups))}")
    print(f"Label distribution: {dict(Counter(y))}")
    print(
        "Evaluation: 5-fold GroupKFold by source video"
    )
    print("All models use identical samples and folds.")
    print(
        "Samples with empty OCR after cleaning:",
        sum(not value for value in clean_ocr),
    )

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

    # Majority-class baseline, using training labels in each fold.
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

    # Save out-of-fold predictions from every comparison.
    prediction_path = OUTPUT_DIR / "oof_predictions.csv"

    with prediction_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fields = [
            "sample_id",
            "video_id",
            "true_label",
            "text",
            "ocr_raw",
            "ocr_debranded",
        ]

        for name in model_templates:
            prefix = name.lower().replace(" ", "_").replace("+", "plus")
            fields.extend([
                f"{prefix}_prediction",
                f"{prefix}_humorous_probability",
            ])

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for index, row in enumerate(records):
            output_row = {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "true_label": y[index],
                "text": row["text"],
                "ocr_raw": row["ocr_raw"],
                "ocr_debranded": row["ocr_debranded"],
            }

            for name in model_templates:
                prefix = name.lower().replace(" ", "_").replace("+", "plus")
                output_row[f"{prefix}_prediction"] = (
                    all_predictions[name]["predictions"][index]
                )
                output_row[
                    f"{prefix}_humorous_probability"
                ] = round(
                    all_predictions[name]["probabilities"][index],
                    6,
                )

            writer.writerow(output_row)

    output = {
        "experiment": "Raw versus de-branded OCR for humor classification",
        "sample_count": len(records),
        "source_video_count": len(set(groups)),
        "label_distribution": dict(Counter(y)),
        "samples_with_nonempty_spoken_text": sum(
            bool(text) for text in texts
        ),
        "samples_with_empty_ocr_after_cleaning": sum(
            not value for value in clean_ocr
        ),
        "evaluation_method": (
            "Five-fold GroupKFold by source video. "
            "Every comparison uses identical samples and folds."
        ),
        "debranding_patterns": DEBRANDING_PATTERNS,
        "majority_class_baseline": majority_metrics,
        "models": results,
        "software": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "This is a small labeled pilot, not the full corpus.",
            "The de-branding rules are manually predefined and may remove "
            "some legitimate words as well as overlays.",
            "Removing observed branding does not eliminate all possible "
            "source-related shortcuts.",
            "Results measure classification performance, not proof of "
            "semantic humor understanding.",
        ],
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n--- Comparison Summary ---")
    print(
        f"{'Model':<25} {'Accuracy':>10} "
        f"{'Balanced Acc.':>14} {'Macro F1':>10} {'Humor F1':>10}"
    )

    for name, result in results.items():
        metric = result["out_of_fold_metrics"]

        print(
            f"{name:<25} "
            f"{metric['accuracy']:>10.4f} "
            f"{metric['balanced_accuracy']:>14.4f} "
            f"{metric['macro_f1']:>10.4f} "
            f"{metric['f1_humorous']:>10.4f}"
        )

    print("\nMajority-class baseline:")
    for metric, value in majority_metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Predictions saved to: {prediction_path}")


if __name__ == "__main__":
    main()