import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import sklearn
from PIL import Image

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
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = ROOT / "annotations" / "pilot" / "pilot_samples.json"
ANNOTATIONS_FILE = (
    ROOT / "annotations" / "pilot"
    / "pilot_annotations_second_completed.json"
)
EXAMPLES_FILE = ROOT / "results" / "modality_examples.csv"
AUDIO_FEATURES_FILE = (
    ROOT / "results" / "audio_baseline" / "audio_features.csv"
)

OUTPUT_DIR = ROOT / "results" / "five_stream_comparison"

LABELS = ["non_humorous", "humorous"]
N_SPLITS = 5
RANDOM_STATE = 42

AUDIO_NAMES = [
    "duration_seconds",
    "rms_mean",
    "rms_std",
    "rms_p90",
    "zero_crossing_mean",
    "zero_crossing_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "silence_frame_ratio",
]

FRAME_NAMES = [
    "red_mean",
    "green_mean",
    "blue_mean",
    "red_std",
    "green_std",
    "blue_std",
    "brightness_mean",
    "brightness_std",
    "saturation_mean",
    "saturation_std",
    "horizontal_gradient",
    "vertical_gradient",
]

VISUAL_NAMES = (
    [f"frame_mean_{name}" for name in FRAME_NAMES]
    + [f"frame_std_{name}" for name in FRAME_NAMES]
)

STRING_COMPONENTS = {"text", "ocr", "emoji"}
NUMERIC_WIDTH = {
    "audio": len(AUDIO_NAMES),
    "visual": len(VISUAL_NAMES),
}

MODEL_COMPONENTS = {
    "Text-only": ["text"],
    "Audio-only": ["audio"],
    "Visual-only": ["visual"],
    "OCR-only": ["ocr"],
    "Text + OCR": ["text", "ocr"],
    "Text + Audio": ["text", "audio"],
    "Audio + OCR": ["audio", "ocr"],
    "Text + Audio + Visual": ["text", "audio", "visual"],
    "Text + Audio + OCR": ["text", "audio", "ocr"],
    "Text + Audio + Visual + OCR": [
        "text", "audio", "visual", "ocr"
    ],
    "All five streams": [
        "text", "audio", "visual", "ocr", "emoji"
    ],
}


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def resolve_path(value):
    if not value:
        return None

    path = Path(str(value).replace("/", "\\"))

    if not path.is_absolute():
        path = ROOT / path

    return path


def extract_one_frame(path):
    """Calculate basic, interpretable image statistics."""
    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize((96, 96)),
            dtype=np.float32,
        ) / 255.0

    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    brightness = (
        0.299 * red + 0.587 * green + 0.114 * blue
    )

    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    saturation = (maximum - minimum) / (maximum + 1e-6)

    return np.asarray([
        np.mean(red),
        np.mean(green),
        np.mean(blue),
        np.std(red),
        np.std(green),
        np.std(blue),
        np.mean(brightness),
        np.std(brightness),
        np.mean(saturation),
        np.std(saturation),
        np.mean(np.abs(np.diff(brightness, axis=1))),
        np.mean(np.abs(np.diff(brightness, axis=0))),
    ], dtype=np.float64)


def extract_visual_features(paths):
    frame_features = []

    for value in paths or []:
        path = resolve_path(value)

        if path is None or not path.is_file():
            continue

        try:
            frame_features.append(extract_one_frame(path))
        except (OSError, ValueError):
            continue

    if not frame_features:
        return None

    matrix = np.vstack(frame_features)

    return np.concatenate([
        np.mean(matrix, axis=0),
        np.std(matrix, axis=0),
    ])


def make_vectorizer(component):
    if component == "text":
        return TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

    if component == "ocr":
        return TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )

    if component == "emoji":
        # Emoji is sparse. This feature is exploratory, not a
        # standalone emoji classifier.
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

    raise ValueError(f"Unknown string component: {component}")


def load_records():
    samples = read_json(SAMPLES_FILE)
    annotations = read_json(ANNOTATIONS_FILE)
    examples = read_csv(EXAMPLES_FILE)
    audio_rows = read_csv(AUDIO_FEATURES_FILE)

    sample_by_id = {s["sample_id"]: s for s in samples}
    label_by_id = {
        a["sample_id"]: a.get("humor_label")
        for a in annotations
    }
    examples_by_id = {r["sample_id"]: r for r in examples}
    audio_by_id = {r["sample_id"]: r for r in audio_rows}

    if len(sample_by_id) != len(samples):
        raise ValueError("Duplicate sample IDs in pilot dataset.")

    if len(label_by_id) != len(annotations):
        raise ValueError("Duplicate annotation IDs.")

    if set(label_by_id) != set(sample_by_id):
        raise ValueError("Annotation IDs do not match pilot samples.")

    records = []
    excluded_no_ocr = 0
    excluded_no_audio = 0
    excluded_no_visual = 0

    for sample_id, annotation_label in label_by_id.items():
        if annotation_label not in LABELS:
            raise ValueError(
                f"Invalid label for {sample_id}: {annotation_label!r}"
            )

        example = examples_by_id.get(sample_id, {})
        ocr = str(example.get("ocr") or "").strip()

        # A common complete-case experiment: every configuration
        # uses the same OCR-available samples.
        if not ocr:
            excluded_no_ocr += 1
            continue

        sample = sample_by_id[sample_id]
        audio_row = audio_by_id.get(sample_id)

        if audio_row is None:
            excluded_no_audio += 1
            continue

        audio = np.asarray(
            [float(audio_row[name]) for name in AUDIO_NAMES],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(audio)):
            raise ValueError(f"Invalid audio features for {sample_id}")

        visual = extract_visual_features(
            sample.get("image_paths", [])
        )

        if visual is None:
            excluded_no_visual += 1
            continue

        text = " ".join(
            str(sample.get("text") or "").split()
        )

        emoji = str(example.get("emoji") or "").strip()

        video_id = str(sample.get("video_id") or "").strip()

        if not video_id:
            raise ValueError(f"Missing source video ID for {sample_id}")

        records.append({
            "sample_id": sample_id,
            "video_id": video_id,
            "text": text,
            "audio": audio,
            "visual": visual,
            "ocr": ocr,
            "emoji": emoji,
            "label": annotation_label,
        })

    return records, {
        "pilot_samples": len(samples),
        "annotation_records": len(annotations),
        "usable_complete_cases": len(records),
        "excluded_no_ocr": excluded_no_ocr,
        "excluded_no_audio_features": excluded_no_audio,
        "excluded_no_visual_features": excluded_no_visual,
    }


def make_input(records, components):
    """Build an input matrix for a particular modality combination."""
    if len(components) == 1:
        component = components[0]

        if component in STRING_COMPONENTS:
            return np.asarray(
                [record[component] for record in records],
                dtype=object,
            )

        return np.vstack([
            record[component] for record in records
        ]).astype(np.float64)

    if all(component not in STRING_COMPONENTS for component in components):
        return np.hstack([
            np.vstack([
                record[component] for record in records
            ])
            for component in components
        ]).astype(np.float64)

    widths = {
        component: (
            1 if component in STRING_COMPONENTS
            else NUMERIC_WIDTH[component]
        )
        for component in components
    }

    total_width = sum(widths.values())
    matrix = np.empty(
        (len(records), total_width),
        dtype=object,
    )

    for row_index, record in enumerate(records):
        cursor = 0

        for component in components:
            value = record[component]
            width = widths[component]

            if component in STRING_COMPONENTS:
                matrix[row_index, cursor] = value
            else:
                matrix[row_index, cursor:cursor + width] = value

            cursor += width

    return matrix


def make_feature_transformer(components):
    transformers = []
    cursor = 0

    for component in components:
        if component in STRING_COMPONENTS:
            transformers.append((
                component,
                make_vectorizer(component),
                cursor,
            ))
            cursor += 1
        else:
            width = NUMERIC_WIDTH[component]

            transformers.append((
                component,
                StandardScaler(),
                list(range(cursor, cursor + width)),
            ))

            cursor += width

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_model(components):
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    if len(components) == 1:
        component = components[0]

        if component in STRING_COMPONENTS:
            return Pipeline([
                ("features", make_vectorizer(component)),
                ("classifier", classifier),
            ])

        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])

    if all(component not in STRING_COMPONENTS for component in components):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])

    return Pipeline([
        ("features", make_feature_transformer(components)),
        ("classifier", classifier),
    ])


def get_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true, predictions,
                labels=LABELS, average="macro", zero_division=0
            )
        ),
        "precision_humorous": float(
            precision_score(
                y_true, predictions,
                pos_label="humorous", zero_division=0
            )
        ),
        "recall_humorous": float(
            recall_score(
                y_true, predictions,
                pos_label="humorous", zero_division=0
            )
        ),
        "f1_humorous": float(
            f1_score(
                y_true, predictions,
                pos_label="humorous", zero_division=0
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
    records, data_summary = load_records()

    if len(records) < N_SPLITS:
        raise ValueError("Not enough complete cases for cross-validation.")

    y = [record["label"] for record in records]
    groups = [record["video_id"] for record in records]

    if set(y) != set(LABELS):
        raise ValueError("Both humor classes are required.")

    if len(set(groups)) < N_SPLITS:
        raise ValueError("Not enough source videos for GroupKFold.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    emoji_rows = [record for record in records if record["emoji"]]

    print("\n--- MMHD Five-Stream Comparison ---")
    for name, value in data_summary.items():
        print(f"{name}: {value}")

    print(f"Source videos: {len(set(groups))}")
    print(f"Label counts: {dict(Counter(y))}")
    print(f"Samples with emoji in this subset: {len(emoji_rows)}")
    print(
        "Emoji labels in this subset: "
        f"{dict(Counter(row['label'] for row in emoji_rows))}"
    )
    print("Evaluation: 5-fold GroupKFold by video_id")
    print("All models use the same complete-case samples and folds.")

    splitter = GroupKFold(n_splits=N_SPLITS)

    splits = list(
        splitter.split(
            np.zeros((len(records), 1)),
            y,
            groups,
        )
    )

    all_results = {}
    all_oof = {}

    for name, components in MODEL_COMPONENTS.items():
        print(f"\n--- {name} ---")

        X = make_input(records, components)

        oof_predictions = [None] * len(records)
        oof_probabilities = [None] * len(records)
        fold_results = []

        for fold_number, (train_idx, test_idx) in enumerate(
            splits,
            start=1,
        ):
            train_videos = {groups[i] for i in train_idx}
            test_videos = {groups[i] for i in test_idx}

            if train_videos & test_videos:
                raise RuntimeError("Source-video leakage detected.")

            y_train = [y[i] for i in train_idx]
            y_test = [y[i] for i in test_idx]

            if set(y_train) != set(LABELS):
                raise ValueError(
                    f"Fold {fold_number} lacks a training class."
                )

            model = build_model(components)
            model.fit(X[train_idx], y_train)

            predictions = model.predict(X[test_idx])
            probabilities = model.predict_proba(X[test_idx])

            classes = list(model.named_steps["classifier"].classes_)
            humorous_index = classes.index("humorous")

            for local_index, original_index in enumerate(test_idx):
                oof_predictions[original_index] = str(
                    predictions[local_index]
                )
                oof_probabilities[original_index] = float(
                    probabilities[local_index, humorous_index]
                )

            fold_accuracy = accuracy_score(y_test, predictions)

            fold_results.append({
                "fold": fold_number,
                "training_samples": len(train_idx),
                "validation_samples": len(test_idx),
                "training_videos": len(train_videos),
                "validation_videos": len(test_videos),
                "accuracy": float(fold_accuracy),
            })

            print(
                f"    Fold {fold_number}/{N_SPLITS}: "
                f"accuracy={fold_accuracy:.3f}"
            )

        if any(value is None for value in oof_predictions):
            raise RuntimeError("Missing out-of-fold predictions.")

        metrics = get_metrics(y, oof_predictions, oof_probabilities)

        matrix = confusion_matrix(
            y, oof_predictions, labels=LABELS
        )

        # Train the final demonstration model on all common cases.
        final_model = build_model(components)
        final_model.fit(X, y)

        safe_name = (
            name.lower()
            .replace(" + ", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

        joblib.dump(final_model, model_dir / f"{safe_name}.joblib")

        all_results[name] = {
            "components": components,
            "out_of_fold_metrics": metrics,
            "confusion_matrix_label_order": LABELS,
            "confusion_matrix": matrix.tolist(),
            "folds": fold_results,
        }

        all_oof[name] = {
            "predictions": oof_predictions,
            "probabilities": oof_probabilities,
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
                y, majority_predictions,
                labels=LABELS, average="macro", zero_division=0
            )
        ),
    }

    results = {
        "experiment": (
            "Common-subset ablation of text, audio, visual, OCR and emoji"
        ),
        "evaluation_method": (
            "Five-fold GroupKFold grouped by source video. "
            "All configurations use identical complete-case samples and folds."
        ),
        "dataset": data_summary,
        "label_distribution": dict(Counter(y)),
        "emoji_examples_in_common_subset": len(emoji_rows),
        "emoji_label_distribution": dict(
            Counter(row["label"] for row in emoji_rows)
        ),
        "majority_class_baseline": majority_metrics,
        "models": all_results,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "The common subset is limited to samples with OCR, audio features and visual features.",
            "OCR contains recognition errors and repeated on-screen branding.",
            "Emoji coverage is sparse; the full-stream comparison is exploratory.",
            "Visual features are low-level image statistics, not semantic vision embeddings.",
            "Performance on this small labeled pilot may not generalize to the full corpus.",
        ],
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    prediction_path = OUTPUT_DIR / "oof_predictions.csv"

    with prediction_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fields = [
            "sample_id",
            "video_id",
            "true_label",
            "text",
            "ocr",
            "emoji",
        ]

        for name in MODEL_COMPONENTS:
            safe_name = (
                name.lower()
                .replace(" + ", "_")
                .replace(" ", "_")
                .replace("-", "_")
            )
            fields.extend([
                f"{safe_name}_prediction",
                f"{safe_name}_humorous_probability",
            ])

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for i, record in enumerate(records):
            row = {
                "sample_id": record["sample_id"],
                "video_id": record["video_id"],
                "true_label": y[i],
                "text": record["text"],
                "ocr": record["ocr"],
                "emoji": record["emoji"],
            }

            for name, values in all_oof.items():
                safe_name = (
                    name.lower()
                    .replace(" + ", "_")
                    .replace(" ", "_")
                    .replace("-", "_")
                )

                row[f"{safe_name}_prediction"] = values["predictions"][i]
                row[f"{safe_name}_humorous_probability"] = round(
                    values["probabilities"][i], 6
                )

            writer.writerow(row)

    print("\n--- Common-subset summary ---")
    print(
        f"{'Model':<34} {'Accuracy':>10} "
        f"{'Balanced Acc.':>14} {'Macro F1':>10} {'Humor F1':>10}"
    )

    for name, output in all_results.items():
        metric = output["out_of_fold_metrics"]

        print(
            f"{name:<34} "
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
    print(f"Saved models: {model_dir}")


if __name__ == "__main__":
    main()