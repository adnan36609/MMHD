import csv
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from PIL import Image
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
    ROOT
    / "annotations"
    / "pilot"
    / "pilot_annotations_second_completed.json"
)
AUDIO_FEATURES_FILE = (
    ROOT / "results" / "audio_baseline" / "audio_features.csv"
)

OUTPUT_DIR = ROOT / "results" / "multimodal_comparison"

LABELS = ["non_humorous", "humorous"]
N_SPLITS = 5
RANDOM_STATE = 42

AUDIO_FEATURE_NAMES = [
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

FRAME_FEATURE_NAMES = [
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

VISUAL_FEATURE_NAMES = (
    [f"frame_mean_{name}" for name in FRAME_FEATURE_NAMES]
    + [f"frame_std_{name}" for name in FRAME_FEATURE_NAMES]
)

MODEL_COMPONENTS = {
    "Text-only": ["text"],
    "Audio-only": ["audio"],
    "Visual-only": ["visual"],
    "Text + Audio": ["text", "audio"],
    "Text + Visual": ["text", "visual"],
    "Audio + Visual": ["audio", "visual"],
    "Text + Audio + Visual": ["text", "audio", "visual"],
}


def resolve_path(value):
    if not value:
        return None

    path = Path(str(value).replace("/", "\\"))

    if not path.is_absolute():
        path = ROOT / path

    return path


def extract_frame_features(path):
    """Calculate simple visual descriptors for one keyframe."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((96, 96))
        rgb = np.asarray(image, dtype=np.float32) / 255.0

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    brightness = (
        0.299 * red + 0.587 * green + 0.114 * blue
    )

    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    saturation = (maximum - minimum) / (maximum + 1e-6)

    horizontal_gradient = np.abs(
        np.diff(brightness, axis=1)
    )
    vertical_gradient = np.abs(
        np.diff(brightness, axis=0)
    )

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
        np.mean(horizontal_gradient),
        np.mean(vertical_gradient),
    ], dtype=np.float64)


def extract_visual_features(image_paths):
    """Aggregate visual descriptors over the available keyframes."""
    frame_features = []

    for image_path in image_paths or []:
        path = resolve_path(image_path)

        if path is None or not path.is_file():
            continue

        try:
            frame_features.append(extract_frame_features(path))
        except (OSError, ValueError):
            continue

    if not frame_features:
        return None

    frame_features = np.vstack(frame_features)

    means = np.mean(frame_features, axis=0)
    variations = np.std(frame_features, axis=0)

    return np.concatenate([means, variations])


def load_records():
    with SAMPLES_FILE.open("r", encoding="utf-8-sig") as file:
        samples = json.load(file)

    with ANNOTATIONS_FILE.open("r", encoding="utf-8-sig") as file:
        annotations = json.load(file)

    sample_by_id = {s["sample_id"]: s for s in samples}

    if len(sample_by_id) != len(samples):
        raise ValueError("Duplicate sample IDs found in pilot samples.")

    annotation_ids = [a["sample_id"] for a in annotations]

    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Duplicate sample IDs found in annotations.")

    if set(annotation_ids) != set(sample_by_id):
        raise ValueError("Annotation IDs do not match pilot samples.")

    audio_by_id = {}

    with AUDIO_FEATURES_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        for row in csv.DictReader(file):
            audio_by_id[row["sample_id"]] = row

    labels_by_id = {
        row["sample_id"]: row.get("humor_label")
        for row in annotations
    }

    records = []
    missing_text = 0
    missing_audio_features = 0
    missing_visual_features = 0

    for sample_id, sample in sample_by_id.items():
        label = labels_by_id[sample_id]

        if label not in LABELS:
            raise ValueError(
                f"Invalid label for {sample_id}: {label!r}"
            )

        text = " ".join(
            str(sample.get("text") or "").split()
        )

        if not text:
            missing_text += 1
            continue

        audio_row = audio_by_id.get(sample_id)

        if audio_row is None:
            missing_audio_features += 1
            continue

        try:
            audio_values = np.asarray([
                float(audio_row[name])
                for name in AUDIO_FEATURE_NAMES
            ], dtype=np.float64)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"Invalid audio features for {sample_id}"
            ) from exc

        if not np.all(np.isfinite(audio_values)):
            raise ValueError(
                f"Non-finite audio feature values for {sample_id}"
            )

        visual_values = extract_visual_features(
            sample.get("image_paths", [])
        )

        if visual_values is None:
            missing_visual_features += 1
            continue

        records.append({
            "sample_id": sample_id,
            "video_id": str(sample.get("video_id") or "").strip(),
            "text": text,
            "label": label,
            "audio": audio_values,
            "visual": visual_values,
        })

    records = [
        record for record in records
        if record["video_id"]
    ]

    if not records:
        raise ValueError("No complete multimodal samples were found.")

    if len({record["label"] for record in records}) != 2:
        raise ValueError("The complete dataset must contain both classes.")

    if len({record["video_id"] for record in records}) < N_SPLITS:
        raise ValueError("Not enough source videos for GroupKFold.")

    return (
        records,
        {
            "pilot_samples": len(samples),
            "annotation_records": len(annotations),
            "samples_with_all_modalities": len(records),
            "missing_text_samples": missing_text,
            "missing_audio_feature_samples": missing_audio_features,
            "missing_visual_feature_samples": missing_visual_features,
        },
    )


def build_input(records, components):
    """Construct inputs for a specific modality combination."""
    texts = [record["text"] for record in records]
    numeric_blocks = []

    if "audio" in components:
        numeric_blocks.append(
            np.vstack([record["audio"] for record in records])
        )

    if "visual" in components:
        numeric_blocks.append(
            np.vstack([record["visual"] for record in records])
        )

    if not numeric_blocks:
        return np.asarray(texts, dtype=object)

    numeric = np.hstack(numeric_blocks)

    if "text" not in components:
        return numeric

    # Column 0 contains text; remaining columns contain numeric features.
    combined = np.empty(
        (len(records), 1 + numeric.shape[1]),
        dtype=object,
    )

    combined[:, 0] = texts
    combined[:, 1:] = numeric

    return combined


def build_model(components, numeric_count):
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    if components == ["text"]:
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
            ("classifier", classifier),
        ])

    if "text" not in components:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])

    transformers = [
        (
            "text",
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
            "numeric",
            StandardScaler(),
            list(range(1, numeric_count + 1)),
        ),
    ]

    return Pipeline([
        (
            "features",
            ColumnTransformer(
                transformers,
                remainder="drop",
            ),
        ),
        ("classifier", classifier),
    ])


def calculate_metrics(y_true, y_pred, probabilities):
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
                [
                    1 if label == "humorous" else 0
                    for label in y_true
                ],
                probabilities,
            )
        ),
    }


def main():
    records, data_summary = load_records()

    texts = [record["text"] for record in records]
    y = [record["label"] for record in records]
    groups = [record["video_id"] for record in records]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = OUTPUT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- MMHD Multimodal Comparison ---")
    for key, value in data_summary.items():
        print(f"{key}: {value}")

    print(f"Distinct source videos: {len(set(groups))}")
    print(f"Label counts: {dict(Counter(y))}")
    print("Evaluation: 5-fold GroupKFold by video_id")

    # The same group splits are reused for every modality combination.
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(
        splitter.split(
            np.zeros((len(records), 1)),
            y,
            groups,
        )
    )

    all_results = {}
    all_predictions = {}

    for model_name, components in MODEL_COMPONENTS.items():
        print(f"\n--- {model_name} ---")

        X = build_input(records, components)
        numeric_count = (
            X.shape[1] - 1
            if "text" in components and len(components) > 1
            else X.shape[1] if "text" not in components else 0
        )

        predictions = [None] * len(records)
        probabilities = [None] * len(records)
        folds_results = []

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

            model = build_model(components, numeric_count)

            model.fit(X[train_idx], y_train)

            fold_predictions = model.predict(X[test_idx])
            fold_probabilities = model.predict_proba(X[test_idx])

            class_order = list(
                model.named_steps["classifier"].classes_
            )

            humorous_index = class_order.index("humorous")

            p_humorous = fold_probabilities[:, humorous_index]

            for local_index, original_index in enumerate(test_idx):
                predictions[original_index] = str(
                    fold_predictions[local_index]
                )
                probabilities[original_index] = float(
                    p_humorous[local_index]
                )

            fold_accuracy = accuracy_score(
                y_test, fold_predictions
            )

            folds_results.append({
                "fold": fold_number,
                "training_samples": len(train_idx),
                "validation_samples": len(test_idx),
                "training_videos": len(train_videos),
                "validation_videos": len(test_videos),
                "accuracy": float(fold_accuracy),
            })

            print(
                f"Fold {fold_number}/{N_SPLITS}: "
                f"accuracy={fold_accuracy:.3f}"
            )

        if any(value is None for value in predictions):
            raise RuntimeError("Some samples have no predictions.")

        if any(value is None for value in probabilities):
            raise RuntimeError("Some samples have no probabilities.")

        metrics = calculate_metrics(y, predictions, probabilities)

        matrix = confusion_matrix(
            y, predictions, labels=LABELS
        )

        # Fit a separate final model on all complete pilot samples.
        final_model = build_model(components, numeric_count)
        final_model.fit(X, y)

        model_file = (
            models_dir
            / (
                model_name.lower()
                .replace(" + ", "_")
                .replace("-", "_")
                .replace(" ", "_")
                + ".joblib"
            )
        )

        joblib.dump(final_model, model_file)

        all_results[model_name] = {
            "components": components,
            "out_of_fold_metrics": metrics,
            "confusion_matrix_label_order": LABELS,
            "confusion_matrix": matrix.tolist(),
            "folds": folds_results,
            "saved_model": str(model_file.relative_to(ROOT)),
        }

        all_predictions[model_name] = {
            "predictions": predictions,
            "probabilities": probabilities,
        }

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    # A majority-class baseline calculated on the same grouped folds.
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

    results = {
        "experiment": (
            "Text, audio and visual handcrafted-feature baseline comparison"
        ),
        "evaluation_method": (
            "5-fold GroupKFold grouped by source video; "
            "all modalities use the same complete-case samples and folds"
        ),
        "positive_class": "humorous",
        "dataset": data_summary,
        "label_distribution": dict(Counter(y)),
        "majority_class_baseline": majority_metrics,
        "models": all_results,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation_note": (
            "Visual features are low-level keyframe statistics, "
            "not learned semantic image embeddings. The multimodal "
            "systems are baseline feature-fusion experiments."
        ),
    }

    with (OUTPUT_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    predictions_file = OUTPUT_DIR / "oof_predictions.csv"

    with predictions_file.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fieldnames = [
            "sample_id",
            "video_id",
            "true_label",
        ]

        for model_name in MODEL_COMPONENTS:
            safe_name = (
                model_name.lower()
                .replace(" + ", "_")
                .replace("-", "_")
                .replace(" ", "_")
            )
            fieldnames.extend([
                f"{safe_name}_prediction",
                f"{safe_name}_humorous_probability",
            ])

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, record in enumerate(records):
            row = {
                "sample_id": record["sample_id"],
                "video_id": record["video_id"],
                "true_label": y[index],
            }

            for model_name, output in all_predictions.items():
                safe_name = (
                    model_name.lower()
                    .replace(" + ", "_")
                    .replace("-", "_")
                    .replace(" ", "_")
                )

                row[f"{safe_name}_prediction"] = (
                    output["predictions"][index]
                )

                row[f"{safe_name}_humorous_probability"] = round(
                    output["probabilities"][index], 6
                )

            writer.writerow(row)

    print("\n--- Summary: out-of-fold results ---")
    print(
        f"{'Model':<25} {'Accuracy':>10} "
        f"{'Balanced Acc.':>14} {'Macro F1':>10} {'Humor F1':>10}"
    )

    for model_name, output in all_results.items():
        metrics = output["out_of_fold_metrics"]

        print(
            f"{model_name:<25} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['balanced_accuracy']:>14.4f} "
            f"{metrics['macro_f1']:>10.4f} "
            f"{metrics['f1_humorous']:>10.4f}"
        )

    print("\nMajority-class baseline:")
    for name, value in majority_metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"\nMetrics: {OUTPUT_DIR / 'metrics.json'}")
    print(f"Predictions: {predictions_file}")
    print(f"Models: {models_dir}")


if __name__ == "__main__":
    main()