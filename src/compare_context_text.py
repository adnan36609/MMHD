import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
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

SAMPLES_FILE = ROOT / "annotations" / "pilot" / "pilot_samples.json"
ANNOTATIONS_FILE = (
    ROOT
    / "annotations"
    / "pilot"
    / "pilot_annotations_second_completed.json"
)

CLIPS_DIR = ROOT / "sample_clips"
OUTPUT_DIR = ROOT / "results" / "context_experiment"

LABELS = ["non_humorous", "humorous"]
N_SPLITS = 5
RANDOM_STATE = 42

SCENE_PATTERN = re.compile(r"Scene-(\d+)")


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def scene_number(value):
    """Extract the numeric scene index from a scene or sample ID."""
    if not value:
        return None

    match = SCENE_PATTERN.search(str(value))
    return int(match.group(1)) if match else None


def get_record_text(record):
    """Read transcript text from a full dataset.json record."""
    modalities = record.get("modalities", {})
    text_data = modalities.get("text", {})

    if isinstance(text_data, dict):
        value = text_data.get("value")
        if value is not None:
            return " ".join(str(value).split())

    return " ".join(str(record.get("text") or "").split())


def load_video_scene_maps(video_ids):
    """
    Load existing per-video dataset.json files and create a map
    from scene number to transcript.

    If duplicate entries exist for the same scene number, prefer
    a non-window scene record with usable text.
    """
    scene_maps = {}
    videos_loaded = 0
    missing_dataset_files = []

    for video_id in sorted(set(video_ids)):
        dataset_path = CLIPS_DIR / video_id / "dataset.json"

        if not dataset_path.is_file():
            missing_dataset_files.append(video_id)
            continue

        raw = read_json(dataset_path)

        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            records = next(
                (
                    raw[key]
                    for key in ("records", "samples", "scenes")
                    if isinstance(raw.get(key), list)
                ),
                None,
            )

            if records is None:
                raise ValueError(
                    f"Unexpected JSON structure in {dataset_path}"
                )
        else:
            raise ValueError(
                f"Unexpected JSON structure in {dataset_path}"
            )

        scene_map = {}

        for record in records:
            sample_id = str(record.get("sample_id") or "")

            source = record.get("source", {})
            source_scene = str(
                source.get("source_scene") or ""
            )

            number = scene_number(source_scene)

            if number is None:
                number = scene_number(sample_id)

            if number is None:
                continue

            text = get_record_text(record)

            # Prefer a base scene over a windowed candidate.
            is_window = (
                "Window-" in sample_id
                or "Window-" in source_scene
            )

            score = (
                0 if is_window else 1,
                1 if text else 0,
                len(text),
            )

            previous = scene_map.get(number)

            if previous is None or score > previous["score"]:
                scene_map[number] = {
                    "sample_id": sample_id,
                    "text": text,
                    "score": score,
                }

        scene_maps[video_id] = scene_map
        videos_loaded += 1

    return scene_maps, videos_loaded, missing_dataset_files


def find_neighbor_context(scene_map, target_number):
    """
    Find the nearest earlier and later available scenes by scene
    number. This is a scene-order proxy, not verified dialogue order.
    """
    if target_number is None or not scene_map:
        return {
            "previous_scene_id": "",
            "previous_text": "",
            "next_scene_id": "",
            "next_text": "",
        }

    earlier = [
        number for number in scene_map
        if number < target_number
    ]

    later = [
        number for number in scene_map
        if number > target_number
    ]

    previous_number = max(earlier) if earlier else None
    next_number = min(later) if later else None

    previous = (
        scene_map.get(previous_number)
        if previous_number is not None
        else None
    )

    following = (
        scene_map.get(next_number)
        if next_number is not None
        else None
    )

    return {
        "previous_scene_id": (
            previous["sample_id"] if previous else ""
        ),
        "previous_text": previous["text"] if previous else "",
        "next_scene_id": (
            following["sample_id"] if following else ""
        ),
        "next_text": following["text"] if following else "",
    }


def build_models():
    """Create target-only and context-augmented text classifiers."""
    target_model = Pipeline([
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

    contextual_features = ColumnTransformer(
        transformers=[
            (
                "target_text",
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
                "neighbor_context",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
                1,
            ),
        ],
        remainder="drop",
    )

    contextual_model = Pipeline([
        ("features", contextual_features),
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
        "Target text only": target_model,
        "Target + neighboring-scene text": contextual_model,
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
    """Obtain predictions for every sample from a held-out-video fold."""
    oof_predictions = [None] * len(y)
    oof_probabilities = [None] * len(y)
    fold_results = []

    for fold_number, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):
        train_videos = {groups[i] for i in train_idx}
        test_videos = {groups[i] for i in test_idx}

        if train_videos & test_videos:
            raise RuntimeError(
                f"Source-video leakage detected in fold {fold_number}."
            )

        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        if set(y_train) != set(LABELS):
            raise ValueError(
                f"Training fold {fold_number} lacks a class."
            )

        model = clone(model_template)
        model.fit(X[train_idx], y_train)

        predictions = model.predict(X[test_idx])
        probabilities = model.predict_proba(X[test_idx])

        classes = list(model.named_steps["classifier"].classes_)
        humorous_index = classes.index("humorous")

        humorous_probabilities = probabilities[:, humorous_index]

        for local_index, original_index in enumerate(test_idx):
            oof_predictions[original_index] = str(
                predictions[local_index]
            )
            oof_probabilities[original_index] = float(
                humorous_probabilities[local_index]
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
        raise RuntimeError("Some samples have no out-of-fold predictions.")

    metrics = calculate_metrics(
        y,
        oof_predictions,
        oof_probabilities,
    )

    matrix = confusion_matrix(
        y,
        oof_predictions,
        labels=LABELS,
    )

    return (
        metrics,
        oof_predictions,
        oof_probabilities,
        matrix.tolist(),
        fold_results,
    )


def main():
    samples = read_json(SAMPLES_FILE)
    annotations = read_json(ANNOTATIONS_FILE)

    samples_by_id = {
        sample["sample_id"]: sample for sample in samples
    }

    annotation_ids = [
        annotation["sample_id"] for annotation in annotations
    ]

    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Duplicate annotation IDs detected.")

    if set(annotation_ids) != set(samples_by_id):
        raise ValueError(
            "Annotation IDs do not match the pilot sample IDs."
        )

    # Build context from the complete per-video extracted datasets.
    video_ids = [
        sample.get("video_id")
        for sample in samples
        if sample.get("video_id")
    ]

    scene_maps, videos_loaded, missing_dataset_files = (
        load_video_scene_maps(video_ids)
    )

    records = []

    for annotation in annotations:
        sample_id = annotation["sample_id"]
        sample = samples_by_id[sample_id]

        label = annotation.get("humor_label")

        if label not in LABELS:
            raise ValueError(
                f"Invalid label for {sample_id}: {label!r}"
            )

        target_text = " ".join(
            str(sample.get("text") or "").split()
        )

        # Match the existing text experiment's usable subset.
        if not target_text:
            continue

        video_id = str(sample.get("video_id") or "").strip()

        if not video_id:
            raise ValueError(f"Missing video_id for {sample_id}")

        number = scene_number(sample_id)

        if number is None:
            number = scene_number(sample.get("source_scene"))

        context = find_neighbor_context(
            scene_maps.get(video_id, {}),
            number,
        )

        previous_text = context["previous_text"]
        next_text = context["next_text"]

        combined_context = (
            f"previous_scene {previous_text} "
            f"next_scene {next_text}"
        ).strip()

        if not previous_text and not next_text:
            combined_context = "no_neighbor_context"

        records.append({
            "sample_id": sample_id,
            "video_id": video_id,
            "scene_number": number,
            "target_text": target_text,
            "previous_scene_id": context["previous_scene_id"],
            "previous_text": previous_text,
            "next_scene_id": context["next_scene_id"],
            "next_text": next_text,
            "context_text": combined_context,
            "label": label,
        })

    if len(records) < N_SPLITS:
        raise ValueError("Not enough labeled text samples.")

    if len({record["video_id"] for record in records}) < N_SPLITS:
        raise ValueError("Not enough source videos for GroupKFold.")

    target_texts = np.asarray(
        [record["target_text"] for record in records],
        dtype=object,
    )

    contextual_texts = np.asarray(
        [
            [record["target_text"], record["context_text"]]
            for record in records
        ],
        dtype=object,
    )

    y = [record["label"] for record in records]
    groups = [record["video_id"] for record in records]

    previous_available = sum(
        bool(record["previous_text"]) for record in records
    )
    next_available = sum(
        bool(record["next_text"]) for record in records
    )
    either_available = sum(
        bool(record["previous_text"] or record["next_text"])
        for record in records
    )
    both_available = sum(
        bool(record["previous_text"] and record["next_text"])
        for record in records
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the contextual text audit for manual inspection.
    context_audit = OUTPUT_DIR / "context_audit.csv"

    with context_audit.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fieldnames = [
            "sample_id",
            "video_id",
            "scene_number",
            "label",
            "previous_scene_id",
            "previous_text",
            "target_text",
            "next_scene_id",
            "next_text",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow({
                key: record[key] for key in fieldnames
            })

    print("\n--- MMHD Context Experiment ---")
    print(f"Pilot samples: {len(samples)}")
    print(f"Annotation records: {len(annotations)}")
    print(f"Samples with target text: {len(records)}")
    print(f"Distinct source videos: {len(set(groups))}")
    print(f"Dataset files loaded: {videos_loaded}")
    print(f"Videos missing dataset.json: {len(missing_dataset_files)}")
    print(f"Samples with previous-scene text: {previous_available}")
    print(f"Samples with next-scene text: {next_available}")
    print(f"Samples with either context: {either_available}")
    print(f"Samples with both contexts: {both_available}")
    print(f"Label distribution: {dict(Counter(y))}")
    print("Evaluation: 5-fold GroupKFold by source video")

    if either_available < len(records) // 2:
        print(
            "WARNING: Fewer than half the samples have neighboring text. "
            "Interpret the context experiment cautiously."
        )

    splitter = GroupKFold(n_splits=N_SPLITS)

    splits = list(
        splitter.split(
            np.zeros((len(records), 1)),
            y,
            groups,
        )
    )

    model_templates = build_models()
    all_results = {}
    all_predictions = {}

    inputs = {
        "Target text only": target_texts,
        "Target + neighboring-scene text": contextual_texts,
    }

    for model_name, model_template in model_templates.items():
        print(f"\n--- {model_name} ---")

        (
            metrics,
            predictions,
            probabilities,
            matrix,
            folds,
        ) = evaluate_model(
            model_template,
            inputs[model_name],
            y,
            groups,
            splits,
        )

        all_results[model_name] = {
            "out_of_fold_metrics": metrics,
            "confusion_matrix_label_order": LABELS,
            "confusion_matrix": matrix,
            "folds": folds,
        }

        all_predictions[model_name] = {
            "predictions": predictions,
            "probabilities": probabilities,
        }

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # Majority-class baseline using the same folds.
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

    # Train demonstration models on all eligible pilot samples.
    for model_name, model_template in model_templates.items():
        final_model = clone(model_template)
        final_model.fit(inputs[model_name], y)

        filename = (
            "model_target_text.joblib"
            if model_name == "Target text only"
            else "model_with_neighbor_context.joblib"
        )

        joblib.dump(final_model, OUTPUT_DIR / filename)

    metrics_output = {
        "experiment": (
            "Neighboring-scene context augmentation for humor recognition"
        ),
        "context_definition": (
            "Text from the nearest available preceding and following "
            "scene indices in the per-video dataset.json files. This is "
            "a scene-order proxy, not verified conversational turn data."
        ),
        "evaluation_method": (
            "Five-fold GroupKFold by video_id. Both classifiers use "
            "the same labeled samples and the same folds."
        ),
        "dataset": {
            "pilot_samples": len(samples),
            "annotation_records": len(annotations),
            "samples_with_text": len(records),
            "source_videos": len(set(groups)),
            "video_dataset_files_loaded": videos_loaded,
            "videos_missing_dataset_file": missing_dataset_files,
            "samples_with_previous_text": previous_available,
            "samples_with_next_text": next_available,
            "samples_with_either_context": either_available,
            "samples_with_both_contexts": both_available,
            "label_distribution": dict(Counter(y)),
        },
        "majority_class_baseline": majority_metrics,
        "models": all_results,
        "software": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "Neighboring scene order is only a proxy for conversational context.",
            "Adjacent scenes may not be semantically related.",
            "The labeled pilot contains 240 samples, with fewer usable text samples.",
        ],
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_output, file, indent=2)

    predictions_path = OUTPUT_DIR / "oof_predictions.csv"

    with predictions_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fieldnames = [
            "sample_id",
            "video_id",
            "true_label",
            "previous_text",
            "target_text",
            "next_text",
            "target_only_prediction",
            "target_only_humorous_probability",
            "context_model_prediction",
            "context_model_humorous_probability",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, record in enumerate(records):
            writer.writerow({
                "sample_id": record["sample_id"],
                "video_id": record["video_id"],
                "true_label": y[index],
                "previous_text": record["previous_text"],
                "target_text": record["target_text"],
                "next_text": record["next_text"],
                "target_only_prediction": (
                    all_predictions["Target text only"]["predictions"][index]
                ),
                "target_only_humorous_probability": round(
                    all_predictions["Target text only"]["probabilities"][index],
                    6,
                ),
                "context_model_prediction": (
                    all_predictions[
                        "Target + neighboring-scene text"
                    ]["predictions"][index]
                ),
                "context_model_humorous_probability": round(
                    all_predictions[
                        "Target + neighboring-scene text"
                    ]["probabilities"][index],
                    6,
                ),
            })

    print("\n--- Summary ---")
    print(
        f"{'Model':<35} {'Accuracy':>10} "
        f"{'Balanced Acc.':>14} {'Macro F1':>10}"
    )

    for name, result in all_results.items():
        metrics = result["out_of_fold_metrics"]

        print(
            f"{name:<35} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['balanced_accuracy']:>14.4f} "
            f"{metrics['macro_f1']:>10.4f}"
        )

    print("\nMajority-class baseline:")
    for name, value in majority_metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"\nMetrics: {metrics_path}")
    print(f"Context audit: {context_audit}")
    print(f"Predictions: {predictions_path}")
    print(f"Saved models: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()