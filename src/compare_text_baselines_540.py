import json
from collections import Counter
from pathlib import Path

import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = ROOT / "annotations" / "expanded_540" / "samples_540.json"
ANNOTATIONS_FILE = ROOT / "annotations" / "expanded_540" / "annotations_540.json"

OUTPUT_DIR = ROOT / "results" / "expanded_540" / "text_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["non_humorous", "humorous"]
N_SPLITS = 5
RANDOM_STATE = 42


def load_data():
    with SAMPLES_FILE.open("r", encoding="utf-8-sig") as f:
        samples = json.load(f)

    with ANNOTATIONS_FILE.open("r", encoding="utf-8-sig") as f:
        annotations = json.load(f)

    sample_by_id = {s["sample_id"]: s for s in samples}

    if len(sample_by_id) != len(samples):
        raise ValueError("Duplicate sample IDs in pilot_samples.json.")

    annotation_ids = [a["sample_id"] for a in annotations]

    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Duplicate sample IDs in annotations.")

    if set(annotation_ids) != set(sample_by_id):
        raise ValueError("Annotation IDs do not match pilot sample IDs.")

    records = []

    for annotation in annotations:
        sample_id = annotation["sample_id"]
        label = annotation.get("humor_label")
        sample = sample_by_id[sample_id]
        text = " ".join(str(sample.get("text") or "").split())
        video_id = str(sample.get("video_id") or "").strip()

        if label not in LABELS:
            raise ValueError(f"Invalid label for {sample_id}: {label!r}")

        if text and video_id:
            records.append({
                "sample_id": sample_id,
                "video_id": video_id,
                "text": text,
                "label": label,
            })

    return records


def build_models():
    logistic_regression = Pipeline([
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

    word_char_svm = Pipeline([
        (
            "features",
            FeatureUnion([
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]),
        ),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    return {
        "TF-IDF + Logistic Regression": logistic_regression,
        "Word + Character TF-IDF + Linear SVC": word_char_svm,
    }


def evaluate_model(model_template, texts, labels, groups):
    predictions = [None] * len(labels)
    splitter = GroupKFold(n_splits=N_SPLITS)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(texts, labels, groups),
        start=1,
    ):
        train_videos = {groups[i] for i in train_idx}
        test_videos = {groups[i] for i in test_idx}

        if train_videos & test_videos:
            raise RuntimeError("Source-video leakage detected.")

        model = clone(model_template)

        model.fit(
            [texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
        )

        fold_predictions = model.predict(
            [texts[i] for i in test_idx]
        )

        for index, prediction in zip(test_idx, fold_predictions):
            predictions[index] = str(prediction)

        fold_accuracy = accuracy_score(
            [labels[i] for i in test_idx],
            fold_predictions,
        )

        print(
            f"    Fold {fold}/{N_SPLITS}: "
            f"accuracy={fold_accuracy:.3f}"
        )

    if any(p is None for p in predictions):
        raise RuntimeError("Some samples did not receive predictions.")

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "humorous_f1": float(
            f1_score(
                labels,
                predictions,
                pos_label="humorous",
                zero_division=0,
            )
        ),
    }

    return metrics, predictions


def main():
    records = load_data()

    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    groups = [r["video_id"] for r in records]

    if len(set(groups)) < N_SPLITS:
        raise ValueError("Not enough source videos for GroupKFold.")

    print("\n--- MMHD Text Model Comparison ---")
    print(f"Samples: {len(records)}")
    print(f"Source videos: {len(set(groups))}")
    print(f"Label counts: {dict(Counter(labels))}")
    print("Evaluation: 5-fold GroupKFold by source video")

    results = {}
    all_predictions = {}

    for name, model_template in build_models().items():
        print(f"\n{name}")

        metrics, predictions = evaluate_model(
            model_template, texts, labels, groups
        )

        results[name] = metrics
        all_predictions[name] = predictions

        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    output = {
        "evaluation_method": (
            "5-fold GroupKFold grouped by video_id; "
            "out-of-fold predictions"
        ),
        "samples": len(records),
        "source_videos": len(set(groups)),
        "label_distribution": dict(Counter(labels)),
        "scikit_learn_version": sklearn.__version__,
        "results": results,
    }

    with (OUTPUT_DIR / "model_comparison.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, indent=2)

    with (OUTPUT_DIR / "comparison_predictions.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            [
                {
                    "sample_id": records[i]["sample_id"],
                    "video_id": records[i]["video_id"],
                    "true_label": labels[i],
                    "logistic_regression_prediction": (
                        all_predictions["TF-IDF + Logistic Regression"][i]
                    ),
                    "word_char_svm_prediction": (
                        all_predictions[
                            "Word + Character TF-IDF + Linear SVC"
                        ][i]
                    ),
                }
                for i in range(len(records))
            ],
            f,
            indent=2,
        )

    print(f"\nSaved comparison results to: {OUTPUT_DIR}")
    print("Files: model_comparison.json, comparison_predictions.json")


if __name__ == "__main__":
    main()