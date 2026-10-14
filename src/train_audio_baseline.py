import csv
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.io import wavfile

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
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

ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = ROOT / "annotations" / "pilot" / "pilot_samples.json"
ANNOTATIONS_FILE = (
    ROOT
    / "annotations"
    / "pilot"
    / "pilot_annotations_second_completed.json"
)

OUTPUT_DIR = ROOT / "results" / "audio_baseline"

LABELS = ["non_humorous", "humorous"]
N_SPLITS = 5
RANDOM_STATE = 42


def resolve_path(value):
    """Resolve a dataset path relative to the project root."""
    if not value:
        return None

    path = Path(str(value).replace("/", "\\"))

    if not path.is_absolute():
        path = ROOT / path

    return path


def audio_to_float(signal, original_dtype):
    """Convert integer or floating-point PCM samples to float values."""
    x = np.asarray(signal, dtype=np.float64)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)

        if np.issubdtype(original_dtype, np.unsignedinteger):
            midpoint = (info.max + 1) / 2.0
            x = (x - midpoint) / midpoint
        else:
            scale = float(max(abs(info.min), abs(info.max)))
            x = x / scale

    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def extract_audio_features(path):
    """
    Extract simple acoustic statistics from a WAV file.

    These are low-level audio features, not semantic understanding
    of speech or a pretrained speech representation.
    """
    sample_rate, signal = wavfile.read(path)

    original_dtype = signal.dtype

    # Convert stereo audio to mono.
    if signal.ndim == 2:
        signal = signal.astype(np.float64).mean(axis=1)

    x = audio_to_float(signal, original_dtype)

    if x.size == 0:
        raise ValueError("Empty audio file")

    duration = len(x) / float(sample_rate)

    # Use 25 ms frames with a 10 ms hop.
    frame_length = max(1, int(round(sample_rate * 0.025)))
    hop_length = max(1, int(round(sample_rate * 0.010)))

    if len(x) < frame_length:
        x = np.pad(x, (0, frame_length - len(x)))

    frames = np.lib.stride_tricks.sliding_window_view(
        x, frame_length
    )[::hop_length]

    # Time-domain features.
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    sign_changes = (
        np.diff(np.signbit(frames), axis=1)
    )
    zero_crossing_rate = np.mean(sign_changes, axis=1)

    # Frequency-domain features.
    window = np.hanning(frame_length)

    centered_frames = frames - np.mean(
        frames, axis=1, keepdims=True
    )

    spectrum = np.abs(
        np.fft.rfft(centered_frames * window, axis=1)
    )

    frequencies = np.fft.rfftfreq(
        frame_length, d=1.0 / float(sample_rate)
    )

    epsilon = 1e-12
    magnitude_sum = np.sum(spectrum, axis=1) + epsilon

    centroid = (
        np.sum(spectrum * frequencies, axis=1)
        / magnitude_sum
    )

    bandwidth = np.sqrt(
        np.sum(
            spectrum * (frequencies[None, :] - centroid[:, None]) ** 2,
            axis=1,
        ) / magnitude_sum
    )

    cumulative_magnitude = np.cumsum(spectrum, axis=1)
    thresholds = 0.85 * magnitude_sum

    rolloff_indices = np.argmax(
        cumulative_magnitude >= thresholds[:, None],
        axis=1,
    )

    rolloff = frequencies[rolloff_indices]

    spectral_flatness = (
        np.exp(np.mean(np.log(spectrum + epsilon), axis=1))
        / (np.mean(spectrum, axis=1) + epsilon)
    )

    features = {
        "duration_seconds": duration,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_p90": float(np.percentile(rms, 90)),
        "zero_crossing_mean": float(np.mean(zero_crossing_rate)),
        "zero_crossing_std": float(np.std(zero_crossing_rate)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "spectral_flatness_std": float(np.std(spectral_flatness)),
        "silence_frame_ratio": float(np.mean(rms < 0.01)),
    }

    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("Non-finite audio features")

    return features


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def main():
    with SAMPLES_FILE.open("r", encoding="utf-8-sig") as file:
        samples = json.load(file)

    with ANNOTATIONS_FILE.open("r", encoding="utf-8-sig") as file:
        annotations = json.load(file)

    samples_by_id = {
        sample["sample_id"]: sample for sample in samples
    }

    if len(samples_by_id) != len(samples):
        raise ValueError("Duplicate IDs in pilot_samples.json")

    annotation_ids = [a["sample_id"] for a in annotations]

    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Duplicate IDs in the annotation file")

    if set(annotation_ids) != set(samples_by_id):
        raise ValueError("Annotation IDs do not match pilot sample IDs")

    records = []
    missing_audio = []
    failed_audio = []

    print("\n--- Preparing audio features ---")

    for annotation in annotations:
        sample_id = annotation["sample_id"]
        label = annotation.get("humor_label")
        sample = samples_by_id[sample_id]

        if label not in LABELS:
            raise ValueError(
                f"Invalid label for {sample_id}: {label!r}"
            )

        audio_path = resolve_path(sample.get("audio_path"))

        if audio_path is None or not audio_path.is_file():
            missing_audio.append(sample_id)
            continue

        try:
            features = extract_audio_features(audio_path)
        except Exception as exc:
            failed_audio.append({
                "sample_id": sample_id,
                "error": str(exc),
            })
            continue

        video_id = str(sample.get("video_id") or "").strip()

        if not video_id:
            raise ValueError(f"Missing video_id for {sample_id}")

        records.append({
            "sample_id": sample_id,
            "video_id": video_id,
            "label": label,
            **features,
        })

    if len(records) < N_SPLITS:
        raise ValueError("Not enough usable audio samples")

    feature_names = list(
        extract_audio_features.__annotations__
    ) if False else [
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

    X = np.asarray(
        [
            [record[name] for name in feature_names]
            for record in records
        ],
        dtype=np.float64,
    )

    y = [record["label"] for record in records]
    groups = [record["video_id"] for record in records]

    if set(y) != set(LABELS):
        raise ValueError("Usable audio data must contain both labels")

    if len(set(groups)) < N_SPLITS:
        raise ValueError(
            f"At least {N_SPLITS} source videos are required"
        )

    print(f"Annotated samples: {len(annotations)}")
    print(f"Usable audio samples: {len(records)}")
    print(f"Missing audio files: {len(missing_audio)}")
    print(f"Audio files that failed processing: {len(failed_audio)}")
    print(f"Distinct source videos: {len(set(groups))}")
    print(f"Label counts: {dict(Counter(y))}")
    print(f"Extracted features per sample: {len(feature_names)}")

    # Save extracted acoustic features for reproducibility.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_csv = OUTPUT_DIR / "audio_features.csv"

    with feature_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "video_id",
                "label",
                *feature_names,
            ],
        )
        writer.writeheader()

        for record in records:
            writer.writerow(record)

    # Five-fold cross-validation, grouped by source video.
    splitter = GroupKFold(n_splits=N_SPLITS)

    oof_predictions = [None] * len(records)
    oof_probabilities = [None] * len(records)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        train_videos = {groups[i] for i in train_idx}
        test_videos = {groups[i] for i in test_idx}

        if train_videos & test_videos:
            raise RuntimeError("Source-video leakage detected")

        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        if set(y_train) != set(LABELS):
            raise ValueError(
                f"Training fold {fold} lacks one of the classes"
            )

        model = build_model()
        model.fit(X[train_idx], y_train)

        predictions = model.predict(X[test_idx])
        class_order = list(model.named_steps["classifier"].classes_)
        humorous_index = class_order.index("humorous")

        probabilities = model.predict_proba(X[test_idx])[
            :, humorous_index
        ]

        for local_index, original_index in enumerate(test_idx):
            oof_predictions[original_index] = str(
                predictions[local_index]
            )
            oof_probabilities[original_index] = float(
                probabilities[local_index]
            )

        fold_accuracy = accuracy_score(y_test, predictions)

        fold_results.append({
            "fold": fold,
            "training_samples": len(train_idx),
            "validation_samples": len(test_idx),
            "training_videos": len(train_videos),
            "validation_videos": len(test_videos),
            "accuracy": float(fold_accuracy),
        })

        print(
            f"Fold {fold}/{N_SPLITS}: "
            f"train={len(train_idx)}, "
            f"validation={len(test_idx)}, "
            f"accuracy={fold_accuracy:.3f}"
        )

    if any(value is None for value in oof_predictions):
        raise RuntimeError("Some samples lack out-of-fold predictions")

    if any(value is None for value in oof_probabilities):
        raise RuntimeError("Some samples lack out-of-fold probabilities")

    metrics = {
        "accuracy": float(accuracy_score(y, oof_predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, oof_predictions)
        ),
        "macro_f1": float(
            f1_score(
                y,
                oof_predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_humorous": float(
            precision_score(
                y,
                oof_predictions,
                pos_label="humorous",
                zero_division=0,
            )
        ),
        "recall_humorous": float(
            recall_score(
                y,
                oof_predictions,
                pos_label="humorous",
                zero_division=0,
            )
        ),
        "f1_humorous": float(
            f1_score(
                y,
                oof_predictions,
                pos_label="humorous",
                zero_division=0,
            )
        ),
        "roc_auc_humorous": float(
            roc_auc_score(
                [1 if label == "humorous" else 0 for label in y],
                oof_probabilities,
            )
        ),
    }

    classification_results = classification_report(
        y,
        oof_predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y, oof_predictions, labels=LABELS
    )

    # Majority-class baseline on the same grouped folds.
    majority_predictions = [None] * len(records)

    for train_idx, test_idx in splitter.split(X, y, groups):
        majority_class = Counter(
            y[i] for i in train_idx
        ).most_common(1)[0][0]

        for index in test_idx:
            majority_predictions[index] = majority_class

    majority_metrics = {
        "accuracy": float(accuracy_score(y, majority_predictions)),
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

    predictions_csv = OUTPUT_DIR / "oof_predictions.csv"

    with predictions_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "video_id",
                "true_label",
                "predicted_label",
                "humorous_probability",
            ],
        )
        writer.writeheader()

        for index, record in enumerate(records):
            writer.writerow({
                "sample_id": record["sample_id"],
                "video_id": record["video_id"],
                "true_label": y[index],
                "predicted_label": oof_predictions[index],
                "humorous_probability": round(
                    oof_probabilities[index], 6
                ),
            })

    # Fit a demonstration model on all usable audio samples.
    final_model = build_model()
    final_model.fit(X, y)

    import joblib
    joblib.dump(final_model, OUTPUT_DIR / "model.joblib")

    results = {
        "experiment": "Audio-only acoustic feature baseline",
        "method": (
            "13 hand-engineered audio features; "
            "StandardScaler + Logistic Regression"
        ),
        "dataset": {
            "annotations": len(annotations),
            "usable_audio_samples": len(records),
            "missing_audio": missing_audio,
            "failed_audio": failed_audio,
            "source_videos": len(set(groups)),
            "label_distribution": dict(Counter(y)),
        },
        "evaluation": {
            "method": f"{N_SPLITS}-fold GroupKFold by video_id",
            "out_of_fold_metrics": metrics,
            "majority_class_baseline": majority_metrics,
            "classification_report": classification_results,
            "confusion_matrix_label_order": LABELS,
            "confusion_matrix": matrix.tolist(),
            "folds": fold_results,
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    with (OUTPUT_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print("\n--- Audio-only out-of-fold evaluation ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n--- Majority-class baseline ---")
    for name, value in majority_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n--- Confusion matrix ---")
    print("Rows=true labels, columns=predicted labels")
    print(f"Label order: {LABELS}")
    print(matrix)

    print(f"\nSaved results to: {OUTPUT_DIR}")
    print(f"Metrics: {OUTPUT_DIR / 'metrics.json'}")
    print(f"Features: {feature_csv}")
    print(f"Predictions: {predictions_csv}")
    print(f"Model: {OUTPUT_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()