import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np

from train_audio_baseline_540 import extract_audio_features


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "full_corpus" / "humor_predictions.csv"
OUTPUT_FILE = ROOT / "results" / "full_corpus" / "humor_explanations.csv"

TEXT_MODEL_FILE = (
    ROOT / "results" / "expanded_540" / "text_baseline" / "model.joblib"
)
AUDIO_MODEL_FILE = (
    ROOT / "results" / "expanded_540" / "audio_baseline" / "model.joblib"
)

AUDIO_FEATURES = [
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

TOP_K = 3
POSITIVE_LABEL = "humorous"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_asset(value):
    if not value:
        return None

    candidate = Path(str(value))

    options = (
        [candidate]
        if candidate.is_absolute()
        else [ROOT / candidate, ROOT / "sample_clips" / candidate]
    )

    for path in options:
        if path.is_file():
            return path

    return None


def parse_probability(value):
    if value is None or str(value).strip() == "":
        return None

    result = float(value)

    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"Invalid probability: {value!r}")

    return result


def prediction_class(probability):
    if probability is None:
        return "unavailable"

    return (
        "humorous"
        if probability >= 0.5
        else "non_humorous"
    )


def top_contributions(names, values, top_k=TOP_K):
    """Return the strongest positive and negative model contributions."""
    positive = [
        {
            "feature": str(name),
            "contribution": round(float(value), 4),
        }
        for name, value in zip(names, values)
        if value > 0
    ]

    negative = [
        {
            "feature": str(name),
            "contribution": round(float(value), 4),
        }
        for name, value in zip(names, values)
        if value < 0
    ]

    positive.sort(
        key=lambda item: item["contribution"],
        reverse=True,
    )

    negative.sort(
        key=lambda item: item["contribution"]
    )

    return positive[:top_k], negative[:top_k]


def load_models():
    for path in (INPUT_FILE, TEXT_MODEL_FILE, AUDIO_MODEL_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)

    text_model = joblib.load(TEXT_MODEL_FILE)
    audio_model = joblib.load(AUDIO_MODEL_FILE)

    text_vectorizer = text_model.named_steps["tfidf"]
    text_classifier = text_model.named_steps["classifier"]

    audio_scaler = audio_model.named_steps["scaler"]
    audio_classifier = audio_model.named_steps["classifier"]

    # LogisticRegression's binary coefficient points toward classes_[1].
    # Reverse the sign if that class is not humorous.
    text_direction = (
        1.0
        if text_classifier.classes_[1] == POSITIVE_LABEL
        else -1.0
    )

    audio_direction = (
        1.0
        if audio_classifier.classes_[1] == POSITIVE_LABEL
        else -1.0
    )

    text_feature_names = text_vectorizer.get_feature_names_out()

    return {
        "text_model": text_model,
        "text_vectorizer": text_vectorizer,
        "text_classifier": text_classifier,
        "text_direction": text_direction,
        "text_feature_names": text_feature_names,
        "audio_model": audio_model,
        "audio_scaler": audio_scaler,
        "audio_classifier": audio_classifier,
        "audio_direction": audio_direction,
    }


def explain_text(text, models):
    if not text.strip():
        return [], []

    vectorizer = models["text_vectorizer"]
    classifier = models["text_classifier"]

    vector = vectorizer.transform([text])

    if vector.nnz == 0:
        return [], []

    # sklearn's binary logistic-regression coefficient corresponds
    # to classes_[1]. Convert its sign to the humorous class.
    indices = vector.indices
    weights = vector.data

    contributions = (
        weights
        * classifier.coef_[0][indices]
        * models["text_direction"]
    )

    active_names = models["text_feature_names"][indices]

    supporting, opposing = top_contributions(
        active_names,
        contributions,
    )

    return supporting, opposing


def explain_audio(audio_path, models):
    if not audio_path:
        return [], [], "no_audio_path"

    resolved = resolve_asset(audio_path)

    if resolved is None:
        return [], [], "audio_file_missing"

    try:
        features = extract_audio_features(resolved)

        vector = np.asarray(
            [[float(features[name]) for name in AUDIO_FEATURES]],
            dtype=np.float64,
        )

        if not np.isfinite(vector).all():
            raise ValueError("Non-finite audio features")

        scaler = models["audio_scaler"]
        classifier = models["audio_classifier"]

        standardized = scaler.transform(vector)[0]

        # Contributions are in the classifier's decision-score space,
        # oriented so positive values support the humorous class.
        contributions = (
            standardized
            * classifier.coef_[0]
            * models["audio_direction"]
        )

        supporting, opposing = top_contributions(
            AUDIO_FEATURES,
            contributions,
        )

        return supporting, opposing, ""

    except Exception as exc:
        message = f"{type(exc).__name__}: {str(exc)[:150]}"
        return [], [], message


def assess_uncertainty(
    humor_probability,
    text_probability,
    audio_probability,
):
    """
    Produce a transparent decision-strength heuristic.

    This is NOT a calibrated confidence estimate.
    """
    if humor_probability is None:
        return None, "unavailable", ["no_model_prediction"]

    decision_strength = min(
        1.0,
        2.0 * abs(humor_probability - 0.5),
    )

    flags = []

    if 0.45 <= humor_probability <= 0.55:
        flags.append("near_decision_threshold")

    disagreement = None

    if text_probability is not None and audio_probability is not None:
        disagreement = abs(text_probability - audio_probability)

        if disagreement >= 0.25:
            flags.append("text_audio_score_disagreement")

    elif text_probability is None or audio_probability is None:
        flags.append("single_modality_prediction")

    if decision_strength < 0.20:
        level = "low"
    elif decision_strength < 0.50:
        level = "moderate"
    else:
        level = "higher_decision_margin"

    if disagreement is not None and disagreement >= 0.25:
        level = "low"
    elif 0.45 <= humor_probability <= 0.55:
        level = "low"

    if not flags:
        flags.append("no_rule_based_uncertainty_flag")

    return (
        round(float(decision_strength), 4),
        level,
        flags,
    )


def make_explanation(
    prediction,
    probability,
    text_probability,
    audio_probability,
    text_supporting,
    text_opposing,
    audio_supporting,
    audio_opposing,
    uncertainty_flags,
):
    if probability is None:
        return (
            "No humor prediction is available because neither the "
            "text nor the audio classifier could score this sample."
        )

    parts = [
        f"The batch classifier predicts {prediction} "
        f"with a raw humor score of {probability:.3f}."
    ]

    if text_probability is not None:
        parts.append(
            f"The text classifier's humor score is "
            f"{text_probability:.3f}."
        )

        if text_supporting:
            terms = ", ".join(
                item["feature"] for item in text_supporting
            )
            parts.append(
                f"Transcript terms pushing the text model toward "
                f"humor include: {terms}."
            )

        if text_opposing:
            terms = ", ".join(
                item["feature"] for item in text_opposing
            )
            parts.append(
                f"Terms pushing its score away from humor include: {terms}."
            )

    if audio_probability is not None:
        parts.append(
            f"The audio classifier's humor score is "
            f"{audio_probability:.3f}."
        )

        if audio_supporting:
            features = ", ".join(
                item["feature"] for item in audio_supporting
            )
            parts.append(
                f"Acoustic features pushing the audio model toward "
                f"humor include: {features}."
            )

        if audio_opposing:
            features = ", ".join(
                item["feature"] for item in audio_opposing
            )
            parts.append(
                f"Acoustic features pushing its score away from "
                f"humor include: {features}."
            )

    if "text_audio_score_disagreement" in uncertainty_flags:
        parts.append(
            "The text and audio models disagree substantially; "
            "interpret the combined prediction cautiously."
        )

    if "near_decision_threshold" in uncertainty_flags:
        parts.append(
            "The combined score is close to the decision threshold."
        )

    parts.append(
        "These features describe associations with the trained "
        "classifier's decision. They do not prove why a human finds "
        "the scene funny."
    )

    return " ".join(parts)


def main():
    start_time = time.perf_counter()

    print("Loading corpus predictions and trained models...")

    rows = read_csv(INPUT_FILE)
    models = load_models()

    if not rows:
        raise ValueError("The prediction file is empty.")

    output_rows = []
    total = len(rows)

    print(f"Records to explain: {total:,}")

    for index, row in enumerate(rows, start=1):
        text = str(row.get("text") or "")

        text_probability = parse_probability(
            row.get("text_humorous_probability")
        )

        audio_probability = parse_probability(
            row.get("audio_humorous_probability")
        )

        humor_probability = parse_probability(
            row.get("humor_probability")
        )

        prediction = str(
            row.get("humor_prediction") or "unavailable"
        )

        if (
            humor_probability is not None
            and prediction != prediction_class(humor_probability)
        ):
            raise ValueError(
                f"Prediction/score mismatch for {row.get('sample_id')}"
            )

        text_supporting, text_opposing = explain_text(
            text,
            models,
        )

        # Recompute acoustic features only when the existing batch
        # prediction indicates that audio inference succeeded.
        audio_supporting = []
        audio_opposing = []
        audio_explanation_error = ""

        if audio_probability is not None:
            (
                audio_supporting,
                audio_opposing,
                audio_explanation_error,
            ) = explain_audio(
                row.get("audio_path", ""),
                models,
            )

        decision_strength, confidence_level, flags = assess_uncertainty(
            humor_probability,
            text_probability,
            audio_probability,
        )

        explanation = make_explanation(
            prediction,
            humor_probability,
            text_probability,
            audio_probability,
            text_supporting,
            text_opposing,
            audio_supporting,
            audio_opposing,
            flags,
        )

        # Target identification will be added after this evidence
        # generation step. Do not manufacture a target from model cues.
        output = dict(row)

        output.update({
            "text_supporting_terms": json.dumps(
                text_supporting,
                ensure_ascii=False,
            ),
            "text_opposing_terms": json.dumps(
                text_opposing,
                ensure_ascii=False,
            ),
            "audio_supporting_features": json.dumps(
                audio_supporting,
                ensure_ascii=False,
            ),
            "audio_opposing_features": json.dumps(
                audio_opposing,
                ensure_ascii=False,
            ),
            "explanation": explanation,
            "decision_strength_score": decision_strength,
            "confidence_level_heuristic": confidence_level,
            "confidence_calibrated": False,
            "uncertainty_flags": json.dumps(flags),
            "target_of_humor": "unknown",
            "target_status": "not_yet_implemented",
            "audio_explanation_error": audio_explanation_error,
            "explanation_scope": (
                "Text/audio model-decision evidence; "
                "not a causal explanation"
            ),
        })

        output_rows.append(output)

        if index % 500 == 0 or index == total:
            print(f"Explained: {index:,}/{total:,}")

    # Preserve original columns first, then append explanation fields.
    fieldnames = list(rows[0].keys())

    for row in output_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "records": len(output_rows),
        "records_with_predictions": sum(
            row["humor_prediction"] in ("humorous", "non_humorous")
            for row in output_rows
        ),
        "records_with_text_evidence": sum(
            bool(row["text_supporting_terms"] != "[]"
                 or row["text_opposing_terms"] != "[]")
            for row in output_rows
        ),
        "records_with_audio_evidence": sum(
            bool(row["audio_supporting_features"] != "[]"
                 or row["audio_opposing_features"] != "[]")
            for row in output_rows
        ),
        "records_with_audio_explanation_errors": sum(
            bool(row["audio_explanation_error"])
            for row in output_rows
        ),
        "near_threshold": sum(
            "near_decision_threshold"
            in json.loads(row["uncertainty_flags"])
            for row in output_rows
        ),
        "text_audio_disagreement": sum(
            "text_audio_score_disagreement"
            in json.loads(row["uncertainty_flags"])
            for row in output_rows
        ),
        "confidence_note": (
            "Decision strength is a heuristic based on distance "
            "from 0.5. It is not calibrated confidence."
        ),
        "target_note": (
            "Target identification is not implemented in this stage. "
            "Every target is marked unknown."
        ),
        "runtime_seconds": round(
            time.perf_counter() - start_time,
            2,
        ),
        "output": str(OUTPUT_FILE.relative_to(ROOT)),
    }

    summary_path = OUTPUT_FILE.with_name(
        "humor_explanations_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== EXPLANATION SUMMARY ===")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()