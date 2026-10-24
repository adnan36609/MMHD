from pathlib import Path
import csv
import json
import re

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "annotations" / "expanded_540"
RESULTS_DIR = ROOT / "results" / "expanded_540"
SAMPLE_ROOT = ROOT / "sample_clips"
SAMPLES_FILE = DATA_DIR / "samples_540.json"
ANNOTATIONS_FILE = DATA_DIR / "annotations_540.json"
MODALITY_FILE = ROOT / "results" / "modality_examples.csv"
FUSION_PREDICTIONS_FILE = RESULTS_DIR / "late_fusion" / "oof_predictions.csv"
FUSION_METRICS_FILE = RESULTS_DIR / "late_fusion" / "metrics.json"
FIVE_STREAM_METRICS_FILE = RESULTS_DIR / "five_stream_comparison" / "metrics.json"
FIVE_STREAM_PREDICTIONS_FILE = (
    RESULTS_DIR / "five_stream_comparison" / "oof_predictions.csv"
)
BOOTSTRAP_FILE = RESULTS_DIR / "late_fusion" / "bootstrap_comparison.json"
STRUCTURED_OUTPUT_FILE = (
    ROOT / "results" / "full_corpus" / "humor_structured_output.csv"
)
CONTEXT_ENRICHED_FILE = ROOT / "results" / "full_corpus" / "humor_context_enriched.csv"
CROSS_MODAL_ATTENTION_PREDICTIONS_FILE = (
    ROOT / "results" / "expanded_540" / "cross_modal_reasoning" / "oof_predictions.csv"
)

CROSS_MODAL_ATTENTION_METRICS_FILE = (
    ROOT / "results" / "expanded_540" / "cross_modal_reasoning" / "metrics.json"
)

st.set_page_config(
    page_title="MMHD | Multimodal Humor Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1020 0%, #10172a 100%);
        color: #edf2ff;
    }
    [data-testid="stSidebar"] {
        background: #11182a;
        border-right: 1px solid #27334b;
    }
    h1, h2, h3 {
        color: #f3f6ff;
        letter-spacing: -0.025em;
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #192b4c, #25204a);
        border: 1px solid #35466c;
        margin-bottom: 1.2rem;
    }
    .hero-kicker {
        color: #aabdfd;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }
    .hero-subtitle {
        color: #c2cce5;
        font-size: 1rem;
        line-height: 1.6;
    }
    .modality-chip {
        display: inline-block;
        background: #202d45;
        color: #dce7ff;
        border: 1px solid #344766;
        padding: 0.35rem 0.7rem;
        margin: 0.15rem 0.2rem 0.15rem 0;
        border-radius: 999px;
        font-size: 0.82rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(28, 40, 64, 0.78);
        border: 1px solid #2d3b59;
        border-radius: 13px;
        padding: 0.9rem 1rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #b5c4e4;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    .small-note {
        color: #aebad3;
        font-size: 0.83rem;
        line-height: 1.55;
    }
    .section-card {
        background: #151f33;
        border: 1px solid #2c3a56;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


@st.cache_data
def load_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@st.cache_data
def load_full_corpus():
    """Load metadata for all accepted samples across source videos."""
    records = []
    manifest_files = sorted(SAMPLE_ROOT.glob("*/dataset.json"))

    if not manifest_files:
        raise FileNotFoundError(f"No dataset manifests found under {SAMPLE_ROOT}")

    seen_ids = set()

    for manifest_path in manifest_files:
        with manifest_path.open("r", encoding="utf-8-sig") as file:
            manifest = json.load(file)

        if not isinstance(manifest, list):
            raise ValueError(f"Unexpected manifest format: {manifest_path}")

        for raw in manifest:
            sample_id = str(raw.get("sample_id") or "").strip()

            if not sample_id:
                continue

            if sample_id in seen_ids:
                raise ValueError(f"Duplicate corpus sample ID: {sample_id}")

            seen_ids.add(sample_id)

            source = raw.get("source") or {}
            timing = raw.get("timing") or {}
            metadata = raw.get("metadata") or {}
            modalities = raw.get("modalities") or {}

            text_data = modalities.get("text") or {}
            image_data = modalities.get("image") or {}
            audio_data = modalities.get("audio") or {}
            ocr_data = modalities.get("ocr") or {}
            emoji_data = modalities.get("emoji") or {}

            text = str(text_data.get("value") or "").strip()
            ocr = str(ocr_data.get("value") or "").strip()
            emoji = str(emoji_data.get("value") or "").strip()

            image_paths = image_data.get("paths") or []

            if isinstance(image_paths, str):
                image_paths = [image_paths]

            image_paths = [str(path) for path in image_paths if path]

            audio_path = audio_data.get("path")

            records.append(
                {
                    "sample_id": sample_id,
                    "video_id": str(source.get("video_id") or ""),
                    "source_scene": str(source.get("source_scene") or ""),
                    "source_file": str(source.get("source_file") or ""),
                    "start_seconds": timing.get("start_seconds"),
                    "end_seconds": timing.get("end_seconds"),
                    "duration_seconds": timing.get("duration_seconds"),
                    "candidate_type": str(metadata.get("candidate_type") or "unknown"),
                    "text": text,
                    "ocr": ocr,
                    "emoji": emoji,
                    "audio_path": str(audio_path) if audio_path else "",
                    "image_paths": image_paths,
                    "image_count": len(image_paths),
                    "has_text": bool(text),
                    "has_audio": bool(
                        audio_path and audio_data.get("status") == "success"
                    ),
                    "has_image": bool(
                        image_paths and image_data.get("status") == "success"
                    ),
                    "has_ocr": bool(ocr),
                    "has_emoji": bool(emoji),
                }
            )

    return records


def resolve_asset(value):
    if not value:
        return None

    candidate = Path(str(value))
    options = (
        [candidate]
        if candidate.is_absolute()
        else [
            ROOT / candidate,
            ROOT / "sample_clips" / candidate,
        ]
    )

    for path in options:
        if path.is_file():
            return path
    return None


try:
    samples = load_json(SAMPLES_FILE)
    corpus_records = load_full_corpus()
    annotations = load_json(ANNOTATIONS_FILE)
    modalities = load_csv(MODALITY_FILE)
    predictions = load_csv(FUSION_PREDICTIONS_FILE)
    fusion_metrics = load_json(FUSION_METRICS_FILE)
    bootstrap = load_json(BOOTSTRAP_FILE)

    five_stream_metrics = load_json(FIVE_STREAM_METRICS_FILE)
    five_stream_predictions = load_csv(FIVE_STREAM_PREDICTIONS_FILE)
    structured_outputs = load_csv(STRUCTURED_OUTPUT_FILE)
    context_enriched = load_csv(CONTEXT_ENRICHED_FILE)
    cross_modal_predictions = load_csv(CROSS_MODAL_ATTENTION_PREDICTIONS_FILE)
    cross_modal_metrics = load_json(CROSS_MODAL_ATTENTION_METRICS_FILE)
except Exception as exc:
    st.error(f"Could not load MMHD data: {exc}")
    st.info(
        "Run the dashboard from the project repository. "
        "Check that the expanded dataset and experiment output files exist."
    )
    st.stop()

sample_by_id = {row["sample_id"]: row for row in samples}
annotation_by_id = {row["sample_id"]: row for row in annotations}
modality_by_id = {row["sample_id"]: row for row in modalities}
prediction_by_id = {row["sample_id"]: row for row in predictions}
structured_by_id = {row["sample_id"]: row for row in structured_outputs}
context_by_id = {row["sample_id"]: row for row in context_enriched}
cross_modal_by_id = {row["sample_id"]: row for row in cross_modal_predictions}


def parse_optional_probability(value):
    """Return a valid probability in [0, 1], or None if unavailable."""
    try:
        score = float(value)
    except (ValueError, TypeError):
        return None

    return score if 0.0 <= score <= 1.0 else None


PREDICTION_METHOD_LABELS = {
    "text_audio_late_fusion": "Text-audio late fusion",
    "audio_only_fallback": "Audio-only fallback",
    "text_only_fallback": "Text-only fallback",
    "no_available_model": "Unavailable (no model input)",
}

enriched_corpus_records = []

for row in corpus_records:
    sample_id = row["sample_id"]

    annotation = annotation_by_id.get(sample_id, {})
    full_prediction = structured_by_id.get(sample_id, {})

    raw_method = str(full_prediction.get("prediction_method") or "").strip()

    raw_label = str(full_prediction.get("humor_prediction") or "").strip().lower()

    humor_probability = parse_optional_probability(
        full_prediction.get("humor_probability")
    )

    text_probability = parse_optional_probability(
        full_prediction.get("text_humorous_probability")
    )

    audio_probability = parse_optional_probability(
        full_prediction.get("audio_humorous_probability")
    )

    has_prediction = (
        raw_label in {"humorous", "non_humorous"}
        and humor_probability is not None
        and raw_method != "no_available_model"
    )

    prediction_status = "Scored" if has_prediction else "Unavailable"

    predicted_label = raw_label if has_prediction else "Unavailable"

    method_label = PREDICTION_METHOD_LABELS.get(
        raw_method,
        raw_method or "No prediction record",
    )

    enriched_corpus_records.append(
        {
            **row,
            # Ground truth: only use the existing annotation.
            "human_label": annotation.get(
                "humor_label",
                "Not annotated",
            ),
            # Full-corpus inference: separate from ground truth.
            "prediction_status": prediction_status,
            "predicted_label": predicted_label,
            "prediction_method_full": method_label,
            "humor_probability_full": humor_probability,
            "text_probability_full": text_probability,
            "audio_probability_full": audio_probability,
        }
    )

corpus_records = enriched_corpus_records

corpus_df = pd.DataFrame(corpus_records)

corpus_by_id = {row["sample_id"]: row for row in corpus_records}

if not predictions:
    st.error("No out-of-fold late-fusion predictions are available.")
    st.stop()

default_id = "f6Ukgpag63w-Scene-161"
sample_ids = sorted(prediction_by_id.keys())

if default_id not in sample_ids:
    default_id = sample_ids[0]


# Sidebar
st.sidebar.markdown("## 🎭 MMHD")
st.sidebar.caption("Multimodal Humor Detection")
st.sidebar.divider()

selected_id = st.sidebar.selectbox(
    "Select an evaluated sample",
    options=sample_ids,
    index=sample_ids.index(default_id),
    format_func=lambda sid: (
        f"{sid}  ·  {sample_by_id.get(sid, {}).get('video_id', 'unknown')}"
    ),
)

st.sidebar.caption(
    "Evaluated Samples is restricted to records with saved out-of-fold "
    "predictions. Full Corpus Explorer includes all accepted records."
)
st.sidebar.divider()
st.sidebar.markdown("**Research setup**")
st.sidebar.markdown(
    "- Binary humor classification\n"
    "- Five-fold grouped cross-validation\n"
    "- Groups defined by source video\n"
    "- Fixed 50:50 text–audio late fusion"
)
st.sidebar.caption("Local research prototype · Not a production detector")


# Header
st.markdown(
    """
<div class="hero">
  <div class="hero-kicker">Research Prototype · MMHD</div>
  <h1>Multimodal Humor Detection</h1>
  <div class="hero-subtitle">
    Explore how text, audio, visual frames, OCR and emoji metadata
    contribute to humor-detection experiments. Inspect out-of-fold
    predictions, compare baselines, and examine the limitations of
    the current feature-fusion approach.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="modality-chip">TEXT</span>'
    '<span class="modality-chip">IMAGES</span>'
    '<span class="modality-chip">AUDIO</span>'
    '<span class="modality-chip">OCR</span>'
    '<span class="modality-chip">EMOJI</span>',
    unsafe_allow_html=True,
)

late_model = fusion_metrics["models"]["Equal-weight Late Fusion (Text + Audio)"]
audio_model = fusion_metrics["models"]["Audio-only"]
text_model = fusion_metrics["models"]["Text-only"]

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Full corpus records",
    f"{len(corpus_records):,}",
)

k2.metric(
    "Human-annotated samples",
    f"{len(annotations):,}",
)

k3.metric(
    "Evaluated samples",
    f"{len(predictions):,}",
)

k4.metric(
    "Source videos",
    f"{corpus_df['video_id'].nunique():,}",
)

k5.metric(
    "Late-fusion macro F1",
    f"{late_model['macro_f1']:.4f}",
)

st.caption(
    "Headline metrics use saved out-of-fold predictions. "
    "The selected sample belongs to an evaluation fold whose model "
    "was trained without that sample's source video."
)

st.divider()


# Tabs
tab_corpus, tab_sample, tab_results, tab_architecture = st.tabs(
    [
        "🌐 Full Corpus Explorer",
        "🔎 Evaluated Samples",
        "📊 Experiments & Findings",
        "🧩 Architecture Status",
    ]
)

with tab_sample:
    sample = sample_by_id.get(selected_id, {})
    modality = modality_by_id.get(selected_id, {})
    prediction = prediction_by_id[selected_id]

    actual = prediction["true_label"]
    late_probability = float(prediction["late_fusion_humorous_probability"])
    audio_probability = float(prediction["audio_humorous_probability"])
    text_probability = float(prediction["text_humorous_probability"])
    feature_probability = float(prediction["feature_fusion_humorous_probability"])

    st.subheader("Selected sample")
    st.caption(
        f"Sample ID: `{selected_id}`  ·  "
        f"Source video: `{sample.get('video_id', 'unknown')}`"
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("#### Transcript")
        transcript = str(sample.get("text") or "").strip()
        if transcript:
            st.info(transcript)
        else:
            st.warning("No transcript available.")

        st.markdown("#### OCR text")
        ocr_text = str(modality.get("ocr") or "").strip()
        if ocr_text:
            st.write(ocr_text)
        else:
            st.caption("No OCR text recorded for this sample.")

        st.markdown("#### Emoji metadata")
        emoji = str(modality.get("emoji") or "").strip()
        if emoji:
            st.markdown(f"### {emoji}")
        else:
            st.caption("No emoji recorded for this sample.")

        st.markdown("#### Audio")
        audio_path = resolve_asset(sample.get("audio_path"))
        if audio_path:
            st.audio(str(audio_path))
        else:
            st.warning("The sample's audio file is not available locally.")

    with right:
        st.markdown("#### Visual keyframes")
        image_paths = sample.get("image_paths") or []
        displayed_images = 0

        if image_paths:
            image_columns = st.columns(2)
            for i, image_value in enumerate(image_paths):
                image_path = resolve_asset(image_value)
                if image_path:
                    with image_columns[i % 2]:
                        st.image(
                            str(image_path),
                            caption=f"Keyframe {i + 1}",
                            use_container_width=True,
                        )
                    displayed_images += 1

        if displayed_images == 0:
            st.caption("No keyframe images could be loaded.")

        st.markdown("#### Modality availability")
        st.write(f"Text: **{modality.get('text_status', 'not indexed')}**")
        st.write(f"OCR: **{modality.get('ocr_status', 'not indexed')}**")
        st.write(f"Emoji: **{modality.get('emoji_status', 'not indexed')}**")

    st.divider()
    st.subheader("Out-of-fold predictions")

    st.caption(
        "These probabilities come from the saved evaluation run, not "
        "from retraining or testing a model on this selected sample. "
        "Probabilities have not been calibrated."
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(
        "Text only",
        f"{text_probability:.1%}",
        help="Estimated probability of the humorous class.",
    )
    p2.metric(
        "Audio only",
        f"{audio_probability:.1%}",
        help="Estimated probability of the humorous class.",
    )
    p3.metric(
        "Feature-level fusion",
        f"{feature_probability:.1%}",
        help="Text and audio combined at the feature level.",
    )
    p4.metric(
        "Late fusion",
        f"{late_probability:.1%}",
        help="Fixed equal-weight average of text and audio probabilities.",
    )

    predicted_label = prediction["late_fusion_prediction"]
    st.markdown(
        f"**Late-fusion prediction:** `{predicted_label}`  ·  "
        f"**Recorded annotation:** `{actual}`"
    )

    if predicted_label == actual:
        st.success("The late-fusion prediction matches the recorded annotation.")
    else:
        st.warning("The late-fusion prediction differs from the recorded annotation.")

    st.info(
        "This is a binary humor experiment. It does not classify sarcasm, "
        "irony, hate speech, or the target of humor."
    )


with tab_results:
    st.subheader("Model comparison")
    st.write(
        "All four approaches are evaluated on the same 391 samples using "
        "the existing source-video-grouped out-of-fold predictions."
    )

    model_names = [
        "Text-only",
        "Audio-only",
        "Feature-level Text + Audio",
        "Equal-weight Late Fusion (Text + Audio)",
    ]

    comparison_rows = []
    for name in model_names:
        metrics = fusion_metrics["models"][name]
        comparison_rows.append(
            {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Balanced accuracy": metrics["balanced_accuracy"],
                "Macro F1": metrics["macro_f1"],
                "Humor F1": metrics["f1_humorous"],
                "ROC-AUC": metrics["roc_auc_humorous"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "Balanced accuracy": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Humor F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_data = comparison_df.set_index("Model")[["Macro F1"]]
    st.markdown("#### Macro-F1 comparison")
    st.bar_chart(chart_data)

    st.divider()
    st.subheader("Late fusion vs audio-only")

    difference = bootstrap["comparisons"]["macro_f1"]
    observed_delta = difference["observed_late_minus_audio"]
    ci_low, ci_high = difference["cluster_bootstrap_95_ci"]

    d1, d2, d3 = st.columns(3)
    d1.metric("Observed macro-F1 difference", f"{observed_delta:+.4f}")
    d2.metric("95% lower bound", f"{ci_low:+.4f}")
    d3.metric("95% upper bound", f"{ci_high:+.4f}")

    st.caption(
        f"The cluster-bootstrap interval is [{ci_low:+.4f}, {ci_high:+.4f}]. "
        "Because it includes zero, the observed improvement does not establish "
        "that late fusion reliably outperforms audio-only."
    )

    # Compare which errors are corrected or introduced by late fusion.
    transitions = {
        "Both correct": 0,
        "Late fusion only correct": 0,
        "Audio-only correct": 0,
        "Both wrong": 0,
    }

    for row in predictions:
        truth = row["true_label"]
        audio_pred = (
            "humorous"
            if float(row["audio_humorous_probability"]) >= 0.5
            else "non_humorous"
        )
        late_pred = row["late_fusion_prediction"]
        a_correct = audio_pred == truth
        l_correct = late_pred == truth

        if a_correct and l_correct:
            transitions["Both correct"] += 1
        elif l_correct:
            transitions["Late fusion only correct"] += 1
        elif a_correct:
            transitions["Audio-only correct"] += 1
        else:
            transitions["Both wrong"] += 1

    st.markdown("#### Error transitions")
    st.dataframe(
        pd.DataFrame(
            [
                {"Outcome": label, "Samples": count}
                for label, count in transitions.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.warning(
        "These results are preliminary. The dataset contains 540 annotated "
        "samples across 24 source videos; independently double-annotated "
        "samples showed weak agreement (Cohen's kappa = 0.2383). "
        "No external test set has been evaluated."
    )

    st.divider()
    st.subheader("Five-stream multimodal ablation")

    five_dataset = five_stream_metrics.get("dataset", {})
    five_sample_count = five_dataset.get(
        "usable_complete_cases",
        len(five_stream_predictions),
    )

    st.write(
        "Compare 11 configurations using text, audio, visual features, "
        "OCR, emoji, and their combinations."
    )

    st.info(
        f"This experiment evaluates {five_sample_count} complete-case "
        f"samples from the 540 annotated records. "
        f"{five_dataset.get('excluded_no_ocr', 0)} records were excluded "
        "because OCR data was unavailable. These results use a different "
        "subset from the primary 391-sample late-fusion evaluation."
    )

    ablation_rows = []

    for name, details in five_stream_metrics["models"].items():
        metrics = details.get("out_of_fold_metrics", {})

        ablation_rows.append(
            {
                "Configuration": name,
                "Macro F1": metrics.get("macro_f1"),
                "Accuracy": metrics.get("accuracy"),
                "Balanced accuracy": metrics.get("balanced_accuracy"),
                "ROC-AUC": metrics.get("roc_auc_humorous"),
            }
        )

    ablation_df = pd.DataFrame(ablation_rows)

    st.dataframe(
        ablation_df.style.format(
            {
                "Macro F1": "{:.4f}",
                "Accuracy": "{:.4f}",
                "Balanced accuracy": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Macro-F1 comparison across configurations")

    st.bar_chart(ablation_df.set_index("Configuration")[["Macro F1"]])

    st.caption(
        "All configurations use the same 343 complete-case samples and "
        "five-fold source-video-grouped evaluation. Results are preliminary "
        "and should not be compared directly with metrics from a different "
        "evaluation subset."
    )

    st.divider()
    st.markdown("#### Inspect a sample across all configurations")

    five_stream_df = pd.DataFrame(five_stream_predictions)

    five_sample_ids = five_stream_df["sample_id"].astype(str).tolist()

    selected_five_id = st.selectbox(
        "Select an evaluated sample",
        five_sample_ids,
        key="five_stream_sample_inspector",
    )

    five_row = five_stream_df.loc[
        five_stream_df["sample_id"].astype(str) == selected_five_id
    ].iloc[0]

    st.write("**Sample ID:**", selected_five_id)
    st.write("**Source video:**", five_row["video_id"])
    st.write("**Human annotation:**", five_row["true_label"])

    if pd.notna(five_row["text"]) and str(five_row["text"]).strip():
        st.write("**Transcript:**", five_row["text"])
    else:
        st.write("**Transcript:** No text available.")

    if pd.notna(five_row["ocr"]) and str(five_row["ocr"]).strip():
        st.write("**OCR:**", five_row["ocr"])
    else:
        st.write("**OCR:** No OCR text available.")

    if pd.notna(five_row["emoji"]) and str(five_row["emoji"]).strip():
        st.write("**Emoji:**", five_row["emoji"])
    else:
        st.write("**Emoji:** No emoji recorded.")

    configuration_columns = [
        ("Text-only", "text_only_prediction", "text_only_humorous_probability"),
        ("Audio-only", "audio_only_prediction", "audio_only_humorous_probability"),
        ("Visual-only", "visual_only_prediction", "visual_only_humorous_probability"),
        ("OCR-only", "ocr_only_prediction", "ocr_only_humorous_probability"),
        ("Text + OCR", "text_ocr_prediction", "text_ocr_humorous_probability"),
        ("Text + Audio", "text_audio_prediction", "text_audio_humorous_probability"),
        ("Audio + OCR", "audio_ocr_prediction", "audio_ocr_humorous_probability"),
        (
            "Text + Audio + Visual",
            "text_audio_visual_prediction",
            "text_audio_visual_humorous_probability",
        ),
        (
            "Text + Audio + OCR",
            "text_audio_ocr_prediction",
            "text_audio_ocr_humorous_probability",
        ),
        (
            "Text + Audio + Visual + OCR",
            "text_audio_visual_ocr_prediction",
            "text_audio_visual_ocr_humorous_probability",
        ),
        (
            "All five streams",
            "all_five_streams_prediction",
            "all_five_streams_humorous_probability",
        ),
    ]

    sample_comparison_rows = []

    for name, prediction_col, probability_col in configuration_columns:
        probability = pd.to_numeric(
            five_row[probability_col],
            errors="coerce",
        )

        sample_comparison_rows.append(
            {
                "Configuration": name,
                "Predicted label": five_row[prediction_col],
                "Humorous probability": probability,
            }
        )

    sample_comparison_df = pd.DataFrame(sample_comparison_rows)

    st.dataframe(
        sample_comparison_df.style.format(
            {
                "Humorous probability": "{:.3f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_architecture:
    st.subheader("Implemented vs proposed architecture")
    st.caption(
        "Implementation status reflects the current repository and saved "
        "experiments. Experimental components are not automatically part "
        "of the full-corpus prediction pipeline."
    )

    st.markdown("#### 1. Implemented and experimentally evaluated")

    st.success(
        "**Dataset and modality processing**\n\n"
        "- 7,289 accepted scene records across 24 source videos.\n"
        "- Transcript and audio feature pipelines.\n"
        "- Visual keyframe descriptors, OCR text, and emoji metadata.\n"
        "- Human annotations for 540 samples."
    )

    st.success(
        "**Model experiments**\n\n"
        "- Text-only and audio-only humor baselines.\n"
        "- Visual-only and OCR-only experiments, plus emoji metadata "
        "in multimodal comparisons.\n"
        "- Feature-level fusion and multimodal ablation experiments.\n"
        "- Fixed 50:50 text–audio late fusion.\n"
        "- Context-text experiment and acoustic feature analysis."
    )

    st.success(
        "**Evaluation and inspection**\n\n"
        "- Source-video-grouped out-of-fold evaluation to reduce "
        "train/test leakage between scenes from the same video.\n"
        "- A primary comparison on 391 complete-case samples.\n"
        "- An 11-configuration five-stream ablation on 343 complete-case "
        "samples.\n"
        "- Saved full-corpus predictions, structured outputs, and "
        "evidence-based heuristic explanations."
    )

    st.divider()
    st.markdown("#### 2. Full-corpus prediction pipeline")

    st.info(
        "**Current inference path:** transcript model + audio model "
        "→ fixed-weight late fusion when both predictions are available "
        "→ audio-only fallback when text predictions are unavailable."
    )

    pipeline_left, pipeline_right = st.columns(2)

    pipeline_left.metric("Text–audio fusion", "5,674 records")
    pipeline_left.metric("Audio-only fallback", "616 records")

    pipeline_right.metric("No available model", "999 records")
    pipeline_right.metric("Total corpus", "7,289 records")

    st.warning(
        "The full-corpus prediction branch does not use all five streams "
        "for every prediction. Visual, OCR, and emoji features have "
        "separate experiments; their availability or experimental "
        "evaluation does not imply integrated five-stream inference."
    )

    st.divider()
    st.markdown("#### 3. Important limitations")

    st.warning(
        "**Small annotated evaluation set:** the primary evaluation uses "
        "391 complete cases from 540 annotated records. The five-stream "
        "experiment uses a different 343-sample subset because OCR data "
        "is unavailable for 197 records. Their metrics should not be "
        "compared as if they came from the same evaluation set."
    )

    st.warning(
        "**Fusion improvement remains uncertain:** primary late fusion "
        "has a macro-F1 of 0.6306. Its observed improvement over audio-only "
        "is +0.0058, with a cluster-bootstrap 95% interval from -0.0270 "
        "to +0.0390. The interval includes zero, so a reliable improvement "
        "has not been established."
    )

    st.warning(
        "**Explanations and target candidates are heuristic:** structured "
        "evidence and target-candidate rules do not establish causal "
        "explanations or validated humor-target identification. Confidence "
        "is not calibrated, and these outputs should not be presented as "
        "ground-truth reasoning."
    )

    st.warning(
        "**Research validity:** independently double-annotated samples "
        "showed weak agreement (Cohen's kappa = 0.2383). Evaluation is "
        "grouped by source video, but no external test set has been "
        "evaluated."
    )

    st.divider()
    st.markdown("#### 4. Proposed future architecture")

    st.info(
        "**Not yet implemented as an integrated, validated system:**\n\n"
        "- Learned semantic alignment across text, audio, images, and OCR.\n"
        "- Cross-image and temporal scene reasoning.\n"
        "- Persistent conversation memory and speaker–listener graphs.\n"
        "- Cultural retrieval-augmented generation (RAG) and MLLM integration.\n"
        "- Validated humor-target identification and task-specific "
        "sarcasm or irony detection.\n"
        "- Calibrated confidence and external evaluation."
    )

    st.markdown("#### Current experimental conclusion")

    st.write(
        "Audio is a strong baseline in the current experiments. Equal-weight "
        "text–audio late fusion has a slightly higher macro-F1 point estimate "
        "on the primary evaluation, but the uncertainty interval includes "
        "zero. The five-stream ablation does not demonstrate a consistent "
        "macro-F1 improvement from adding more modalities with the current "
        "feature representations. These findings motivate further research; "
        "they do not establish that more complex fusion is inherently better."
    )


st.divider()
st.markdown(
    '<div class="small-note">'
    "MMHD research prototype · Local assets are loaded from the repository. "
    "Predictions are experimental outputs, not guarantees. "
    "Do not interpret feature contributions as causal explanations."
    "</div>",
    unsafe_allow_html=True,
)

with tab_corpus:
    st.subheader("Full Multimodal Corpus")
    st.caption(
        "Explore accepted scene records from all source videos, "
        "including samples without human humor labels or saved "
        "out-of-fold predictions."
    )

    total_records = len(corpus_df)

    if total_records == 0:
        st.warning("The accepted corpus contains no records.")
        st.stop()

    # Corpus-wide statistics
    m1, m2, m3 = st.columns(3)

    m1.metric("Accepted records", f"{total_records:,}")

    m2.metric(
        "Source videos",
        f"{corpus_df['video_id'].nunique():,}",
    )

    m3.metric(
        "Records with transcripts",
        f"{int(corpus_df['has_text'].sum()):,}",
        delta=(f"{100 * corpus_df['has_text'].mean():.1f}% coverage"),
    )

    m4, m5, m6 = st.columns(3)

    m4.metric(
        "Records with audio",
        f"{int(corpus_df['has_audio'].sum()):,}",
    )

    m5.metric(
        "Records with OCR text",
        f"{int(corpus_df['has_ocr'].sum()):,}",
    )

    m6.metric(
        "Records with emoji",
        f"{int(corpus_df['has_emoji'].sum()):,}",
    )

    st.metric(
        "Total extracted image keyframes",
        f"{int(corpus_df['image_count'].sum()):,}",
    )

    st.divider()
    st.markdown("#### Modality coverage")

    modality_definitions = [
        ("Transcript", "has_text"),
        ("Audio", "has_audio"),
        ("At least one image", "has_image"),
        ("OCR text", "has_ocr"),
        ("Emoji", "has_emoji"),
    ]

    coverage_rows = []

    for modality_name, column_name in modality_definitions:
        count = int(corpus_df[column_name].sum())

        coverage_rows.append(
            {
                "Modality": modality_name,
                "Available records": count,
                "Coverage (%)": round(100 * count / total_records, 1),
            }
        )

    st.dataframe(
        pd.DataFrame(coverage_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("#### Search and filter records")

    filter_left, filter_middle, filter_right = st.columns([1, 1, 2])

    video_options = [
        "All videos",
        *sorted(video for video in corpus_df["video_id"].unique() if video),
    ]

    with filter_left:
        selected_video = st.selectbox(
            "Source video",
            video_options,
            key="corpus_video_filter",
        )

    with filter_middle:
        annotation_filter = st.selectbox(
            "Annotation status",
            [
                "All records",
                "Human-labelled only",
                "Unlabelled only",
            ],
            key="corpus_annotation_filter",
        )

    with filter_right:
        search_query = st.text_input(
            "Search transcript or OCR",
            placeholder="Enter a word or phrase",
            key="corpus_search",
        )

    required_modalities = st.multiselect(
        "Require available modalities",
        ["Transcript", "Audio", "Image", "OCR", "Emoji"],
        key="corpus_required_modalities",
    )

    filtered = corpus_df.copy()

    if selected_video != "All videos":
        filtered = filtered[filtered["video_id"] == selected_video]

    if annotation_filter == "Human-labelled only":
        filtered = filtered[filtered["human_label"] != "Not annotated"]
    elif annotation_filter == "Unlabelled only":
        filtered = filtered[filtered["human_label"] == "Not annotated"]

    if search_query.strip():
        query = re.escape(search_query.strip())

        text_matches = filtered["text"].str.contains(
            query,
            case=False,
            regex=True,
            na=False,
        )

        ocr_matches = filtered["ocr"].str.contains(
            query,
            case=False,
            regex=True,
            na=False,
        )

        filtered = filtered[text_matches | ocr_matches]

    modality_columns = {
        "Transcript": "has_text",
        "Audio": "has_audio",
        "Image": "has_image",
        "OCR": "has_ocr",
        "Emoji": "has_emoji",
    }

    for modality_name in required_modalities:
        column_name = modality_columns[modality_name]
        filtered = filtered[filtered[column_name]]

    st.markdown("#### Matching records")

    st.caption(
        f"{len(filtered):,} matching records out of {total_records:,} accepted records."
    )

    display_columns = [
        "sample_id",
        "video_id",
        "start_seconds",
        "duration_seconds",
        "human_label",
        "candidate_type",
        "has_text",
        "has_audio",
        "has_image",
        "has_ocr",
        "has_emoji",
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    if filtered.empty:
        st.info(
            "No records match these filters. Change the search or filters to continue."
        )
    else:
        st.divider()
        st.markdown("#### Inspect an individual record")

        filtered_ids = filtered["sample_id"].tolist()

        selected_corpus_id = st.selectbox(
            "Select a corpus sample",
            options=filtered_ids,
            format_func=lambda sample_id: (
                f"{sample_id} · {corpus_by_id[sample_id]['video_id']}"
            ),
            key="corpus_selected_sample",
        )

        record = corpus_by_id[selected_corpus_id]
        context_record = context_by_id.get(selected_corpus_id, {})

        st.caption(
            f"Sample: `{record['sample_id']}` · "
            f"Video: `{record['video_id']}` · "
            f"Scene: `{record['source_scene']}`"
        )

        detail_left, detail_right = st.columns(
            [1.05, 0.95],
            gap="large",
        )

        with detail_left:
            st.markdown("##### Transcript")

            if record["text"]:
                st.info(record["text"])
            else:
                st.caption("No transcript available.")

            st.divider()
            st.markdown("##### Neighboring-scene context")

            context_status = context_record.get(
                "context_availability",
                "unavailable",
            )
            st.caption(f"Context availability: {context_status}")

            previous_text = str(context_record.get("previous_scene_text", "")).strip()
            previous_scene_id = context_record.get("previous_scene_id", "")

            next_text = str(context_record.get("next_scene_text", "")).strip()
            next_scene_id = context_record.get("next_scene_id", "")

            with st.expander(
                "Previous scene",
                expanded=bool(previous_text),
            ):
                if previous_text:
                    st.caption(f"Scene ID: {previous_scene_id}")
                    st.write(previous_text)
                else:
                    st.caption("No neighboring transcript is available.")

            with st.expander(
                "Next scene",
                expanded=bool(next_text),
            ):
                if next_text:
                    st.caption(f"Scene ID: {next_scene_id}")
                    st.write(next_text)
                else:
                    st.caption("No neighboring transcript is available.")

            st.caption(
                "Context comes from the nearest available scenes in the "
                "same video's scene order. It is not verified conversational "
                "dialogue and does not alter the humor prediction."
            )

            st.markdown("##### On-screen OCR")

            if record["ocr"]:
                st.write(record["ocr"])
            else:
                st.caption("No OCR text recorded.")

            st.markdown("##### Emoji metadata")

            if record["emoji"]:
                st.markdown(f"### {record['emoji']}")
            else:
                st.caption("No emoji recorded.")

            st.markdown("##### Human annotation")

            if record["human_label"] != "Not annotated":
                st.success(f"Recorded label: {record['human_label']}")
            else:
                st.warning("This sample has no human-verified humor label.")

            st.markdown("##### Timing and metadata")

            st.write(
                {
                    "Start (seconds)": record["start_seconds"],
                    "End (seconds)": record["end_seconds"],
                    "Duration (seconds)": record["duration_seconds"],
                    "Candidate type": record["candidate_type"],
                }
            )

        with detail_right:
            st.markdown("##### Audio")

            audio_file = resolve_asset(record["audio_path"])

            if audio_file:
                st.audio(str(audio_file))
            else:
                st.caption("Audio is not available locally.")

            st.markdown("##### Visual keyframes")

            image_count = 0

            for index, image_reference in enumerate(
                record["image_paths"][:4],
                start=1,
            ):
                image_file = resolve_asset(image_reference)

                if image_file:
                    st.image(
                        str(image_file),
                        caption=f"Keyframe {index}",
                        use_container_width=True,
                    )
                    image_count += 1

            if image_count == 0:
                st.caption("No local keyframes are available.")

        if selected_corpus_id in prediction_by_id:
            st.info(
                "This sample also has saved evaluation predictions. "
                "Open the Evaluated Samples tab to inspect them."
            )
        else:
            st.caption(
                "No saved out-of-fold evaluation prediction is available for "
                "this scene. The structured analysis below shows the separate "
                "full-corpus inference result, not an evaluation score or "
                "human-verified conclusion."
            )
    st.divider()
    st.markdown("#### Structured humor analysis")

    structured_record = structured_by_id.get(selected_corpus_id)

    def parse_structured_value(value):
        import json

        if value is None:
            return []

        if isinstance(value, (list, dict)):
            return value

        text = str(value).strip()

        if not text or text.lower() in {"nan", "none", "null"}:
            return []

        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return [text]

    def display_evidence(value, label):
        items = parse_structured_value(value)

        st.markdown(f"**{label}**")

        if not items:
            st.caption("No evidence recorded.")
            return

        if isinstance(items, dict):
            items = [items]

        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    feature = item.get(
                        "feature",
                        item.get("term", item.get("name", "Unknown feature")),
                    )
                    contribution = item.get("contribution")

                    if contribution is not None:
                        try:
                            st.write(
                                f"- `{feature}` — contribution: "
                                f"`{float(contribution):+.4f}`"
                            )
                        except (ValueError, TypeError):
                            st.write(f"- `{feature}`: {contribution}")
                    else:
                        st.write(f"- `{feature}`")
                else:
                    st.write(f"- {item}")
        else:
            st.write(items)

    def display_value_list(value, label):
        items = parse_structured_value(value)

        st.markdown(f"**{label}**")

        if not items:
            st.caption("None recorded.")
        elif isinstance(items, list):
            for item in items:
                st.write(f"- {item}")
        elif isinstance(items, dict):
            for key, item in items.items():
                st.write(f"- **{key}:** {item}")
        else:
            st.write(items)

    def display_probability(value, label):
        try:
            probability = float(value)
            if not 0.0 <= probability <= 1.0:
                raise ValueError("Probability outside [0, 1].")

            st.metric(label, f"{probability:.1%}")
            st.progress(probability)
        except (ValueError, TypeError):
            st.metric(label, "N/A")

    if structured_record is None:
        st.caption("No structured analysis is available for this sample.")
    else:
        interpretation_status = str(structured_record.get("interpretation_status", ""))

        if interpretation_status == (
            "annotated_sample_direct_inference_not_evaluation"
        ):
            st.warning(
                "This is the full-corpus direct-inference result for an "
                "annotated scene, not its out-of-fold evaluation prediction. "
                "The model may have been fit using this sample. Use the "
                "Evaluated Samples tab and saved out-of-fold predictions "
                "for performance evaluation."
            )
        else:
            st.caption(
                "Experimental full-corpus inference. Predictions and "
                "explanations are not human-verified conclusions."
            )

        st.markdown("##### Humor predictions")

        prediction_left, prediction_right = st.columns(2)

        with prediction_left:
            display_probability(
                structured_record.get("text_humorous_probability"),
                "Text humor score",
            )

        with prediction_right:
            display_probability(
                structured_record.get("audio_humorous_probability"),
                "Audio humor score",
            )

        display_probability(
            structured_record.get("humor_probability"),
            "Overall humor score",
        )

        st.write(
            "**Predicted label:**",
            structured_record.get("humor_prediction", "N/A"),
        )

        st.write(
            "**Prediction method:**",
            structured_record.get("prediction_method", "N/A"),
        )

        st.write(
            "**Interpretation status:**",
            structured_record.get("interpretation_status", "N/A"),
        )

        st.divider()
        st.markdown("##### Cross-modal attention experiment")

        experiment_results = cross_modal_metrics.get("metrics", {})
        baseline_metrics = experiment_results.get("text_audio_late_fusion", {})
        attention_metrics = experiment_results.get(
            "text_visual_cross_modal_attention", {}
        )

        if baseline_metrics and attention_metrics:
            metric_left, metric_right = st.columns(2)

            baseline_macro_f1 = float(baseline_metrics["macro_f1"])
            attention_macro_f1 = float(attention_metrics["macro_f1"])

            metric_left.metric(
                "Text–audio macro F1",
                f"{baseline_macro_f1:.4f}",
            )

            metric_right.metric(
                "Text–visual attention macro F1",
                f"{attention_macro_f1:.4f}",
                delta=(f"{attention_macro_f1 - baseline_macro_f1:+.4f}"),
                delta_description="versus text–audio baseline",
            )

            baseline_roc_auc = float(baseline_metrics["roc_auc_humorous"])
            attention_roc_auc = float(attention_metrics["roc_auc_humorous"])

            st.caption(
                f"ROC-AUC — text–audio: {baseline_roc_auc:.4f}; "
                f"text–visual attention: {attention_roc_auc:.4f}."
            )

        evaluation_record = prediction_by_id.get(selected_corpus_id)
        attention_record = cross_modal_by_id.get(selected_corpus_id)

        if evaluation_record and attention_record:
            baseline_probability = float(
                evaluation_record["late_fusion_humorous_probability"]
            )
            attention_probability = float(attention_record["cross_modal_probability"])

            st.markdown("###### Selected sample: out-of-fold scores")

            score_left, score_middle, score_right = st.columns(3)

            score_left.metric(
                "Text–audio probability",
                f"{baseline_probability:.1%}",
            )

            score_middle.metric(
                "Text–visual probability",
                f"{attention_probability:.1%}",
            )

            score_right.metric(
                "Probability difference",
                f"{(attention_probability - baseline_probability) * 100:+.1f} pp",
            )

            st.write(
                "**Text–audio prediction:**",
                evaluation_record.get("late_fusion_prediction", "N/A"),
            )

            st.write(
                "**Text–visual attention prediction:**",
                attention_record.get("cross_modal_prediction", "N/A"),
            )

        else:
            st.info(
                "This scene is outside the shared 391-sample evaluation "
                "subset, or a saved cross-modal prediction is unavailable. "
                "No out-of-fold comparison is displayed for this scene."
            )

        st.caption(
            "The cross-modal attention model is experimental and does not "
            "replace the primary text–audio classifier. Its probabilities "
            "are not calibrated, and a difference in one sample's score "
            "does not establish a performance improvement. Both models' "
            "aggregate results are evaluated on the saved 391-sample set."
        )

        st.markdown("##### Explanation")

        explanation = structured_record.get("explanation")

        if explanation and str(explanation).strip():
            st.write(explanation)
        else:
            st.caption("No explanation available.")

        st.markdown("##### Supporting and opposing evidence")

        evidence_left, evidence_right = st.columns(2)

        with evidence_left:
            display_evidence(
                structured_record.get("text_supporting_terms"),
                "Text evidence supporting humor",
            )
            display_evidence(
                structured_record.get("text_opposing_terms"),
                "Text evidence opposing humor",
            )

        with evidence_right:
            display_evidence(
                structured_record.get("audio_supporting_features"),
                "Audio evidence supporting humor",
            )
            display_evidence(
                structured_record.get("audio_opposing_features"),
                "Audio evidence opposing humor",
            )

        st.markdown("##### Uncertainty and confidence")

        confidence_left, confidence_right = st.columns(2)

        with confidence_left:
            st.metric(
                "Heuristic confidence",
                str(structured_record.get("confidence_level_heuristic", "N/A")).upper(),
            )

        with confidence_right:
            calibrated = (
                str(structured_record.get("confidence_calibrated", "")).strip().lower()
            )

            calibration_label = {
                "true": "Calibrated",
                "false": "Not calibrated",
                "1": "Calibrated",
                "0": "Not calibrated",
            }.get(calibrated, "Unknown")

            st.metric("Calibration status", calibration_label)

        try:
            margin = float(structured_record.get("decision_boundary_margin"))
            st.metric("Decision boundary margin", f"{margin:.4f}")
        except (ValueError, TypeError):
            st.caption("Decision boundary margin unavailable.")

        display_value_list(
            structured_record.get("uncertainty_flags"),
            "Uncertainty flags",
        )

        st.caption(
            "Heuristic confidence is not a calibrated probability. "
            "Decision margins and feature contributions are model "
            "diagnostics, not guarantees or causal explanations."
        )

        st.markdown("##### Humor target analysis")

        st.write(
            "**Proposed target:**",
            structured_record.get("target_of_humor", "Unknown"),
        )

        st.write(
            "**Target status:**",
            structured_record.get("target_status", "N/A"),
        )

        display_value_list(
            structured_record.get("target_candidates"),
            "Unverified target candidates",
        )

        display_value_list(
            structured_record.get("target_evidence"),
            "Target evidence",
        )

        st.caption(
            "Target candidates are unverified model-generated phrases. "
            "They must not be interpreted as confirmed humor targets."
        )
