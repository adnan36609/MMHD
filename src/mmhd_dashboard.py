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
BOOTSTRAP_FILE = RESULTS_DIR / "late_fusion" / "bootstrap_comparison.json"

st.set_page_config(
    page_title="MMHD | Multimodal Humor Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
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
""", unsafe_allow_html=True)


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
        raise FileNotFoundError(
            f"No dataset manifests found under {SAMPLE_ROOT}"
        )

    seen_ids = set()

    for manifest_path in manifest_files:
        with manifest_path.open("r", encoding="utf-8-sig") as file:
            manifest = json.load(file)

        if not isinstance(manifest, list):
            raise ValueError(
                f"Unexpected manifest format: {manifest_path}"
            )

        for raw in manifest:
            sample_id = str(raw.get("sample_id") or "").strip()

            if not sample_id:
                continue

            if sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate corpus sample ID: {sample_id}"
                )

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

            image_paths = [
                str(path)
                for path in image_paths
                if path
            ]

            audio_path = audio_data.get("path")

            records.append({
                "sample_id": sample_id,
                "video_id": str(source.get("video_id") or ""),
                "source_scene": str(
                    source.get("source_scene") or ""
                ),
                "source_file": str(
                    source.get("source_file") or ""
                ),
                "start_seconds": timing.get("start_seconds"),
                "end_seconds": timing.get("end_seconds"),
                "duration_seconds": timing.get(
                    "duration_seconds"
                ),
                "candidate_type": str(
                    metadata.get("candidate_type") or "unknown"
                ),
                "text": text,
                "ocr": ocr,
                "emoji": emoji,
                "audio_path": str(audio_path) if audio_path else "",
                "image_paths": image_paths,
                "image_count": len(image_paths),
                "has_text": bool(text),
                "has_audio": bool(
                    audio_path
                    and audio_data.get("status") == "success"
                ),
                "has_image": bool(
                    image_paths
                    and image_data.get("status") == "success"
                ),
                "has_ocr": bool(ocr),
                "has_emoji": bool(emoji),
            })

    return records


def resolve_asset(value):
    if not value:
        return None

    candidate = Path(str(value))
    options = [candidate] if candidate.is_absolute() else [
        ROOT / candidate,
        ROOT / "sample_clips" / candidate,
    ]

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

corpus_records = [
    {
        **row,
        "human_label": annotation_by_id.get(
            row["sample_id"], {}
        ).get("humor_label", "Not annotated"),
    }
    for row in corpus_records
]

corpus_df = pd.DataFrame(corpus_records)

corpus_by_id = {
    row["sample_id"]: row
    for row in corpus_records
}

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
    "Selection is restricted to samples with saved out-of-fold predictions."
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
st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown(
    '<span class="modality-chip">TEXT</span>'
    '<span class="modality-chip">IMAGES</span>'
    '<span class="modality-chip">AUDIO</span>'
    '<span class="modality-chip">OCR</span>'
    '<span class="modality-chip">EMOJI</span>',
    unsafe_allow_html=True,
)

late_model = fusion_metrics["models"][
    "Equal-weight Late Fusion (Text + Audio)"
]
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
tab_corpus, tab_sample, tab_results, tab_architecture = st.tabs([
    "🌐 Full Corpus Explorer",
    "🔎 Evaluated Samples",
    "📊 Experiments & Findings",
    "🧩 Architecture Status",
])

with tab_sample:
    sample = sample_by_id.get(selected_id, {})
    modality = modality_by_id.get(selected_id, {})
    prediction = prediction_by_id[selected_id]

    actual = prediction["true_label"]
    late_probability = float(
        prediction["late_fusion_humorous_probability"]
    )
    audio_probability = float(
        prediction["audio_humorous_probability"]
    )
    text_probability = float(
        prediction["text_humorous_probability"]
    )
    feature_probability = float(
        prediction["feature_fusion_humorous_probability"]
    )

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
        st.write(
            f"Text: **{modality.get('text_status', 'not indexed')}**"
        )
        st.write(
            f"OCR: **{modality.get('ocr_status', 'not indexed')}**"
        )
        st.write(
            f"Emoji: **{modality.get('emoji_status', 'not indexed')}**"
        )

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
        comparison_rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Balanced accuracy": metrics["balanced_accuracy"],
            "Macro F1": metrics["macro_f1"],
            "Humor F1": metrics["f1_humorous"],
            "ROC-AUC": metrics["roc_auc_humorous"],
        })

    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "Balanced accuracy": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Humor F1": "{:.4f}",
            "ROC-AUC": "{:.4f}",
        }),
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
        pd.DataFrame([
            {"Outcome": label, "Samples": count}
            for label, count in transitions.items()
        ]),
        hide_index=True,
        use_container_width=True,
    )

    st.warning(
        "These results are preliminary. The dataset contains 540 annotated "
        "samples across 24 source videos; independently double-annotated "
        "samples showed weak agreement (Cohen's kappa = 0.2383). "
        "No external test set has been evaluated."
    )


with tab_architecture:
    st.subheader("Implemented vs proposed architecture")

    st.success(
        "**Implemented:** transcript TF-IDF baseline, acoustic-feature baseline, "
        "keyframe descriptors, OCR and emoji experiments, feature-level fusion, "
        "fixed-weight text–audio late fusion, context-text experiment, "
        "out-of-fold evaluation, and acoustic feature contributions."
    )

    st.warning(
        "**Partial:** modality-specific feature extraction and association by "
        "sample ID are implemented, but they do not constitute learned semantic "
        "or temporal multimodal alignment. Neighbouring-scene text is a basic "
        "context proxy rather than a persistent conversation-memory module."
    )

    st.info(
        "**Proposed future work:** learned multimodal encoders and alignment, "
        "cross-image semantic reasoning, persistent conversation memory, "
        "speaker–listener graphs, cultural retrieval-augmented generation (RAG), "
        "MLLM integration, natural-language explanations, and calibrated confidence."
    )

    st.info(
        "**Not currently implemented as evaluated tasks:** sarcasm, irony, "
        "hate-speech detection, and humor-target identification. These require "
        "suitable labels, training pipelines, and task-specific evaluation."
    )

    st.markdown("#### Current experimental conclusion")
    st.write(
        "Audio-only is a strong baseline. Equal-weight text–audio late fusion "
        "has a slightly higher macro-F1 point estimate, but its bootstrap "
        "interval includes zero. Adding more modalities does not consistently "
        "improve results with the current feature representations."
    )


st.divider()
st.markdown(
    '<div class="small-note">'
    'MMHD research prototype · Local assets are loaded from the repository. '
    'Predictions are experimental outputs, not guarantees. '
    'Do not interpret feature contributions as causal explanations.'
    '</div>',
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
        delta=(
            f"{100 * corpus_df['has_text'].mean():.1f}% coverage"
        ),
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

        coverage_rows.append({
            "Modality": modality_name,
            "Available records": count,
            "Coverage (%)": round(
                100 * count / total_records, 1
            ),
        })

    st.dataframe(
        pd.DataFrame(coverage_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("#### Search and filter records")

    filter_left, filter_middle, filter_right = st.columns(
        [1, 1, 2]
    )

    video_options = [
        "All videos",
        *sorted(
            video
            for video in corpus_df["video_id"].unique()
            if video
        ),
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
        filtered = filtered[
            filtered["video_id"] == selected_video
        ]

    if annotation_filter == "Human-labelled only":
        filtered = filtered[
            filtered["human_label"] != "Not annotated"
        ]
    elif annotation_filter == "Unlabelled only":
        filtered = filtered[
            filtered["human_label"] == "Not annotated"
        ]

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
        f"{len(filtered):,} matching records out of "
        f"{total_records:,} accepted records."
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
            "No records match these filters. "
            "Change the search or filters to continue."
        )
    else:
        st.divider()
        st.markdown("#### Inspect an individual record")

        filtered_ids = filtered["sample_id"].tolist()

        selected_corpus_id = st.selectbox(
            "Select a corpus sample",
            options=filtered_ids,
            format_func=lambda sample_id: (
                f"{sample_id} · "
                f"{corpus_by_id[sample_id]['video_id']}"
            ),
            key="corpus_selected_sample",
        )

        record = corpus_by_id[selected_corpus_id]

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
                st.success(
                    f"Recorded label: {record['human_label']}"
                )
            else:
                st.warning(
                    "This sample has no human-verified humor label."
                )

            st.markdown("##### Timing and metadata")

            st.write({
                "Start (seconds)": record["start_seconds"],
                "End (seconds)": record["end_seconds"],
                "Duration (seconds)": record["duration_seconds"],
                "Candidate type": record["candidate_type"],
            })

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
                "No saved out-of-fold prediction is shown for this "
                "sample. This explorer does not generate a new prediction."
            )