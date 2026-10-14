import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="MMHD Pilot Annotation",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent
PILOT_FILE = ROOT / "annotations" / "pilot" / "pilot_samples.json"
ANNOTATIONS_FILE = (
    ROOT / "annotations" / "pilot" / "pilot_annotations_second.json"
)


@st.cache_data
def load_samples():
    with open(PILOT_FILE, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_annotations():
    if not ANNOTATIONS_FILE.exists():
        return {}

    try:
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    return {
        record["sample_id"]: record.get("humor_label")
        for record in records
        if record.get("sample_id")
    }


def save_annotations(annotations):
    records = [
        {
            "sample_id": sample["sample_id"],
            "humor_label": annotations.get(sample["sample_id"]),
        }
        for sample in samples
    ]

    with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


samples = load_samples()
samples = samples[:240]
if "annotations" not in st.session_state:
    st.session_state.annotations = load_annotations()

if "current_index" not in st.session_state:
    annotated_indices = [
        i
        for i, sample in enumerate(samples)
        if st.session_state.annotations.get(sample["sample_id"])
        in {"humorous", "non_humorous"}
    ]

    st.session_state.current_index = (
        max(annotated_indices) + 1 if annotated_indices else 0
    )

    if st.session_state.current_index >= len(samples):
        st.session_state.current_index = len(samples) - 1


st.title("MMHD — Pilot Humor Annotation")

total = len(samples)
index = st.session_state.current_index
current = samples[index]

annotated_count = sum(
    1
    for sample in samples
    if st.session_state.annotations.get(sample["sample_id"])
    in {"humorous", "non_humorous"}
)

st.progress(annotated_count / total)

st.caption(f"Progress: {annotated_count} / {total}")

st.divider()

st.subheader(current["sample_id"])

metadata_col1, metadata_col2, metadata_col3 = st.columns(3)

with metadata_col1:
    st.write(f"**Video:** `{current['video_id']}`")

with metadata_col2:
    st.write(f"**Scene:** `{current['source_scene']}`")

with metadata_col3:
    st.write(
        f"**Time:** {current['start_seconds']:.3f}s → "
        f"{current['end_seconds']:.3f}s"
    )


st.subheader("Text")

text = current.get("text", "").strip()

if text:
    st.info(text)
else:
    st.caption("No extracted text available.")


st.subheader("Images")

image_paths = current.get("image_paths", [])

image_columns = st.columns(len(image_paths)) if image_paths else []

for column, relative_path in zip(image_columns, image_paths):
    image_path = ROOT / relative_path

    with column:
        if image_path.exists():
            st.image(str(image_path))
        else:
            st.warning(f"Image not found: {relative_path}")


st.subheader("Audio")

audio_relative_path = current.get("audio_path", "")
audio_path = ROOT / audio_relative_path

if audio_path.exists():
    with open(audio_path, "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/wav")
else:
    st.warning(f"Audio not found: {audio_relative_path}")


st.divider()

st.subheader("Annotation")

st.write(
    "Does this candidate itself contain a recognizable intentional "
    "comedic element?"
)

button_col1, button_col2 = st.columns(2)

with button_col1:
    humorous = st.button(
        "Humorous",
        use_container_width=True,
        type="primary",
    )

with button_col2:
    non_humorous = st.button(
        "Non-humorous",
        use_container_width=True,
)


def annotate(label):
    sample_id = current["sample_id"]

    st.session_state.annotations[sample_id] = label
    save_annotations(st.session_state.annotations)

    if index < total - 1:
        st.session_state.current_index += 1

    st.rerun()


if humorous:
    annotate("humorous")

if non_humorous:
    annotate("non_humorous")


st.caption(
    "Use only the candidate's available text, audio, and representative "
    "images. Do not infer the label from the source video, scene name, "
    "candidate type, sample ID, or position."
)