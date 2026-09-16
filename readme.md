# MMHD Phase 1

Dataset-generation pipeline for **Multimodal Humor & Hate Speech Detection (MMHD)**.

Phase 1 converts source videos into structured multimodal samples containing:

**Text | Image | Audio | OCR | Emoji**

The pipeline is designed to be **reproducible, resumable, and scalable**.

> Phase 1 generates the dataset. It does not perform final humor, sarcasm, irony, or hate-speech classification.

---

## Pipeline

```text
YouTube URL
    ↓
yt-dlp
    ↓
Source Video
    ↓
PySceneDetect
    ↓
Scene Segmentation
    ↓
Candidate Formation
    ↓
Multimodal Extraction
    ↓
Text + Image + Audio + OCR + Emoji
    ↓
dataset.json
```

---

## Repository Structure

```text
MMHD/
├── source_videos/          # Downloaded source videos (ignored by Git)
├── sample_clips/           # Generated scenes, candidates and extracted data (ignored by Git)
├── sample_data/            # Local/generated sample data (ignored where applicable)
├── src/
│   ├── download.py         # YouTube video downloading
│   ├── segment.py          # Scene detection and scene splitting
│   ├── candidate.py        # Candidate generation
│   ├── extract.py          # Multimodal extraction
│   ├── pipeline.py         # Main end-to-end pipeline
│   └── analyze_scenes.py   # Scene analysis utility
├── requirements.txt
├── .gitignore
└── readme.md
```

Generated videos, clips, extracted modalities, and other large local data should not be committed to Git.

---

# Requirements

## Software

* Python 3.12
* FFmpeg
* Git
* Conda recommended

## Python Dependencies

```text
faster-whisper==1.2.1
easyocr
yt-dlp
scenedetect
```

---

# Setup

Create the Conda environment:

```powershell
conda create -n mmhd python=3.12
```

Activate it:

```powershell
conda activate mmhd
```

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Verify FFmpeg:

```powershell
ffmpeg -version
```

Verify Python:

```powershell
python --version
```

The expected Python version is:

```text
Python 3.12.x
```

---

# Running the Pipeline

The main entry point is:

```powershell
python src/pipeline.py
```

The program asks for a YouTube URL:

```text
Enter YouTube video URL:
```

Enter one URL and press Enter.

The pipeline then performs:

```text
STEP 1: Download
        ↓
STEP 2: Scene Detection
        ↓
STEP 3: Candidate Formation
        ↓
STEP 4: Multimodal Extraction
```

The current pipeline accepts **one YouTube URL per execution**.

To process another video, run the command again with the next URL.

Already downloaded videos are skipped.

---

# Pipeline Stages

## 1. Download

`src/download.py` uses **yt-dlp** to download the supplied YouTube video.

Downloaded videos are stored in:

```text
source_videos/
```

The filename is based on the YouTube video ID:

```text
source_videos/<video_id>.mp4
```

If the video has already been downloaded, the existing file is reused.

---

## 2. Scene Detection

`src/segment.py` uses **PySceneDetect ContentDetector** with the **PyAV backend** for FFmpeg-backed video decoding.

Current configuration:

```text
ContentDetector threshold = 27.0
```

Detected scenes are split and stored under:

```text
sample_clips/<video_id>/
```

A segmentation report is also generated:

```text
sample_clips/<video_id>/segmentation_report.json
```

The report contains:

* scene count
* scene durations
* average scene duration
* minimum scene duration
* maximum scene duration
* detection time
* threshold

### Important

PySceneDetect is only a **scene/candidate generation mechanism**.

It does **not** determine whether a scene is humorous, hateful, sarcastic, ironic, or otherwise meaningful.

---

# 3. Candidate Formation

`src/candidate.py` converts detected scenes into dataset candidates.

## Natural scenes

Scenes with duration:

```text
≤ 30 seconds
```

are retained as a single natural-scene candidate.

Example:

```text
Scene-001
```

The complete scene duration is used.

## Long scenes

Scenes longer than:

```text
30 seconds
```

are represented using multiple **10-second windows** distributed across the full scene.

Current configuration:

```text
Window duration: 10 seconds
Minimum candidates: 2
Maximum candidates: 6
```

The windows are distributed across the long scene rather than being generated as a fixed sequence of overlapping windows.

This provides candidate coverage across different portions of a long scene.

Candidate metadata is stored in:

```text
sample_clips/<video_id>/candidate_report.json
```

### Important

Candidate generation is not humor classification.

A candidate is simply a unit that will later be represented by multimodal features.

---

# 4. Multimodal Extraction

`src/extract.py` extracts the available modalities for every candidate.

Each candidate can contain:

```text
Text
Image
Audio
OCR
Emoji
```

## Text

Text is extracted using:

**Faster-Whisper**

Current transcription configuration uses:

* VAD filtering
* `beam_size=1`
* `condition_on_previous_text=False`

The resulting transcript is stored in the sample's `text` modality.

---

## Image

Two representative frames are extracted from each candidate.

The frames are sampled approximately at:

```text
1/3 of candidate duration
2/3 of candidate duration
```

Images are stored under the generated extraction directories.

The purpose is to provide visual information from different points within the candidate rather than relying on a single frame.

---

## Audio

Audio is extracted using **FFmpeg**.

Current audio format:

```text
Mono
16 kHz
WAV
```

Audio is stored as a candidate-specific file.

---

## OCR

OCR is performed using **EasyOCR**.

OCR is applied to both extracted representative images.

A persistent OCR worker is used so that EasyOCR does not need to be repeatedly initialized for every candidate.

OCR uses `mag_ratio=0.75` to reduce CPU processing time while retaining useful text extraction quality.

If no readable text is present in the sampled images, OCR may legitimately return:

```text
status: empty
```

This is not considered an extraction failure.

---

## Emoji

The current emoji field is **not visual emoji detection**.

Instead, it is derived heuristically from the transcript.

For example, transcript keywords related to:

```text
laugh / funny / haha / lol
love
angry
sad / cry
wow / surprise
shock
```

can produce corresponding emoji values.

Therefore:

```text
emoji.status = "derived"
```

means that the value was generated indirectly from extracted text.

---

# Dataset Output

For each processed video, the dataset is stored at:

```text
sample_clips/<video_id>/dataset.json
```

A simplified sample looks like:

```json
{
  "sample_id": "57_--62bZUQ-Scene-001",
  "source": {
    "video_id": "57_--62bZUQ",
    "video_file": "57_--62bZUQ.mp4",
    "source_scene": "Scene-001",
    "source_file": "57_--62bZUQ-Scene-001.mp4"
  },
  "timing": {
    "start_seconds": 0.0,
    "end_seconds": 7.774,
    "duration_seconds": 7.774
  },
  "modalities": {
    "text": {
      "status": "success",
      "value": "..."
    },
    "image": {
      "status": "success",
      "paths": [
        "..."
      ]
    },
    "audio": {
      "status": "success",
      "path": "..."
    },
    "ocr": {
      "status": "empty",
      "value": ""
    },
    "emoji": {
      "status": "derived",
      "value": "😂"
    }
  },
  "metadata": {
    "candidate_type": "natural_scene"
  }
}
```

The exact extracted paths depend on the local dataset structure.

---

# Modality Status

Each modality can have a status describing its result.

### `success`

Extraction completed and useful data was found.

### `empty`

Extraction completed successfully, but the modality was naturally absent.

For example:

```text
OCR = empty
```

can simply mean that there was no readable text in the sampled frames.

### `failed`

The extraction process encountered an actual processing failure.

### `derived`

The value was generated indirectly rather than directly detected.

Currently this applies to the transcript-derived emoji field.

---

# Missing Modalities Are Allowed

A sample does **not** need all five modalities to contain meaningful information.

For example:

```text
Text   → success
Image  → success
Audio  → success
OCR    → empty
Emoji  → derived
```

is a valid sample.

Do not convert naturally empty modalities into artificial values just to make every sample complete.

This is particularly important when scaling the dataset.

---

# Checkpointing and Resume

Dataset extraction is resumable.

After candidates are processed, the dataset is saved locally.

When extraction is run again, existing sample IDs are detected and skipped.

Therefore, if processing is interrupted:

```text
Run 1
  ↓
Some candidates completed
  ↓
Process interrupted
  ↓
Run pipeline again
  ↓
Completed candidates skipped
  ↓
Remaining candidates processed
```

This allows large datasets to be generated without restarting the entire extraction process after an interruption.

---

# Scaling to 8K+

The intended dataset target is approximately:

```text
8,000+ samples
```

Candidate yield depends on the source videos and their scene structure, so there is no fixed samples-per-minute conversion.

The general scaling relationship is:

```text
More source videos
        ↓
More detected scenes
        ↓
More candidates
        ↓
More multimodal samples
```

For large-scale collection, monitor:

* number of source videos
* scene count
* candidate count
* completed samples
* failed extractions
* empty modalities
* missing files
* disk usage
* processing time
* sample quality

The pipeline should first be tested on a small batch before scaling substantially.

---

# Recommended Scaling Workflow

For a new batch:

```text
1. Add/process source video
        ↓
2. Run pipeline
        ↓
3. Inspect candidate count
        ↓
4. Inspect dataset.json
        ↓
5. Validate files and sample IDs
        ↓
6. Continue scaling
```

Do not immediately modify segmentation or candidate-generation parameters simply because individual samples look imperfect.

The pipeline is intended to produce candidate samples for later dataset analysis and labeling.

---

# Important Rules

1. **PySceneDetect does not detect humor.**
2. Candidate generation does not assign humor labels.
3. Do not automatically assign humor, sarcasm, irony, or hate-speech labels during Phase 1.
4. Natural missing modalities are allowed.
5. `empty` does not mean `failed`.
6. Emoji is currently transcript-derived, not visually detected.
7. Two representative images are extracted per candidate.
8. Long scenes use distributed candidate windows.
9. Do not change segmentation or candidate parameters without validating the effect.
10. Keep generated videos, clips, audio, images, and datasets outside Git.
11. Test pipeline changes on a small dataset before large-scale processing.
12. Preserve checkpoint/resume behavior when modifying extraction logic.

---

# Git and Generated Data

The following generated directories are ignored:

```text
source_videos/
sample_clips/
__pycache__/
```

Generated datasets and media should not normally be pushed to GitHub.

The repository should contain:

```text
source code
configuration
requirements
documentation
```

rather than the full generated dataset.

Use focused Git commits for meaningful pipeline changes.

---

# Current Validation Status

The current Phase 1 pipeline has been validated end-to-end on a pilot dataset.

The validation covered:

* YouTube downloading
* scene detection
* natural-scene candidates
* long-scene candidate generation
* distributed long-scene candidate selection
* text extraction
* image extraction
* audio extraction
* OCR
* transcript-derived emoji
* checkpointing
* resume behavior
* multimodal dataset generation

The pilot produced:

```text
388 samples
0 duplicate sample IDs
0 pending candidates
0 missing audio files
0 missing image files
0 extraction errors
```

Empty modalities were observed where the corresponding information was naturally absent.

The pipeline is therefore ready for **controlled scaling and handoff**.

---

# Phase 1 Scope

Phase 1 is responsible for:

```text
Source Videos
     ↓
Scene Segmentation
     ↓
Candidate Generation
     ↓
Multimodal Feature Extraction
     ↓
Structured Dataset
```

Phase 1 does **not** perform:

```text
Humor Classification
Sarcasm Classification
Irony Classification
Hate-Speech Classification
Final Dataset Labeling
```

Those tasks belong to later stages of the project.

---

# Handoff

The intended operator workflow is:

```powershell
conda activate mmhd
python src/pipeline.py
```

Provide the YouTube URL when prompted.

For interrupted processing, rerun the pipeline. Existing completed candidates will be skipped during extraction.

Before large-scale processing, verify:

```text
1. Python environment is active
2. FFmpeg is installed
3. Dependencies are installed
4. Source video downloads correctly
5. Scene segmentation completes
6. Candidate report is generated
7. dataset.json is generated
8. Extracted audio/images exist
9. Sample IDs are unique
10. Empty modalities are interpreted correctly
```

The pipeline should be treated as the **validated dataset-generation baseline**. Any future modification should first be tested on a small controlled sample before being used for large-scale collection.
