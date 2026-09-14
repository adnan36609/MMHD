# MMHD Phase 1

Dataset-generation pipeline for **Multimodal Humor & Hate Speech Detection (MMHD)**.

Phase 1 converts source videos into structured samples containing:

**Text | Image | Audio | OCR | Emoji**

The pipeline is designed to be reproducible, resumable, and scalable.

> Phase 1 generates the dataset. It does not perform final humor, sarcasm, irony, or hate-speech classification.

## Pipeline

```text
YouTube URLs
    ↓
yt-dlp
    ↓
PySceneDetect
    ↓
Candidate Formation
    ↓
Multimodal Extraction
    ↓
Text + Image + Audio + OCR + Emoji
    ↓
dataset.json
```

## Project Structure

```text
MMHD-phase1/
├── source_videos/          # Downloaded videos (ignored by Git)
├── sample_clips/           # Generated clips/data (ignored by Git)
├── src/
│   ├── download.py         # YouTube downloading
│   ├── segment.py          # Scene detection
│   ├── candidate.py        # Candidate generation
│   ├── extract.py          # Multimodal extraction
│   ├── pipeline.py         # End-to-end pipeline
│   ├── analyze_scenes.py   # Scene analysis utility
│   └── extract_backup.py   # Backup extractor
├── requirements.txt
├── .gitignore
└── readme.md
```

## Requirements

* Python 3.12
* FFmpeg
* Git
* Conda recommended

Python dependencies:

```text
faster-whisper==1.2.1
easyocr
yt-dlp
scenedetect
```

## Setup

```powershell
conda create -n mmhd python=3.12
conda activate mmhd
pip install -r requirements.txt
ffmpeg -version
```

## Pipeline Stages

### 1. Download

`download.py` uses **yt-dlp** to download YouTube videos into `source_videos/`.

Already downloaded videos are skipped.

### 2. Scene Detection

`segment.py` uses **PySceneDetect ContentDetector**.

Current threshold:

```text
27.0
```

Detected scenes and their metadata are stored under:

```text
sample_clips/<video_id>/
```

PySceneDetect is a **candidate generator**, not a humor detector.

### 3. Candidate Formation

`candidate.py` converts scenes into dataset candidates.

* Scenes ≤30 seconds → kept as one natural scene.
* Scenes >30 seconds → 10-second windows with 5-second overlap.
* Final fragments shorter than 3 seconds are ignored.

Candidate metadata is stored in `candidate_report.json`.

### 4. Multimodal Extraction

`extract.py` extracts:

| Modality | Method                      |
| -------- | --------------------------- |
| Text     | Faster-Whisper              |
| Image    | Representative middle frame |
| Audio    | FFmpeg                      |
| OCR      | EasyOCR                     |
| Emoji    | Transcript-based heuristic  |

OCR uses a persistent worker process to avoid repeatedly loading EasyOCR.

The emoji field is currently **derived**, not genuine visual emoji detection.

## Dataset

Output:

```text
sample_clips/<video_id>/dataset.json
```

Example:

```json
{
  "sample_id": "57_--62bZUQ-Scene-001",
  "source": {
    "video_id": "57_--62bZUQ",
    "video_file": "57_--62bZUQ.mp4",
    "source_scene": "Scene-001"
  },
  "timing": {
    "start_seconds": 0.0,
    "end_seconds": 7.774,
    "duration_seconds": 7.774
  },
  "modalities": {
    "text": {"status": "success", "value": "..."},
    "image": {"status": "success", "path": "..."},
    "audio": {"status": "success", "path": "..."},
    "ocr": {"status": "empty", "value": ""},
    "emoji": {"status": "derived", "value": "😂"}
  },
  "metadata": {
    "candidate_type": "natural_scene"
  }
}
```

### Modality Status

* `success` — extraction completed and data was found.
* `empty` — extraction completed but the modality was naturally absent.
* `failed` — processing failed.
* `derived` — value was generated indirectly.

A sample does **not** need all five modalities to contain information. For example, an OCR value of `empty` can be a valid result when no text is visible.

## Checkpointing

Extraction is resumable.

After each candidate is processed, the dataset is saved. When the pipeline is restarted, already completed candidates are skipped.

This allows large datasets to be processed without restarting from zero after an interruption.

## Running

Use `pipeline.py` as the main entry point:

```powershell
python src/pipeline.py
```

The pipeline performs:

```text
Download → Segment → Generate Candidates → Extract Modalities
```

## Adding Videos

Add YouTube URLs to the `VIDEO_URLS` configuration:

```python
VIDEO_URLS = [
    "https://www.youtube.com/watch?v=VIDEO_ID_1",
    "https://www.youtube.com/watch?v=VIDEO_ID_2"
]
```

Then run:

```powershell
python src/pipeline.py
```

## Scaling

The target is approximately **8,000+ candidate samples**.

Candidate yield depends on video content, so there is no fixed samples-per-minute rate.

For scaling:

```text
More source videos
       ↓
More detected scenes
       ↓
More candidates
       ↓
More multimodal samples
```

Before large-scale collection, monitor:

* candidate count
* processing failures
* OCR failures
* missing modalities
* disk usage
* sample quality

## Important Rules

1. **PySceneDetect does not detect humor.**
2. Do not automatically assign humor labels.
3. Natural missing modalities are allowed.
4. `empty` does not mean `failed`.
5. Emoji is currently a transcript-derived heuristic.
6. Do not change segmentation parameters without validating the effect.
7. Keep generated media outside Git.
8. Test pipeline changes on a small sample before large-scale processing.

Current segmentation configuration:

```text
Scene threshold: 27.0
Window duration: 10 seconds
Window overlap: 5 seconds
```

## Git

Generated data is ignored:

```text
source_videos/
sample_clips/
__pycache__/
```

Only source code, configuration, and documentation should normally be pushed to GitHub.

Use focused commits for meaningful pipeline changes.

## Current Status

The pipeline has been validated end-to-end for:

* YouTube downloading
* Scene detection
* Candidate generation
* Natural scenes
* Long-scene windows
* Text extraction
* Image extraction
* Audio extraction
* OCR
* Derived emoji
* Checkpointing
* Resume behavior
* End-to-end execution

The pipeline is now ready for **controlled scaling and production hardening** toward the larger dataset.
