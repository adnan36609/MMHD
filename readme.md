# MMHD Phase 1

Dataset-generation pipeline for **Multimodal Humor & Hate Speech Detection (MMHD)**.

Phase 1 converts source videos into structured multimodal samples containing:

**Text | Image | Audio | OCR | Emoji**

The pipeline is designed to be **reproducible, resumable, and scalable**.

> Phase 1 generates the dataset. It does not perform final humor, sarcasm, irony, or hate-speech classification.

---

# Pipeline

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

---

# Repository Structure

    MMHD/

    ├── source_videos/                 # Downloaded source videos (ignored by Git)
    ├── sample_clips/                  # Generated scenes, candidates and extracted data
    │
    ├── src/
    │   ├── download.py                # YouTube video downloading
    │   ├── segment.py                 # Scene detection and scene splitting
    │   ├── candidate.py               # Candidate generation
    │   ├── extract.py                 # Multimodal extraction
    │   ├── pipeline.py                # Main end-to-end pipeline
    │   └── analyze_scenes.py          # Scene analysis utility
    │
    ├── benchmark_cpu.py               # CPU performance benchmark
    ├── cleanup_duplicate.py           # Dataset cleanup utility
    ├── validate_dataset.py            # Dataset integrity and contract validation
    ├── semantic_redundancy_qc.py      # Semantic redundancy QC utility
    ├── semantic_redundancy_calibration.json
    │                                  # Calibration data for redundancy QC
    ├── cpu_environment.yml            # Exported CPU environment configuration
    ├── requirements.txt
    ├── .gitignore
    └── readme.md

Generated videos, clips, extracted modalities, and other large local data should not be committed to Git.

---

# Requirements

## Software

- Python 3.12
- FFmpeg
- Git
- Conda recommended

## Python Dependencies

The main pipeline dependencies include:

    faster-whisper==1.2.1
    av<19
    easyocr
    yt-dlp
    scenedetect

The complete CPU environment can also be reproduced from:

    cpu_environment.yml

---

# Setup

Create the Conda environment:

    conda create -n mmhd python=3.12

Activate it:

    conda activate mmhd

Install Python dependencies:

    pip install -r requirements.txt

Verify FFmpeg:

    ffmpeg -version

Verify Python:

    python --version

Expected:

    Python 3.12.x

---

# Running the Pipeline

The main entry point is:

    python src/pipeline.py

The program asks for a YouTube URL:

    Enter YouTube video URL:

Enter one URL and press Enter.

The pipeline performs:

    STEP 1: Download
            ↓
    STEP 2: Scene Detection
            ↓
    STEP 3: Candidate Formation
            ↓
    STEP 4: Multimodal Extraction

The current pipeline accepts **one YouTube URL per execution**.

To process another video, run the command again with the next URL.

Already downloaded videos are reused.

---

# Pipeline Stages

## 1. Download

`src/download.py` uses **yt-dlp** to download the supplied YouTube video.

Downloaded videos are stored in:

    source_videos/

The filename is based on the YouTube video ID:

    source_videos/<video_id>.mp4

If the video has already been downloaded, the existing file is reused.

---

# 2. Scene Detection

`src/segment.py` uses **PySceneDetect ContentDetector** with the **PyAV backend** for FFmpeg-backed video decoding.

Current configuration:

    ContentDetector threshold = 27.0

Detected scenes are split and stored under:

    sample_clips/<video_id>/

A segmentation report is generated:

    sample_clips/<video_id>/segmentation_report.json

The report contains:

- scene count
- scene durations
- average scene duration
- minimum scene duration
- maximum scene duration
- detection time
- threshold

### Important

PySceneDetect is only a **scene/candidate generation mechanism**.

It does **not** determine whether a scene is humorous, hateful, sarcastic, ironic, or otherwise meaningful.

---

# 3. Candidate Formation

`src/candidate.py` converts detected scenes into dataset candidates.

## Natural Scenes

Scenes with duration:

    ≤ 30 seconds

are retained as a single natural-scene candidate.

Example:

    Scene-001

The complete scene duration is used.

## Long Scenes

Scenes longer than:

    30 seconds

are represented using multiple **10-second windows** distributed across the full scene.

Current configuration:

    Window duration: 10 seconds
    Minimum candidates: 2
    Maximum candidates: 6

The windows are distributed across the long scene rather than being generated as a fixed sequence of overlapping windows.

This provides candidate coverage across different portions of a long scene while reducing unnecessary overlap.

Candidate metadata is stored in:

    sample_clips/<video_id>/candidate_report.json

### Important

Candidate generation is not humor classification.

A candidate is simply a unit that will later be represented by multimodal features.

---

# 4. Multimodal Extraction

`src/extract.py` extracts the available modalities for every candidate.

Each candidate can contain:

    Text
    Image
    Audio
    OCR
    Emoji

---

## Text

Text is extracted using **Faster-Whisper**.

The transcription configuration uses:

- VAD filtering
- `beam_size=1`
- `condition_on_previous_text=False`

The resulting transcript is stored in the sample's `text` modality.

---

## Image

Two representative frames are extracted from each candidate.

The frames are sampled approximately at:

    1/3 of candidate duration
    2/3 of candidate duration

The images are stored under the generated extraction directories.

The purpose is to provide visual information from different points within the candidate rather than relying on a single frame.

Every accepted dataset sample is required to contain two successfully extracted images.

---

## Audio

Audio is extracted using **FFmpeg**.

Current audio format:

    Mono
    16 kHz
    WAV

Audio is stored as a candidate-specific file.

---

## OCR

OCR is performed using **EasyOCR**.

OCR is applied to both extracted representative images.

A persistent OCR worker is used so that EasyOCR does not need to be repeatedly initialized for every candidate.

OCR uses:

    mag_ratio=0.75

to reduce CPU processing time while retaining useful text extraction quality.

If no readable text is present in the sampled images, OCR may legitimately return:

    status: empty

This is not considered an extraction failure.

---

## Emoji

The current emoji field is **not visual emoji detection**.

Instead, it is derived heuristically from the transcript.

For example, transcript keywords related to:

    laugh / funny / haha / lol
    love
    angry
    sad / cry
    wow / surprise
    shock

can produce corresponding emoji values.

Therefore:

    emoji.status = "derived"

means that the value was generated indirectly from extracted text.

---

# Dataset Output

For each processed video, the dataset is stored at:

    sample_clips/<video_id>/dataset.json

A simplified sample looks like:

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

The exact extracted paths depend on the local dataset structure.

---

# Modality Status

Each modality can have a status describing its result.

### `success`

Extraction completed and useful data was found.

### `empty`

Extraction completed successfully, but the modality was naturally absent.

For example:

    OCR = empty

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

    Text   → success
    Image  → success
    Audio  → success
    OCR    → empty
    Emoji  → derived

is a valid sample.

Do not convert naturally empty modalities into artificial values just to make every sample complete.

This is particularly important when scaling the dataset.

---

# Checkpointing and Resume

Dataset extraction is resumable.

Before skipping an existing sample, the extraction pipeline verifies that the stored record matches the current candidate definition.

The checkpoint validation includes:

- source scene
- source file
- start time
- end time
- candidate duration
- candidate type

If all values match, the candidate is skipped.

If a sample ID exists but its stored metadata does not match the current candidate definition, the record is treated as **stale** and the candidate is reprocessed.

This prevents stale dataset records from being incorrectly reused after candidate-generation changes.

The resulting workflow is:

    Run 1
        ↓
    Some candidates completed
        ↓
    Process interrupted
        ↓
    Run pipeline again
        ↓
    Matching completed candidates skipped
        ↓
    Remaining candidates processed
        ↓
    Stale records reprocessed when detected

This allows large datasets to be generated without restarting the entire extraction process after an interruption.

---

# Scaling to 8K–10K Samples

The intended dataset target is approximately:

    8,000–10,000 samples

Candidate yield depends on the source videos and their scene structure, so there is no fixed samples-per-minute conversion.

The general scaling relationship is:

    More source videos
            ↓
    More detected scenes
            ↓
    More candidates
            ↓
    More multimodal samples

For large-scale collection, monitor:

- number of source videos
- scene count
- candidate count
- completed samples
- diversity rejections
- failed extractions
- empty modalities
- missing files
- disk usage
- processing time
- sample quality

The pipeline should first be tested on a small batch before scaling substantially.

---

# Recommended Scaling Workflow

For a new batch:

    1. Add/process source video
            ↓
    2. Run pipeline
            ↓
    3. Inspect candidate count
            ↓
    4. Inspect dataset.json
            ↓
    5. Run validation
            ↓
    6. Continue scaling

Do not immediately modify segmentation or candidate-generation parameters simply because individual samples look imperfect.

The pipeline is intended to produce candidate samples for later dataset analysis and labeling.

---

# Validation

The latest full-corpus validation covers 7,289 accepted samples and 7,368 candidates.

Run validation with:

    python validate_dataset.py

## Current Structural Validation

| Check | Result |
|---|---:|
| Dataset samples | 7,289 |
| Unique sample IDs | 7,289 |
| Duplicate sample IDs | 0 |
| Candidates checked | 7,368 |
| Accepted samples | 7,289 |
| Diversity rejections | 79 |
| Candidate accounting problems | 0 |
| Candidate boundary problems | 0 |
| Candidate/dataset contract problems | 0 |
| Missing audio files | 0 |
| Missing images | 0 |
| Invalid image counts | 0 |
| Temporal overlaps detected | 0 |

Candidate accounting:

    7,289 accepted + 79 diversity rejections = 7,368 candidates

The structural, file-integrity, candidate-boundary, candidate-contract, and temporal-overlap checks passed in the latest validation run.

The exact-transcript-diversity audit reports a separate warning, documented below. Therefore, the final validation message should not be interpreted as proof that every quality check is warning-free.

---

# Modality Validation

Current full-corpus modality-status counts:

| Modality | Success | Empty | Failed |
|---|---:|---:|---:|
| Text | 5,674 | 616 | 999 |
| Image | 6,159 | 0 | 1,130 |
| Audio | 6,290 | 0 | 999 |
| OCR | 3,865 | 2,294 | 1,130 |
| Emoji | 273 | 7,016 | 0 |

The image and audio tables use zero for the empty status because the current audit reports those modalities as success or failed. These counts describe extraction status; they do not necessarily measure semantic usefulness.

Emoji values are currently derived heuristically from transcripts rather than detected directly from video pixels.

An empty modality is not equivalent to an extraction failure. For example, OCR can complete successfully and return no readable text.

---

# Redundancy and Diversity

## Exact Transcript Diversity

The production pipeline applies an exact-text diversity constraint when accepting candidates. However, the latest full-corpus validation still found repeated normalized transcripts among accepted samples.

    Samples with usable text: 5,674
    Exact duplicate groups: 8
    Samples in duplicate groups: 26
    Candidates rejected for diversity: 79

Examples of repeated phrases include "oh my god", "no no no", and "thank you very much".

The 79 diversity rejections count candidates rejected during production extraction. They do not imply that the accepted corpus has zero repeated transcripts.

The current validator reports the duplicate groups as a warning. Investigate these remaining repetitions before claiming that the full corpus is free of exact transcript duplicates.

## Semantic Redundancy QC

Semantic redundancy inspection remains a quality-control diagnostic rather than an automatic production deduplication mechanism.

Semantically similar samples may still contain different audio, visual context, or dialogue and should not automatically be removed.

An earlier semantic-QC snapshot examined a smaller subset:

    Samples with usable text: 433
    Exact duplicate groups reported by that run: 0

This is a historical result from a different subset, not a current full-corpus result. It should not be used to override the latest audit of 5,674 samples with usable text.

The semantic redundancy tooling and calibration remain available in:

    semantic_redundancy_qc.py
    semantic_redundancy_calibration.json

---

# Baseline and Performance

The currently validated CPU pipeline represents the **CPU execution baseline**.

The production candidate definitions, dataset schema, extraction semantics, and validation requirements are treated as the baseline for GPU execution and future optimization.

## CPU Benchmark

The current CPU benchmark measured approximately:

    34.10 seconds/sample

Approximate CPU-only processing time at this measured rate:

    8,000 samples  → ~3.16 days
    10,000 samples → ~3.95 days

These are baseline estimates. Actual processing time can vary depending on hardware, video characteristics, scene structure, and modality workload.

The CPU benchmark can be run with:

    python benchmark_cpu.py

---

# GPU Support

The pipeline supports GPU acceleration for computationally expensive extraction components, primarily **Faster-Whisper** and **EasyOCR**.

GPU usage is controlled through the `MMHD_USE_GPU` environment variable.

## Enable GPU

PowerShell:

    $env:MMHD_USE_GPU="1"
    python src/pipeline.py

## Use CPU

PowerShell:

    $env:MMHD_USE_GPU="0"
    python src/pipeline.py

The GPU and CPU versions use the same:

- candidate definitions
- candidate IDs
- source scene/file references
- candidate timing
- candidate types
- dataset schema
- physical output requirements
- checkpoint/resume behavior
- extraction semantics
- validation requirements

The GPU implementation was compared against the CPU implementation on a fixed test subset.

The comparison verified:

- output consistency
- transcript consistency
- modality extraction consistency
- two-image extraction requirement
- dataset schema compatibility
- extraction behavior

The tested GPU implementation produced equivalent outputs for the fixed comparison subset while reducing processing time.

The fixed benchmark measured approximately:

    GPU speedup: ~2.52×

The benchmark was performed on a small fixed subset and should be treated as a measured optimization result rather than a guaranteed end-to-end speedup for every video or hardware configuration.

For systems without a compatible GPU, use:

    $env:MMHD_USE_GPU="0"

and run the same pipeline using CPU execution.

---

# Important Rules

1. **PySceneDetect does not detect humor.**

2. Candidate generation does not assign humor labels.

3. Do not automatically assign humor, sarcasm, irony, or hate-speech labels during Phase 1.

4. Natural missing modalities are allowed.

5. `empty` does not mean `failed`.

6. Emoji is currently transcript-derived, not visually detected.

7. Two representative images are extracted per accepted candidate.

8. Long scenes use distributed candidate windows.

9. Exact transcript diversity is enforced during production extraction.

10. Semantic similarity is treated as a QC signal rather than automatic deduplication.

11. Do not change segmentation or candidate parameters without validating the effect.

12. Keep generated videos, clips, audio, images, and datasets outside Git.

13. Test pipeline changes on a small dataset before large-scale processing.

14. Preserve checkpoint/resume behavior when modifying extraction logic.

15. Do not change the dataset schema without revalidating the pipeline.

16. Treat the validated CPU pipeline as the reference baseline when evaluating GPU or future performance changes.

17. GPU optimization must preserve the validated candidate-generation and extraction behavior.

---

# Git and Generated Data

The following generated directories are ignored:

    source_videos/
    sample_clips/
    __pycache__/

Generated datasets and media should not normally be pushed to GitHub.

The repository should contain:

    source code
    configuration
    requirements
    validation tools
    documentation

rather than the full generated dataset.

Use focused Git commits for meaningful pipeline changes.

---

# Handoff

The intended operator workflow is:

    conda activate mmhd

    $env:MMHD_USE_GPU="1"

    python src/pipeline.py

For CPU execution:

    conda activate mmhd

    $env:MMHD_USE_GPU="0"

    python src/pipeline.py

Provide the YouTube URL when prompted.

For interrupted processing, rerun the pipeline.

Existing candidates whose stored records match the current candidate definition will be skipped. Stale records are reprocessed when a mismatch is detected.

Before large-scale processing, verify:

    1. Python environment is active
    2. FFmpeg is installed
    3. Dependencies are installed
    4. GPU/CPU execution mode is configured correctly
    5. Source video downloads correctly
    6. Scene segmentation completes
    7. Candidate report is generated
    8. dataset.json is generated
    9. Extracted audio/images exist
    10. Sample IDs are unique
    11. Candidate ↔ dataset consistency passes
    12. Empty modalities are interpreted correctly
    13. Exact transcript diversity validation passes
    14. Checkpoint/resume behavior works

For GPU execution, verify that the environment variable is set before starting the pipeline:

    $env:MMHD_USE_GPU="1"

For CPU execution:

    $env:MMHD_USE_GPU="0"

The pipeline should be treated as the **validated dataset-generation baseline**.

Any future modification should first be tested on a small controlled sample and validated before being used for large-scale collection.

---

# Phase 1 Scope

Phase 1 is responsible for:

    Source Videos
         ↓
    Scene Segmentation
         ↓
    Candidate Generation
         ↓
    Multimodal Feature Extraction
         ↓
    Structured Dataset

Phase 1 does **not** perform Humor Classification.

---

# Humor Classification Experiments (Separate Research Stage)

Phase 1 remains responsible for dataset generation and multimodal extraction. The following experiments were conducted separately using the annotated pilot; they do not change the scope of the Phase 1 pipeline.

## Pilot and Evaluation Subsets

The initial completed pilot contains 240 manually labeled samples:

    Humorous:      82
    Non-humorous: 158

Different experiments used different modality-eligible subsets. The main subsets included 191 samples for text experiments, 232 for the audio baseline, 185 for the text/audio/visual comparison, and 145 for OCR-related comparisons.

The reported experiments used five-fold GroupKFold evaluation by source video. Scores from different subsets should not be treated as a single directly comparable ranking.

## Main Findings

- In the 185-sample text/audio/visual comparison, audio-only achieved 58.09% balanced accuracy and 57.18% macro F1. Its accuracy was 57.84%, below the 61.62% majority-class baseline.
- On the 145-sample OCR-eligible subset, raw OCR-only achieved 67.59% accuracy, 58.93% balanced accuracy, and 59.12% macro F1. Its accuracy was only 0.69 percentage points above the majority baseline.
- Removing selected branding and promotional phrases reduced OCR-only macro F1 from 59.12% to 42.13%. This indicates sensitivity to the removed text, but does not prove that branding alone explains the original performance.
- Simple multimodal fusion did not consistently outperform the strongest individual modality.
- The neighboring-scene context experiment used nearby scene transcripts as a proxy, not verified conversational turns, and did not demonstrate a meaningful improvement.
- Emoji coverage was too sparse to establish the value of emoji as a supervised modality in the current pilot.
- These are exploratory binary humor-recognition experiments. They do not constitute a completed hate-speech classifier or a demonstrated reliable humor-understanding system.

The full experimental tables, evaluation details, and limitations are documented in [Research Results](docs/research_results.md).

A second annotation round has been prepared with 300 new samples and a 100-sample independent review subset. Its results remain pending until the annotation files are returned and validated.
