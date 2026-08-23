# Multimodal Multiparty Humour Detection (Phase 1) 🎭

This repository contains the automated data extraction pipeline for creating a large-scale multimodal, multiparty humor detection dataset. 

Our goal is to build the foundation for an advanced AI model capable of detecting humor across five distinct modalities, ultimately translating this into a Hindi-language benchmark.

## The Pipeline Architecture

Instead of manually clipping videos and noting timestamps, we built an automated Python engine that takes a raw video file and automatically outputs a structured dataset containing 5 modalities:

* **Video/Context:** Automated scene detection using `PySceneDetect` to isolate multiparty camera cuts.
* **Audio:** Isolated `.wav` extraction via `FFmpeg`.
* **Text (Transcript):** AI Speech-to-Text generation using `OpenAI Whisper`.
* **OCR (Optical Character Recognition):** On-screen text detection using `EasyOCR`.
* **Emoji (Sentiment):** Rule-based NLP mapping translating dialogue context into reactive emojis.

##  Quick Start for the Junior Team

To scale this dataset to our 1,000+ clip milestone, please follow these steps on your local Windows machines.

1. **Install System Requirements:**
   Install FFmpeg via command prompt: `winget install ffmpeg` (Restart terminal after installing).

2. **Install Python Libraries:**
   `python -m pip install -U yt-dlp scenedetect[opencv] openai-whisper easyocr googletrans==4.0.0-rc1`

3. **Run the Extraction:**
   * Download a source video using `yt-dlp`.
   * Chop the scenes using `scenedetect -i raw_video.mp4 split-video -o dataset_clips/`.
   * Run our master extraction script: `python src/extract_hindi.py`.
   * Validate the outputs in the JSON file.