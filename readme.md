# Multimodal Humor & Hate Speech Detection (MMHD) - Phase 1

This repository contains the Phase 1 automated data extraction pipeline for processing raw video clips into a highly optimized, 5-modality dataset. This blueprint prepares raw media for downstream multimodal machine learning models.

## 🚀 Core Optimizations
To handle large-scale dataset generation (5,000–8,000 clips) efficiently, this pipeline is optimized for maximum hardware utilization:
* **Parallel Processing:** Implements Python's `ThreadPoolExecutor` to process multiple scenes concurrently, eliminating sequential bottlenecks and utilizing all available CPU cores.
* **Faster-Whisper Integration:** Replaced the standard OpenAI Whisper library with `faster-whisper` (CTranslate2 backend) using INT8 quantization, achieving up to 4x faster speech-to-text processing on standard CPUs.

## 🧠 Extracted Modalities
1. **Video:** Isolated scene `.mp4` (via SceneDetect & FFmpeg)
2. **Audio:** Extracted `.wav` (via FFmpeg)
3. **Visual Frame:** Exact middle keyframe `.jpg` (via FFmpeg)
4. **Transcript:** Dialogue extraction (via Faster-Whisper)
5. **OCR:** Background text detection (via EasyOCR)
* **Bonus:** Automated heuristic emoji mapping based on transcript context.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/saurav80325-create/MMHD-phase1.git](https://github.com/saurav80325-create/MMHD-phase1.git)
   cd MMHD-phase1
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: FFmpeg must be installed and added to your system PATH).*

## 🏃‍♂️ How to Run

1. Place your raw `.mp4` video clips into the `dataset_clips/` directory. Use `yt-dlp` and `scenedetect` to download and split your source videos.
2. Execute the extraction script from the root directory:
   ```bash
   python src/extract.py
   ```
3. The pipeline will dynamically assemble a `master_dataset.json` file containing the aligned metadata and local file paths for every scene.

## ☁️ Scaling to 8,000+ Clips
For massive dataset generation, executing on a local CPU is not recommended. 
1. Clone this repository into a **Google Colab** or Kaggle notebook environment.
2. In `src/extract.py`, update EasyOCR to utilize the GPU: `reader = easyocr.Reader(['en'], gpu=True)`.
3. Leverage the free NVIDIA GPU acceleration to process thousands of clips in hours instead of weeks.