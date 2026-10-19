# MMHD: Multimodal Humor Detection

MMHD is a research prototype for **scene-level binary humor detection** using information extracted from video scenes: text, images, audio, OCR, and transcript-derived emoji features. The project combines a reproducible dataset-generation pipeline with supervised baseline evaluation, exploratory multimodal experiments, full-corpus inference, model-evidence explanations, uncertainty indicators, and a Streamlit dashboard.

**Task scope:** the prediction target is humor only (`humorous` / `non_humorous`). The project does not implement separate classifiers for other content categories.

> MMHD is an ongoing research prototype. The implemented baseline and experimental branches are distinguished below from the proposed full architecture. Experimental components should not be described as validated or integrated into the primary inference path unless stated explicitly.

## At a glance

| Item | Current status |
|---|---:|
| Candidate records generated | 7,368 |
| Accepted scene-level records | **7,289** |
| Source videos | 24 |
| Human-annotation records available | 540 |
| Primary source-video-grouped evaluation | 391 samples |
| Five-stream ablation evaluation | 343 samples |
| Full-corpus prediction rows | 7,289 |
| Records with a primary inference prediction | 6,290 |
| Records marked unavailable by the primary inference pipeline | 999 |

The accepted dataset accounting is `7,289 accepted + 79 diversity rejections = 7,368 candidates`. The corpus is not fully human-labeled: full-corpus inference is not the same as supervised training or held-out evaluation.

## 1. Project workflow

### Dataset-generation pipeline

```text
Source video / YouTube URL
          |
          v
        yt-dlp
          |
          v
   Scene segmentation
      PySceneDetect
          |
          v
    Candidate formation
          |
          v
   Multimodal extraction
   +------+------+------+------+------+
   |      |      |      |      |      |
 Text  Images  Audio  OCR   Emoji-derived
          |
          v
   Per-video dataset.json
          |
          v
   Dataset validation
```

### Implemented primary inference path

```text
Validated scene corpus
          |
          v
Modality availability check
          |
          +---------------------------+
          |                           |
          v                           v
     Text model                  Audio model
          |                           |
          +-------------+-------------+
                        |
                        v
       Fixed 50:50 text-audio probability fusion
              when both outputs are available
                        |
             +----------+----------+
             |                     |
             v                     v
       Text + audio             Audio-only
          fusion                 fallback
             |                     |
             +----------+----------+
                        |
                        v
                Humor prediction
                        |
                        v
     Model evidence and uncertainty indicators
                        |
                        v
       Structured output and dashboard

No supported model input -> explicit unavailable status
```

The primary full-corpus inference path uses text and audio, with an audio-only fallback. OCR, emoji, visual, context, ViT, and cross-modal-attention work exists in separate experiments; these experimental branches are **not all integrated into the primary full-corpus inference pipeline**.

## 2. Five input modalities

| Modality | Extraction / representation | Current corpus coverage |
|---|---|---:|
| Text | Speech transcripts generated with Faster-Whisper | 5,674 / 7,289 (77.8%) |
| Images | Two representative keyframes from each candidate | 6,159 / 7,289 (84.5%) |
| Audio | Mono 16 kHz WAV extracted with FFmpeg | 6,290 / 7,289 (86.3%) |
| OCR | EasyOCR applied to representative frames | 3,865 / 7,289 (53.0%) |
| Emoji | Heuristic information derived from transcripts | 273 / 7,289 (3.7%) |

The modality counts overlap and must not be added together. The emoji field is **transcript-derived**; it is not visual emoji detection. A modality can be naturally empty, unavailable, or failed. Missing modalities must not be replaced with fabricated content.

### Dataset schema and local assets

Each processed video has a `sample_clips/<video_id>/dataset.json` file. Records include a stable sample ID, source video and scene references, timing, candidate metadata, modality status, transcript/OCR values, and paths to extracted images/audio.

Typical local directories:

```text
MMHD/
├── source_videos/       # Downloaded source videos; ignored by Git
├── sample_clips/        # Dataset JSON and extracted media; ignored by Git
├── src/                 # Extraction, inference, experiments, and dashboard code
├── annotations/         # Local annotation files; some are not tracked by Git
├── results/             # Metrics and generated experiment artifacts
├── docs/
│   └── research_results.md
├── MMHD_TECHNICAL_HANDOFF.md
├── requirements.txt
├── cpu_environment.yml
├── validate_dataset.py
├── benchmark_cpu.py
└── readme.md
```

Generated media, large result CSVs, trained model files, and embedding caches may be excluded from Git. A fresh clone may therefore **not** contain everything required to reproduce the current outputs or launch every dashboard panel. See [MMHD Technical Handoff](MMHD_TECHNICAL_HANDOFF.md) for the artifact checklist and verified results.

## 3. Dataset validation and quality notes

The latest documented full-corpus structural validation reported:

| Check | Result |
|---|---:|
| Unique sample IDs | 7,289 |
| Duplicate sample IDs | 0 |
| Candidate accounting problems | 0 |
| Candidate boundary problems | 0 |
| Dataset-contract problems | 0 |
| Temporal-overlap pairs detected | 0 |
| Diversity rejections | 79 |

The audit also reported **eight exact-transcript duplicate groups involving 26 accepted samples**. This remains a warning; do not describe the accepted corpus as having zero repeated transcripts or automatically remove repeated phrases without reviewing their scene context.

Structural validity does not guarantee semantic quality. Transcripts may be noisy, OCR may capture channel branding or promotional text, and a present modality may not contain useful humor-related evidence.

## 4. Supervised evaluation results

The results below are from source-video-grouped evaluation. Evaluation subsets differ in size and composition; do not rank values from different subsets as if they were a single, directly comparable experiment.

### Primary expanded comparison

The primary complete-case evaluation used **391 annotated samples** from 24 source videos (179 humorous and 212 non-humorous).

| Configuration | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| Text-only | 54.99% | 54.98% | — |
| Audio-only | 62.66% | 62.48% | — |
| Feature-level text + audio | 61.89% | 61.53% | — |
| Equal-weight text-audio late fusion | **63.17%** | **63.06%** | **63.17%** |

Late fusion was the highest-scoring configuration in this comparison. Its Macro F1 advantage over audio-only is small (about 0.00584); the saved video-cluster bootstrap 95% interval for the difference includes zero. This result does **not** establish a reliable improvement. Fusion weights are fixed at 50:50 and the model probabilities are not calibrated.

### Five-stream ablation

The separate five-stream comparison evaluated **343 of the 540 annotation records** on a shared complete-case subset. Records lacking OCR were excluded from this particular comparison.

It evaluated text-only, audio-only, visual-only, OCR-only, and combinations of available streams, including a configuration using all five streams.

| Configuration | Macro F1 | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| Audio-only | **60.47%** | 60.93% | **61.16%** |
| Text + audio | 60.45% | 61.22% | 60.81% |
| All five streams | 52.53% | 53.94% | 52.58% |

With the current representations and fusion approaches, adding more streams did not consistently improve results. This does not establish that those modalities are inherently unhelpful; it describes the evaluated implementations and subsets.

Full metric tables and interpretation notes are in [Research Results](docs/research_results.md) and [MMHD Technical Handoff](MMHD_TECHNICAL_HANDOFF.md).

## 5. Full-corpus inference and outputs

The existing inference output contains 7,289 rows with unique sample IDs:

| Inference method/status | Records |
|---|---:|
| Text-audio late fusion | 5,674 |
| Audio-only fallback | 616 |
| No applicable model; marked unavailable | 999 |
| **Total records** | **7,289** |

The 6,290 scored rows are inference outputs, not human-verified labels and not out-of-fold evaluation predictions. No model score is fabricated for the 999 unavailable rows.

The project also has structured prediction outputs and explanation/evidence artifacts. Humor-target extraction is experimental and heuristic-based; candidate targets are not independently verified. Explanation text describes model evidence/associations, not necessarily the actual reason a human finds a scene funny. Decision-boundary and confidence labels are heuristic, and probabilities are not calibrated.

## 6. Additional research experiments

These are separate experimental branches. They do not replace or feed into the primary full-corpus inference pipeline.

### Neighboring-scene context

A context-enrichment artifact covers all **7,289 scene records** across 24 videos. At least one neighboring scene had text for 6,116 records (83.9%). This is context derived from chronological scene order, not verified dialogue turns.

The contextual model evaluated on the 391-sample set reported Macro F1 **0.5471**, compared with **0.6306** for the primary text-audio baseline. The context model did not outperform the baseline. Saved fold assignments may differ, so this comparison carries that qualification. Context remains exploratory and does not alter primary predictions.

### ViT keyframe and cross-frame experiments

Frozen ViT features were extracted for annotated-scene keyframes: 1,048 of 1,080 frame records had successful embeddings; 524 scenes had both frames available. Cross-frame representations were tested with visual-only classifiers. The strongest listed visual-only representation by Macro F1 was the absolute difference between frame embeddings (0.5219), which did not outperform the primary text-audio baseline.

A visual late-fusion experiment also tested multiple visual probability weights. None of the tested weights improved on the text-audio baseline. These experiments are exploratory and are not integrated into primary full-corpus inference. They are not a reproduction of CIRM.

### Text-visual cross-modal attention

A lightweight experiment used frozen text-token representations, image embeddings, and trainable text-visual cross-attention. It evaluated 391 samples using five-fold source-video-grouped out-of-fold predictions:

| Metric | Text-visual attention |
|---|---:|
| Macro F1 | 54.26% |
| Accuracy | 54.48% |
| Balanced accuracy | 54.28% |
| ROC-AUC | 0.5574 |

This experiment did not outperform the primary text-audio baseline and is not integrated into the primary inference path. The probabilities are not calibrated. These experiments are adaptations informed by multimodal research themes, not faithful reproductions of the referenced papers.

## 7. Dashboard

The Streamlit dashboard (`src/mmhd_dashboard.py`) provides access to the documented experiments and saved outputs, including:

- **Full Corpus Explorer:** inspect scene records, available inputs, neighboring-scene context, and experimental scores when present.
- **Evaluated Samples:** inspect saved out-of-fold evaluation predictions; this view is restricted to records with evaluation predictions.
- **Experiments & Findings:** compare the five-stream ablation configurations and inspect individual-sample results.
- **Architecture Status:** distinguish the primary inference path, experimental branches, limitations, and proposed future work.

The dashboard's key interface behavior and Python syntax checks were verified in the project workflow on 19 October 2026. Run it again in the actual presentation environment to confirm that required data, models, dependencies, and saved artifacts are available. A successful syntax check alone does not guarantee runtime readiness.

## 8. Research inspiration and roadmap

The mentor's intended architecture is:

1. Text, images, audio, OCR, and emoji input
2. Encoder
3. Multimodal alignment
4. Cross-image reasoning
5. Conversation memory
6. Speaker-listener graph
7. Cultural retrieval-augmented context
8. Multimodal large language model (MLLM)
9. Humor target
10. Explanation generator
11. Confidence
12. Final output

Current implementation status:

| Stage | Status |
|---|---|
| 1. Input modalities | Extraction and experiments exist; coverage varies by modality |
| 2. Encoder | Modality-specific features/models exist; a unified five-modality encoder pipeline is not established |
| 3. Multimodal alignment | Fixed-weight text-audio late fusion exists; learned alignment remains future work |
| 4. Cross-image reasoning | ViT cross-frame and text-visual attention experiments exist; the full intended reasoning module is not implemented |
| 5. Conversation memory | Neighboring-scene context exists as an exploratory branch; full memory is not integrated into primary inference |
| 6. Speaker-listener graph | Not implemented in the documented system |
| 7. Cultural RAG | Not implemented in the documented system |
| 8. MLLM | Not integrated into the proposed full architecture |
| 9. Humor target | Experimental heuristic candidate extraction exists; not validated |
| 10. Explanation generator | Limited model-evidence explanations exist |
| 11. Confidence | Heuristic uncertainty indicators exist; not calibrated |
| 12. Final output | Structured outputs and dashboard exist for the current system |

### Research references

The project takes methodological inspiration from the following work. These references do not imply full reproduction of their methods. **MMHD's prediction objective remains humor detection only.**

- **M2H2: A Multimodal Multiparty Hindi Dataset for Humor Recognition in Conversations.** Motivation for multimodal humor recognition and conversational context. [Paper](https://arxiv.org/abs/2108.01260)
- **Detecting Emotional Incongruity of Sarcasm by Commonsense Reasoning (EICR).** A source of transferable ideas about retrieved context and structured reasoning; these ideas may be adapted to humor, without adding another classification task. [ACL Anthology](https://aclanthology.org/2025.coling-main.608/)
- **CFMS: Towards Explainable and Fine-Grained Chinese Multimodal Sarcasm Detection Benchmark.** Inspiration for fine-grained target/explanation design, adapted to humor and not a reproduction of the benchmark or its training method. [Manuscript](https://openreview.net/pdf/77f4cb13b3274cc3cde801fdf1e8a35c00d1dd89.pdf)
- **Additional visual-reasoning reference: MMSD3.0 / CIRM.** Relevant to cross-image and text-image reasoning; MMHD's ViT/cross-frame experiments are not a reproduction of CIRM. [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_MMSD3.0_A_Multi-Image_Benchmark_for_Real-World_Multimodal_Sarcasm_Detection_CVPR_2026_paper.html)

## 9. Setup

### Requirements

- Python 3.12
- Git
- FFmpeg available on `PATH`
- Conda recommended

The dataset-generation pipeline's direct dependencies are listed in `requirements.txt`; `cpu_environment.yml` provides the project's CPU environment. Research experiments and the dashboard may require their associated modeling/dashboard dependencies and local artifacts. Verify those against the actual environment before assuming a fresh clone is demo-ready.

Example setup for the extraction pipeline:

```powershell
conda create -n mmhd python=3.12
conda activate mmhd
pip install -r requirements.txt
ffmpeg -version
python --version
```

Expected Python version: `3.12.x`.

## 10. Run the dataset-generation pipeline

The main entry point accepts one YouTube URL per execution:

```powershell
python src/pipeline.py
```

Follow the prompt to enter a URL. Existing downloads and valid processed candidates may be reused. The pipeline supports checkpoint/resume; rerunning it can continue processing after an interruption.

Enable GPU acceleration for supported extraction components in PowerShell:

```powershell
$env:MMHD_USE_GPU="1"
python src/pipeline.py
```

Use CPU mode instead:

```powershell
$env:MMHD_USE_GPU="0"
python src/pipeline.py
```

GPU acceleration is primarily used for Faster-Whisper and EasyOCR where the configured environment supports it. Extraction output and schema should be validated after changing execution environments.

## 11. Validate the dataset

From the repository root:

```powershell
python validate_dataset.py
```

The latest documented validation covered 7,289 accepted samples and 7,368 candidates. It reported zero duplicate sample IDs and zero contract problems, while separately warning about eight exact-transcript duplicate groups involving 26 accepted samples. Use the generated report/log for the actual run rather than assuming historical counts still describe a modified dataset.

## 12. Run the dashboard

When its dependencies, dataset assets, models, and expected result files are available:

```powershell
streamlit run .\src\mmhd_dashboard.py
```

The primary experiment outputs and some generated CSV/model/embedding artifacts are excluded from Git. If the dashboard reports missing artifacts after a fresh clone, supply the required local artifacts or regenerate them using the corresponding scripts, then verify the app at runtime.

## 13. Useful documentation and scripts

- [`MMHD_TECHNICAL_HANDOFF.md`](MMHD_TECHNICAL_HANDOFF.md) — verified metrics, artifact locations, architecture boundaries, dashboard checks, and report/PPT guardrails.
- [`docs/research_results.md`](docs/research_results.md) — earlier experimental results and methodological limitations.
- `src/predict_full_corpus_humor.py` — primary full-corpus inference.
- `src/explain_full_corpus_humor.py` — evidence and explanation generation.
- `src/compare_multimodal_540.py` — grouped modality/model comparisons.
- `src/compare_late_fusion_540.py` — late-fusion evaluation using saved out-of-fold predictions.
- `src/enrich_structured_output_with_context.py` — neighboring-scene context enrichment.
- `src/extract_vit_keyframe_embeddings_540.py` — frozen ViT keyframe embeddings.
- `src/compare_cross_frame_vit_540.py` — cross-frame visual experiments.
- `src/train_cross_modal_attention_391.py` — text-visual cross-attention experiment.

Confirm each script and artifact exists in the checked-out version before relying on it.

## 14. Limitations and reporting rules

- Only human-annotated subsets support the documented supervised evaluation; the whole corpus is not human-labeled.
- Full-corpus inference outputs are not evaluation predictions.
- The primary inference path does not integrate every modality or experimental model.
- Current late-fusion improvement over audio-only is small and not established as reliable.
- The contextual and cross-modal experiments did not outperform the primary baseline in their documented comparisons.
- OCR can capture irrelevant overlays, and the derived emoji signal is sparse and heuristic.
- Humor-target candidates are not independently verified.
- Explanations reflect model evidence and are not causal explanations of human humor.
- Confidence indicators are heuristic; model probabilities are not calibrated.
- The research papers inform selected design choices; the project does not claim faithful reproduction of their complete architectures.
- Unimplemented stages in the intended architecture must remain labelled as future work.

## License and data

Review the source-video terms, dataset distribution restrictions, and licenses of third-party software and models before redistributing generated media or trained artifacts. Large local datasets, media, outputs, and caches should not be committed to Git unless there is an explicit, appropriate reason and authorization.
