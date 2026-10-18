# MMHD Technical Handoff

**Purpose:** This document is the technical source of truth for preparing the MMHD academic report, presentation, and dashboard demonstration. Keep the agreed scope fixed. Do not add major features or retrain models for this submission unless a critical validity issue is discovered and explicitly approved.

## 1. Project framing

**Suggested project title:** *MMHD: Scene-Level Humor Detection with Text–Audio Fusion and Interpretable Model Evidence*

**Current contribution:** A corpus-processing and validation pipeline for video-scene records; source-video-grouped text/audio/fusion evaluation; a separate 11-configuration five-stream ablation; full-corpus text–audio inference with an audio-only fallback; structured model-decision evidence and uncertainty indicators; and an interactive Streamlit explorer.

Describe this as a research prototype. Do not describe unimplemented modules as completed features.

## 2. Verified dataset and evaluation facts

- Accepted corpus records: **7,289**.
- Candidate records: **7,368**; diversity rejections: **79**; accounting reconciles as 7,289 + 79 = 7,368.
- Source videos: **24**.
- Human annotation records: **540**.
- Primary common complete-case multimodal evaluation: **391 samples**.
- The saved `results/expanded_540/multimodal_comparison/metrics.json` reports 143 samples missing text, 0 missing audio features, and 6 missing visual features. It identifies 391 samples with all modalities. Report these as the stated complete-case selection rationale; do not imply all 540 records were evaluated.
- Dataset validation reported **0 temporal-overlap pairs** and **0 contract problems**.
- The validation report warns about **8 exact-transcript duplicate groups involving 26 samples**. These are not automatically erroneous; common phrases can occur in distinct scenes. Do not delete them without a separate review.

## 3. Primary evaluated model results

Source: `results/expanded_540/late_fusion/metrics.json` and matching out-of-fold predictions.

| Configuration | Macro F1 | Accuracy |
|---|---:|---:|
| Text-only | 0.549846 | 0.549872 |
| Audio-only | 0.624809 | 0.626598 |
| Feature-level Text + Audio | 0.615292 | 0.618926 |
| Equal-weight Late Fusion (Text + Audio) | **0.630648** | **0.631714** |

For late fusion, balanced accuracy is **0.631680**. The evaluation uses 391 samples from 24 videos; label distribution is 179 humorous and 212 non-humorous.

The late-fusion macro F1 was independently recalculated from `results/expanded_540/late_fusion/oof_predictions.csv`, reproducing 0.6306481238520074 exactly. Saved labels agree with the 0.5 threshold and no evaluation values were missing in the checked fields.

**Interpretation constraint:** Late fusion is the highest-scoring configuration in this comparison, but its macro-F1 advantage over audio-only is small (about 0.00584). The saved cluster-bootstrap 95% interval for the macro-F1 difference is **[-0.0270, +0.0390]**, which includes zero; therefore, the observed gain does not establish a reliable improvement. Input probabilities are not calibrated. The fixed 50:50 weights were not tuned against evaluation labels.

## 3A. Five-stream multimodal ablation

Source: `results/expanded_540/five_stream_comparison/metrics.json` and `results/expanded_540/five_stream_comparison/oof_predictions.csv`. The Streamlit **Experiments & Findings** tab displays this comparison and supports inspecting an individual sample across the 11 configurations.

- Evaluated cases: **343** out of 540 annotation records. **197 records were excluded because OCR data was unavailable**; the configurations share the same complete-case subset and five-fold source-video-grouped folds.
- This is a different evaluation subset from the primary 391-sample comparison in Section 3. Do not rank numbers across the two experiments as a direct apples-to-apples comparison.
- The comparison covers text-only, audio-only, visual-only, OCR-only, text+OCR, text+audio, audio+OCR, text+audio+visual, text+audio+OCR, text+audio+visual+OCR, and all five streams. Emoji metadata is part of the five-stream configuration.

| Configuration | Macro F1 | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| Text-only | 0.588475 | 0.591837 | 0.596960 |
| Audio-only | **0.604730** | 0.609329 | **0.611594** |
| Visual-only | 0.503789 | 0.510204 | 0.507352 |
| OCR-only | 0.530701 | 0.618076 | 0.554966 |
| Text + OCR | 0.569435 | 0.623907 | 0.576423 |
| Text + Audio | 0.604500 | 0.612245 | 0.608112 |
| Audio + OCR | 0.585226 | 0.600583 | 0.585331 |
| Text + Audio + Visual | 0.499045 | 0.507289 | 0.501361 |
| Text + Audio + OCR | 0.584527 | 0.597668 | 0.585260 |
| Text + Audio + Visual + OCR | 0.510793 | 0.524781 | 0.511258 |
| All five streams | 0.525314 | 0.539359 | 0.525822 |

**Interpretation:** Audio-only has the highest macro F1 within this five-stream experiment (0.6047), with text+audio essentially tied (0.6045). The all-five-stream configuration scores 0.5253 macro F1. The current feature representations and fusion methods do not demonstrate that adding more streams reliably improves classification. This does not prove that those modalities are inherently unhelpful; it describes these implementations on this subset.

## 4. Full-corpus inference facts

Source: `results/full_corpus/humor_predictions_summary.json` and `results/full_corpus/humor_predictions.csv`.

- Corpus records: 7,289.
- Records with text predictions: 5,674.
- Audio paths attempted: 6,290; usable audio predictions: 6,290; recorded audio-processing errors: 0.
- Method counts: 5,674 `text_audio_late_fusion`, 616 `audio_only_fallback`, 999 `no_available_model`.
- Scored records: 6,290. The 999 no-model rows are explicitly labeled `unavailable`; therefore, counting non-null `humor_prediction` values alone is not a valid scored-record count.
- All saved probabilities checked were within [0, 1]. Fusion probabilities matched the fixed 50:50 average. Audio-only fallback rows contained the required scores, no-model rows had no probability scores, and all scored labels agreed with the 0.5 decision threshold.
- The prediction CSV had 7,289 rows and 0 duplicate sample IDs.
- Full-corpus predictions are **not** out-of-fold evaluation results. Do not calculate research performance from this file or claim that all these predictions are human-verified.

## 5. Structured output and explanation facts

Relevant files:
- `results/full_corpus/humor_structured_output.csv`
- `results/full_corpus/humor_structured_output_summary.json`
- `results/full_corpus/humor_explanations.csv`
- `results/full_corpus/humor_explanations_summary.json`

Verified structured-output consistency:
- Both the base prediction CSV and structured output contain 7,289 rows.
- Structured sample IDs are unique.
- Sample-ID coverage, probabilities, labels, and prediction methods match the original prediction CSV exactly.

Structured target extraction is conservative and heuristic-based (lexical cues plus spaCy named-entity recognition). The summary reports 14 records with candidate targets, and states that candidates are not independently verified. Do not claim reliable or validated target identification.

The older explanation stage reports 5,456 records with text evidence, 6,290 with audio evidence, zero audio-explanation errors, 2,159 near-threshold predictions, and 661 cases where text/audio probability difference is at least 0.25. These 661 are **substantial probability-score disagreements**, not simply the number of cases where each model predicts opposite classes. The explanation text describes model-decision associations, not causal reasons a person finds a scene funny.

Decision-boundary margin and confidence labels are heuristic; probabilities are not calibrated. The older explanation output marks targets unknown/not yet implemented, while the later structured output has a separate experimental candidate heuristic. Clearly distinguish these stages.

## 6. Implemented architecture to show in the main report

```text
MMHD Scene Corpus
        |
Dataset Validation & Preprocessing
        |
Modality Availability Check
        |
  +-----+------+
  |            |
Text Model  Audio Model
  |            |
  +-----+------+
        |
Text–Audio Probability Fusion (fixed 50:50)
        |
  +-----+----------------+
  |                      |
Text+Audio available   Audio-only fallback
  |                      |
  +----------+-----------+
             |
       Humor Prediction
             |
Text/Audio Model Evidence + Uncertainty Flags
             |
Structured Output + Experimental Target Candidates
             |
       Streamlit Dashboard

Neither model available -> explicit unavailable prediction
```

The repository also contains separate research experiments/artifacts involving OCR, emoji, context, semantic text, and multimodal/five-stream comparisons. The five-stream ablation is visualized in the dashboard's **Experiments & Findings** tab. Mention these only with the exact experiment scope and corresponding evidence. Do not imply every auxiliary modality is used in the primary full-corpus scoring path.

## 6A. Additional experiments and dashboard verification (19 October 2026)

The following experiments were completed after the initial architecture and baseline evaluation. They are separate experimental branches. They do not replace or feed into the primary full-corpus text-audio inference pipeline.

### Context enrichment and contextual model

Artifacts:
- `src/enrich_structured_output_with_context.py`
- `results/full_corpus/humor_context_enriched.csv`
- `results/full_corpus/humor_context_enriched_summary.json`
- `results/expanded_540/context_experiment/`
- `results/expanded_540/model_comparison_391/`

The context-enriched corpus contains 7,289 unique scene records across 24 videos:
- Both previous- and next-scene text available: 5,193 records.
- Previous-scene text only: 466 records.
- Next-scene text only: 457 records.
- No neighboring-scene text: 1,173 records.
- At least one neighboring scene with text: 6,116 records (83.9%).

This is context derived from scene order, not verified conversational turns. It is displayed for inspection and does not alter the primary prediction.

On the shared 391-sample evaluation set, the saved context-model comparison reports:
- Macro F1: 0.547125.
- Accuracy: 0.547315.
- Balanced accuracy: 0.549502.
- ROC-AUC: 0.566512.

The primary text-audio baseline has macro F1 0.630648 and ROC-AUC 0.650680. The recorded context-model macro-F1 difference is -0.08352, with a video-cluster bootstrap 95% interval of [-0.13461, -0.03323]. Context is therefore exploratory and did not outperform the primary baseline. The saved context and baseline out-of-fold predictions may use different fold assignments, so this comparison should be interpreted with that qualification.

### Frozen ViT keyframe and cross-frame experiments

Artifacts:
- `src/extract_vit_keyframe_embeddings_540.py`
- `src/compare_cross_frame_vit_540.py`
- `src/compare_visual_late_fusion_540.py`
- `results/expanded_540/vit_keyframe_embeddings/`
- `results/expanded_540/cross_frame_vit/`
- `results/expanded_540/visual_late_fusion/`

Frozen ViT keyframe extraction generated 1,080 frame records for the 540 annotated scenes: 1,048 successful frame embeddings and 32 missing-path records. A total of 524 scenes had both keyframes available, and 16 had neither successful frame embedding. Both frame embeddings were available for all 391 primary evaluation samples.

Cross-frame visual-only experiment results:

| Visual representation | Macro F1 | ROC-AUC |
|---|---:|---:|
| Frame 1 | 0.457513 | 0.46777 |
| Frame 2 | 0.474961 | 0.47599 |
| Mean frame embeddings | 0.474824 | 0.44687 |
| Absolute frame difference | 0.521850 | 0.48477 |
| Ordered frame pair | 0.493180 | 0.48722 |
| Cross-frame relation features | 0.485526 | 0.47083 |

The absolute-difference representation was the strongest listed visual-only configuration by macro F1, but it did not outperform the primary text-audio baseline.

Visual late-fusion experiments combined the saved absolute-frame-difference probabilities with text-audio probabilities. The primary baseline on this comparison was macro F1 0.630648 and ROC-AUC 0.650680.

| Visual probability weight | Macro F1 | ROC-AUC |
|---|---:|---:|
| 0.10 | 0.60331 | 0.63513 |
| 0.20 | 0.58233 | 0.60700 |
| 0.30 | 0.56171 | 0.57890 |
| 0.50 | 0.51952 | 0.53499 |

None of these tested visual weights improved on the text-audio baseline. The ViT experiments are exploratory and are not integrated into primary full-corpus inference. The cross-frame experiment is not a reproduction of the MMSD3.0 CIRM method.

### Experimental text-visual cross-modal attention

Artifacts:
- `src/extract_roberta_text_embeddings_391.py`
- `src/train_cross_modal_attention_391.py`
- `results/expanded_540/cross_modal_reasoning/`
- `src/compare_context_crossmodal_391.py`
- `results/expanded_540/model_comparison_391/`

The experiment uses frozen text-token representations and image embeddings with a lightweight trainable text-visual cross-attention model. It evaluated 391 samples with five-fold source-video-grouped out-of-fold predictions.

| Metric | Text-visual attention |
|---|---:|
| Macro F1 | 0.542576 |
| Accuracy | 0.544757 |
| Balanced accuracy | 0.542795 |
| ROC-AUC | 0.557421 |

The saved comparison reports a macro-F1 difference of -0.08807 relative to the primary text-audio baseline, with a video-cluster bootstrap 95% interval of [-0.15043, -0.02769]. This experiment did not improve the primary result and must remain separate from the production-style inference path. Its probabilities are not calibrated.

These experiments are adaptations informed by multimodal research themes, not faithful reproductions of the referenced papers.

### Dashboard verification and validation (19 October 2026)

The following dashboard behavior has been observed in the current project workflow:
- **Full Corpus Explorer:** displays neighboring-scene context and experimental cross-modal attention scores where saved predictions exist.
- **Annotated scene safeguards:** warns that full-corpus direct inference is not an out-of-fold evaluation prediction.
- **Evaluated Samples:** provides access to saved out-of-fold results.
- **Experiments & Findings:** the five-stream ablation metrics and per-sample inspector were verified.
- **Architecture Status:** distinguishes the primary inference path, experimental branches, limitations, and proposed future work.
- **Sidebar:** clarifies that the Evaluated Samples tab is restricted to records with saved out-of-fold predictions, whereas Full Corpus Explorer includes all accepted records.

The command `python -m compileall -q .\src` completed successfully on 19 October 2026. `git diff --check` reported only an LF-to-CRLF line-ending warning for `src/mmhd_dashboard.py`, not a whitespace error. A successful syntax check does not replace runtime testing.

### Version-control and artifact policy

The repository .gitignore excludes `results/**/*.csv`, `results/**/*.png`, and `results/**/*.joblib`. Preserve this policy for large generated outputs and embedding artifacts rather than committing the full-corpus CSVs, ViT embeddings, or text-token embeddings.

The experiment scripts, dashboard changes, technical handoff, and small JSON metrics/summary files are suitable for version control. Generated CSVs, including out-of-fold prediction files, remain local artifacts under the current ignore policy.

A fresh clone may therefore lack files needed by some dashboard panels. Before presenting or handing off a fresh clone, ensure the required dataset/assets and model dependencies are available, regenerate the required outputs, and verify the dashboard at runtime. Do not describe a fresh clone as immediately demo-ready without this check.

## 7. Future-work architecture (do not claim implemented)

Proposed extensions may include a visual encoder in the main inference pipeline, learned cross-modal alignment/fusion, temporal or cross-image reasoning, conversation context/memory, speaker–listener graphs, cultural retrieval-augmented context, optional MLLM reasoning, separately supervised sarcasm/irony heads, validated target identification, and calibrated uncertainty.

Label these explicitly as **proposed future work**. Fixed 50:50 probability averaging is late fusion, not learned multimodal alignment. The current project has a binary humor classifier; sarcasm and irony need suitable labels and separate evaluation.

## 8. Key source files and evidence

- `src/mmhd_dashboard.py` — Streamlit dashboard.
- `src/predict_full_corpus_humor.py` — full-corpus prediction pipeline.
- `src/explain_full_corpus_humor.py` — model-decision explanation generation.
- `src/compare_multimodal_540.py` — grouped multimodal comparison and out-of-fold inference.
- `src/compare_late_fusion_540.py` — fixed equal-weight late-fusion evaluation from existing OOF probabilities.
- `results/dataset_validation.log` — dataset checks and exact-transcript warning.
- `results/expanded_540/multimodal_comparison/metrics.json` — complete-case counts and multimodal comparison results.
- `results/expanded_540/late_fusion/metrics.json` — primary metrics and limitations.
- `results/expanded_540/late_fusion/oof_predictions.csv` — 391 saved OOF predictions used for independent metric verification.
- `results/expanded_540/five_stream_comparison/metrics.json` — metrics and complete-case counts for the 11-configuration five-stream experiment.
- `results/expanded_540/five_stream_comparison/oof_predictions.csv` — sample-level out-of-fold predictions for all five-stream configurations.
- `results/full_corpus/humor_predictions_summary.json` — full-corpus prediction counts.
- `results/full_corpus/humor_structured_output_summary.json` — structured target/status counts.
- `results/full_corpus/humor_explanations_summary.json` — explanation evidence and uncertainty counts.

The focused syntax check below completed without errors on 18 Oct 2026. On 19 Oct 2026, `python -m compileall -q .\src` also completed successfully for all Python files under `src`.

```powershell
python -m py_compile `
    .\src\mmhd_dashboard.py `
    .\src\predict_full_corpus_humor.py `
    .\src\explain_full_corpus_humor.py `
    .\src\compare_multimodal_540.py `
    .\src\compare_late_fusion_540.py
```

The dashboard has also been launched in the project workflow and the structured analysis UI was observed rendering. Before the final live demo, launch it once again using the existing project instructions, for example `streamlit run .\src\mmhd_dashboard.py`, and confirm the expected local URL opens.

## 9. Report/PPT guardrails

1. State that the primary evaluation is **391 samples**, not all 540 annotations or all 7,289 corpus records.
2. Separate full-corpus inference statistics from evaluated out-of-fold performance.
3. Do not claim calibrated confidence, causal explanations, verified humor targets, or validated sarcasm/irony classification.
4. State that source-video grouping is used by the five-fold evaluation and that the inspected code checks train/test video overlap.
5. Describe the 7,289-record corpus as model-scored only where a prediction is available; 999 records have no model prediction.
6. Do not claim statistical significance for the small late-fusion gain over audio-only.
7. Distinguish implemented work from experimental results and proposed future work in diagrams and slides.
8. Cite actual papers and verify bibliographic details before submission. Do not claim a faithful reproduction of a research paper unless the implementation matches its method.

## 10. Suggested report outline

1. Abstract
2. Introduction and problem statement
3. Objectives and contributions
4. Related work
5. Dataset and preprocessing
6. Methodology and model configurations
7. Evaluation protocol and metrics
8. Results and discussion
9. Dashboard and structured evidence
10. Limitations and threats to validity
11. Conclusion and future work
12. References

## 11. Suggested PPT outline

1. Title and problem statement
2. Motivation and objectives
3. Dataset and validation
4. Related work and research gap
5. Implemented architecture
6. Evaluation methodology
7. Model comparison and results
8. Full-corpus inference and availability
9. Dashboard demonstration and evidence
10. Limitations
11. Proposed future architecture
12. Conclusion and questions

## 12. Handoff completion checklist

- [x] Explain the 540-to-391 primary complete-case evaluation selection and separate 343-sample five-stream subset accurately.
- [x] Explain the five-stream finding: audio-only leads macro F1 within that experiment; adding modalities does not consistently improve results with the current implementations.
- [x] Demonstrate both the primary model comparison and five-stream ablation/sample inspector in the dashboard.
- [x] Document the context-enrichment counts and limitations.
- [x] Document the ViT cross-frame and visual late-fusion experiments.
- [x] Document the cross-modal attention results and limitations.
- [x] Record the 19 October dashboard verification and full-source syntax check.
- [ ] Provide this handoff file and the repository/result artifacts to the juniors.
- [ ] Ensure report and slides use the verified numbers above.
- [x] Mark future modules as future work, not current implementation.
- [ ] Test the dashboard once immediately before the presentation.
- [ ] Review the final report/PPT for factual consistency only; no major new development after handoff.
