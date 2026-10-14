# MMHD Experimental Results

## 1. Project overview

MMHD combines a multimodal dataset-generation pipeline with exploratory experiments for binary humor recognition. The pipeline extracts five modalities: text, images, audio, optical character recognition (OCR), and transcript-derived emoji information. Dataset generation itself does not assign humor or hate-speech labels; the classification experiments are a separate research stage.

The current work includes a 7,289-sample corpus, dataset validation and modality auditing, a 240-sample annotated pilot, and preliminary classification experiments.

## 2. Dataset summary

| Measure | Result |
|---|---:|
| Extracted dataset samples | 7,289 |
| Candidates generated | 7,368 |
| Accepted samples | 7,289 |
| Diversity rejections | 79 |
| Samples with usable text | 5,674 (77.8%) |
| Samples with usable audio | 6,290 (86.3%) |
| Samples with usable visual images | 6,159 (84.5%) |
| Samples with usable OCR text | 3,865 (53.0%) |
| Samples with derived emoji content | 273 (3.7%) |

The current validation log reports 7,289 unique sample IDs, no missing audio files, no missing images, no invalid image counts, no candidate-boundary problems, and no temporal overlaps among the checked candidates.

The transcript-diversity audit identifies eight exact-duplicate groups involving 26 samples. This remains a data-quality limitation and must not be described as zero duplicate transcripts. Modality availability does not necessarily indicate that extracted content is semantically informative; OCR, for example, can capture channel branding or promotional overlays.

## 3. Annotation pilot

The completed initial pilot contains 240 labeled samples:

- Humorous: 82
- Non-humorous: 158

Most of the 7,289-sample corpus does not yet have a human-verified humor label. Therefore, supervised experiments use smaller annotated subsets according to their modality requirements.

A second annotation round has been prepared with 300 new samples and a 100-sample independent-review subset. Those annotations are pending. They must not be described as completed until both annotators return their files and the labels pass validation.

## 4. Text classification

The text experiment used 191 samples with usable transcripts from 24 source videos. Evaluation used five-fold GroupKFold grouped by source video.

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Majority baseline | 62.30% | 50.00% | 38.39% |
| TF-IDF + Logistic Regression | 53.93% | 49.04% | 48.69% |
| Word and character TF-IDF + Linear SVC | 55.50% | 50.30% | 49.87% |
| Target text + neighboring-scene text | 57.07% | 49.36% | 47.07% |

The text models did not exceed the majority baseline in accuracy. The neighboring-scene experiment used nearby scene transcripts as a proxy for context, not verified conversational turns. It did not improve balanced accuracy or macro F1 over the target-text baseline.

## 5. Audio classification

The audio baseline used 232 samples with usable audio, including 80 humorous and 152 non-humorous examples. It extracted handcrafted acoustic features and trained a Logistic Regression classifier.

| Model | Accuracy | Balanced accuracy | Macro F1 | Humorous F1 |
|---|---:|---:|---:|---:|
| Majority baseline | 65.52% | 50.00% | 39.58% | — |
| Handcrafted audio features + Logistic Regression | 57.76% | 57.40% | 56.18% | 47.87% |

The audio model improved balanced accuracy and macro F1 over the majority baseline, but its overall accuracy was lower. Feature-effect analysis describes associations learned by the model; it does not establish why a person finds a scene humorous.

## 6. Text, audio, and visual comparison

This experiment used 185 samples with text, audio, and visual features available, including 71 humorous and 114 non-humorous samples. All configurations used the same five-fold video-grouped evaluation.

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Majority baseline | 61.62% | 50.00% | 38.13% |
| Text-only | 51.35% | 46.98% | 46.63% |
| Audio-only | 57.84% | 58.09% | 57.18% |
| Visual-only | 46.49% | 45.42% | 45.23% |
| Text + audio | 56.76% | 56.41% | 55.82% |
| Text + visual | 49.19% | 46.82% | 46.83% |
| Audio + visual | 55.14% | 53.77% | 53.57% |
| Text + audio + visual | 54.05% | 51.83% | 51.80% |

Audio-only achieved the strongest balanced accuracy and macro F1 in this comparison. None of the configurations exceeded the majority baseline in accuracy. Simple multimodal fusion did not consistently improve upon audio-only.

## 7. OCR and five-stream experiments

The OCR comparison used 145 OCR-eligible samples from 24 source videos, including 48 humorous and 97 non-humorous samples.

| Model | Accuracy | Balanced accuracy | Macro F1 | Humorous F1 |
|---|---:|---:|---:|---:|
| Majority baseline | 66.90% | 50.00% | 40.08% | — |
| Text-only | 57.93% | 51.19% | 51.16% | 32.97% |
| OCR-only | 67.59% | 58.93% | 59.12% | 40.51% |
| Text + OCR | 66.21% | 57.90% | 58.03% | 39.51% |

The OCR-only model had the strongest scores in this subset, but its accuracy exceeded the majority baseline by only 0.69 percentage points.

### OCR de-branding sensitivity

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Raw OCR-only | 67.59% | 58.93% | 59.12% |
| De-branded OCR-only | 54.48% | 43.88% | 42.13% |
| Text + raw OCR | 66.21% | 57.90% | 58.03% |
| Text + de-branded OCR | 53.79% | 43.89% | 42.61% |

The cleaning rules removed manually identified branding and promotional phrases. Fifteen samples had no OCR text remaining after cleaning. The decline indicates substantial sensitivity to the removed content. It does not prove branding alone explains the original performance, because the heuristic may also have removed useful words.

The five-stream comparison used the same 145-sample subset. Emoji content was present for only nine samples in that experiment, so the result does not establish the benefit of emoji as a supervised modality.

## 8. Evaluation protocol and limitations

The experiments used five-fold GroupKFold evaluation grouped by source video to avoid placing samples from the same source video in both training and validation folds.

The evaluation subsets differ in size and composition. Their scores should not be combined into one ranking or directly compared as though they measured performance on identical data.

Key limitations include:

- The human-labeled pilot is small and class-imbalanced.
- Some transcripts are noisy, empty, or incomplete.
- OCR can capture branding and promotional overlays instead of humor-related content.
- Simple multimodal fusion did not consistently outperform individual modalities.
- The neighboring-scene context proxy is not verified dialogue context.
- Emoji coverage is sparse.
- The current classification experiments address binary humor recognition, not a completed hate-speech classifier.
- The current work has not reproduced the complete architectures of the reviewed research papers.

## 9. Relationship to prior research

Three research directions informed the project design:

- **M2H2: A Multimodal Multiparty Hindi Dataset for Humor Recognition in Conversations.** This work motivates multimodal humor analysis and the importance of conversational context. The neighboring-scene experiment is only a context proxy and does not reproduce M2H2's conversational models.
- **Detecting Emotional Incongruity of Sarcasm by Commonsense Reasoning (EICR).** This paper motivates future work on incongruity and commonsense reasoning. EICR's reasoning mechanisms were not implemented in the current baselines.
- **CFMS: Towards Explainable and Fine-Grained Chinese Multimodal Sarcasm Detection Benchmark.** This work motivates fine-grained annotation and explainability. The current project does not reproduce its Chinese benchmark or reinforcement-learning-based exemplar-selection method.

Full bibliographic details and in-text citations should be verified and added to the final research report.

## 10. Conclusion and next steps

The dataset-generation pipeline produces a structured corpus of 7,289 multimodal samples. Initial classification experiments suggest that acoustic and OCR-derived features contain exploratory predictive signal, but the models do not establish reliable humor understanding, and multimodal fusion does not consistently improve performance.

The next evaluation stage is to validate the expanded annotation round, assess agreement on the 100 overlapping samples, and rerun the core comparisons with the consolidated labels. Conclusions should be updated if the expanded experiments materially change these findings.
