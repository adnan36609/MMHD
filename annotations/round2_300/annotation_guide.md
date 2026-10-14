# MMHD Round 2 Annotation Guide

## Task
Assign one binary label to each specific sample:
- `humorous`
- `non_humorous`

Each sample pack contains the extracted audio and two representative keyframes,
plus transcript, OCR, and a heuristic emoji field. The original candidate MP4
files were not retained in this repository, so do not assume the two images show
every moment of the segment.

## Recommended annotation procedure
1. Listen to the provided audio and inspect both images before reading transcript/OCR.
2. Decide whether the sampled segment itself contains clear humor.
3. Use the transcript and OCR as supporting context only. Speech recognition may
   be inaccurate, and OCR may mostly capture show names, logos, or promotions.
4. Label the sample `humorous` or `non_humorous`.
5. Record confidence as `high`, `medium`, or `low`.
6. If genuinely ambiguous, choose the best label, set `needs_adjudication` to `yes`,
   and briefly note why.

## Label definitions
**Humorous:** The supplied segment clearly presents or attempts humor, such as a
joke, punchline, wordplay, deliberate comic exaggeration, absurdity, slapstick,
or sarcasm/irony clearly used for comic effect. Humor may be carried by speech,
audio delivery, visuals, or their combination.

**Non-humorous:** The supplied segment does not clearly present humor. Examples
include ordinary conversation, greetings, factual explanation, serious/emotional
content without clear comic intent, or content consisting mainly of branding,
channel names, or subscription prompts.

## Consistency rules
- Do not label a segment humorous merely because it comes from a comedy show or comedian.
- Audience laughter, a smile, or an emoji alone is not sufficient evidence.
- On-screen branding is not itself humor; do not infer a label from show/channel names.
- Transcript and OCR may be noisy or incomplete. Consider the actual audio and images first.
- The `derived_emoji` value is heuristically inferred from transcript words; it is not
  visual emoji detection and must not determine the label by itself.
- Judge only the supplied assets. Do not search for the source video or use outside knowledge.
- Keep sample IDs unchanged; do not add or delete samples.

## Fields to complete
- `humor_label`: exactly `humorous` or `non_humorous`
- `confidence`: `high`, `medium`, or `low`
- `needs_adjudication`: `yes` or `no`
- `annotator_notes`: short reason where useful, especially for ambiguity

The second annotator's 100-sample pack overlaps with the primary batch. They must
label independently and must not see the primary annotator's labels first.
