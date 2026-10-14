# MMHD Pilot Annotation Guidelines

## 1. Annotation Task

Assign exactly one binary label to each candidate sample:

- `humorous`
- `non_humorous`

The task is to determine whether the candidate itself contains a recognizable intentional comedic element.

The fact that a sample comes from a comedy-oriented source does not by itself make the sample humorous.

---

## 2. Label: humorous

Assign `humorous` when the candidate contains a recognizable comedic element that a reasonable viewer can identify from the available multimodal evidence.

Possible comedic elements include:

- A joke or punchline
- A deliberately funny statement
- A humorous reaction
- Comic timing or an unexpected comedic action
- A visual gag
- A funny interaction
- An intentionally absurd or exaggerated situation
- A comedic use of language, image, or sound

The humor should be attributable to the candidate itself, rather than requiring extensive information from unrelated parts of the source video.

---

## 3. Label: non_humorous

Assign `non_humorous` when the candidate does not contain a recognizable comedic element.

Examples include:

- Ordinary conversation
- Exposition or narration
- Scene transitions
- Setup without a recognizable comedic payoff
- Neutral reactions
- Ordinary physical actions
- Background material
- Serious or informational content
- A scene that is from a comedy source but is not itself humorous

---

## 4. Available Evidence

Use the evidence contained in the candidate:

- Spoken/transcribed text
- Audio
- Representative image(s)

Consider these modalities together when available.

A humorous sample does not need to contain humor in every modality. For example, a visual gag may be humorous even if the spoken text is neutral.

---

## 5. Do Not Infer Humor From Metadata

Do not use the following as evidence for the label:

- `candidate_type`
- Scene number
- Sample ID
- Video filename
- Source video title
- The assumption that the source is a comedy video
- The position of the sample in the video
- Any existing dataset statistics

Metadata may be used to locate a sample, but not to determine its humor label.

---

## 6. Ambiguous Cases

When the evidence is genuinely ambiguous, assign the label based on the strongest interpretation supported by the candidate itself.

Do not assume that a sample is humorous simply because it could become humorous when combined with a large amount of surrounding context.

Do not assign `humorous` merely because something is unusual, surprising, emotional, or interesting.

The decision should be based on recognizable comedic intent or comedic effect.

---

## 7. Context

The candidate should be judged primarily as an individual sample.

Do not inspect unrelated parts of the source video to find a reason to label the candidate as humorous.

If minimal immediate context is necessary to understand the candidate, that context may be considered. However, the humor should still be reasonably attributable to the candidate.

---

## 8. Required Labels

Only these two values are valid:

`humorous`

`non_humorous`

Do not use:

- `maybe`
- `uncertain`
- `partially_humorous`
- `sarcastic`
- `ironic`
- `funny_setup`
- Any other label

---

## 9. Annotation Scope

This pilot annotates only binary humor.

Do not assign labels for:

- Sarcasm
- Irony
- Emotion
- Humor target
- Explanation
- Cultural reference
- Commonsense reasoning
- Speaker identity
- Audience reaction

These attributes may be studied in later phases but are outside the current annotation task.

---

## 10. Quality Principle

The goal is consistent annotation, not maximizing the number of `humorous` samples.

Apply the same criteria to every source video.

A candidate should receive the same label regardless of which source video it came from.
