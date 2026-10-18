import csv
import json
import re
import time
from pathlib import Path

import spacy


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results" / "full_corpus" / "humor_explanations.csv"
)

OUTPUT_DIR = ROOT / "results" / "full_corpus"
OUTPUT_FILE = OUTPUT_DIR / "humor_structured_output.csv"
SUMMARY_FILE = OUTPUT_DIR / "humor_structured_output_summary.json"

SPACY_MODEL = "en_core_web_sm"
BATCH_SIZE = 64

TARGET_CUES = re.compile(
    r"\b(?:"
    r"making fun of|make fun of|made fun of|makes fun of|"
    r"poking fun at|"
    r"laughing at|laughed at|laugh at|"
    r"joking about|joked about|joke about|jokes about|"
    r"making jokes about|"
    r"at the expense of|"
    r"ridiculing|ridiculed|ridicule|"
    r"mocking|mocked|mock|"
    r"roasting|roasted|roast|"
    r"teasing|teased|tease"
    r")\s+(?P<target>[^.!?;\n]{1,100})",
    flags=re.IGNORECASE,
)

TARGET_CATEGORIES = {
    "PERSON",
    "NORP",
    "ORG",
    "GPE",
    "FAC",
    "PRODUCT",
    "WORK_OF_ART",
    "EVENT",
}

CUT_TARGET_AT = re.compile(
    r"\b(?:because|although|while|when|but|"
    r"and then|which|who|that|so that)\b",
    flags=re.IGNORECASE,
)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def clean_candidate(value):
    value = " ".join(value.split())

    # Avoid including the rest of a sentence as the target.
    value = CUT_TARGET_AT.split(value, maxsplit=1)[0]

    value = value.strip(" \t\r\n,.:;!?\"'`()[]{}")

    # Keep candidates short and suitable for display.
    if len(value) > 80:
        value = value[:80].rsplit(" ", 1)[0]

    return value


def get_explicit_target(text):
    """
    Extract the phrase following an explicit target-related cue.

    This is a heuristic. A matching phrase is a candidate,
    not a human-verified target annotation.
    """
    matches = []

    for match in TARGET_CUES.finditer(text):
        candidate = clean_candidate(match.group("target"))

        if candidate:
            matches.append({
                "candidate": candidate,
                "cue": match.group(0)[:120],
                "method": "explicit_lexical_cue",
            })

    # Preserve order while removing exact duplicate candidates.
    unique = []
    seen = set()

    for match in matches:
        key = match["candidate"].casefold()

        if key not in seen:
            seen.add(key)
            unique.append(match)

    return unique


def extract_entities(doc):
    entities = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ not in TARGET_CATEGORIES:
            continue

        text = " ".join(ent.text.split())

        if not text:
            continue

        key = (text.casefold(), ent.label_)

        if key in seen:
            continue

        seen.add(key)

        entities.append({
            "text": text,
            "type": ent.label_,
        })

    return entities


def classify_target(row, doc):
    """
    Assign a conservative target status.

    Named entities alone do not establish who is being mocked
    or laughed at. The system therefore leaves the target unknown
    unless a target-related linguistic cue is found.
    """
    entities = extract_entities(doc)

    probability_text = str(
        row.get("humor_probability") or ""
    ).strip()

    if not probability_text:
        return {
            "target_of_humor": "unknown",
            "target_status": "no_humor_prediction",
            "target_candidates": [],
            "context_entities": entities,
            "target_evidence": [],
        }

    probability = float(probability_text)

    if not 0 <= probability <= 1:
        raise ValueError(
            f"Invalid humor score for {row.get('sample_id')}: "
            f"{probability}"
        )

    prediction = str(row.get("humor_prediction") or "")

    if prediction == "unavailable":
        status = "no_humor_prediction"

    elif 0.45 <= probability <= 0.55:
        status = "uncertain_humor_prediction"

    elif prediction != "humorous":
        status = "not_applicable_predicted_non_humorous"

    else:
        explicit_candidates = get_explicit_target(
            str(row.get("text") or "")
        )

        if explicit_candidates:
            status = "candidate_found_needs_validation"

            return {
                "target_of_humor": "unknown",
                "target_status": status,
                "target_candidates": explicit_candidates,
                "context_entities": entities,
                "target_evidence": [
                    {
                        "type": "linguistic_cue",
                        "text": explicit_candidates[0]["cue"],
                    }
                ],
            }

        status = "unknown_insufficient_target_evidence"

    # Entity mentions are context only. Do not promote them to targets.
    return {
        "target_of_humor": "unknown",
        "target_status": status,
        "target_candidates": [],
        "context_entities": entities,
        "target_evidence": [],
    }


def build_structured_output(row, target):
    humor_probability = (
        float(row["humor_probability"])
        if str(row.get("humor_probability") or "").strip()
        else None
    )

    decision_strength = (
        float(row["decision_strength_score"])
        if str(row.get("decision_strength_score") or "").strip()
        else None
    )

    uncertainty_flags = json.loads(
        row.get("uncertainty_flags") or "[]"
    )

    text_supporting = json.loads(
        row.get("text_supporting_terms") or "[]"
    )

    text_opposing = json.loads(
        row.get("text_opposing_terms") or "[]"
    )

    audio_supporting = json.loads(
        row.get("audio_supporting_features") or "[]"
    )

    audio_opposing = json.loads(
        row.get("audio_opposing_features") or "[]"
    )

    return {
        "sample_id": row["sample_id"],
        "prediction": row.get(
            "humor_prediction", "unavailable"
        ),
        "humor_score": humor_probability,
        "confidence": {
            "level": row.get(
                "confidence_level_heuristic", "unavailable"
            ),
            "decision_strength": decision_strength,
            "calibrated": False,
            "interpretation": (
                "Decision-margin heuristic only; not calibrated "
                "probability of correctness."
            ),
        },
        "target_of_humor": {
            "candidate": target["target_of_humor"],
            "status": target["target_status"],
            "candidates": target["target_candidates"],
            "context_entities": target["context_entities"],
            "evidence": target["target_evidence"],
        },
        "explanation": {
            "text": row.get("explanation", ""),
            "text_supporting_terms": text_supporting,
            "text_opposing_terms": text_opposing,
            "audio_supporting_features": audio_supporting,
            "audio_opposing_features": audio_opposing,
            "uncertainty_flags": uncertainty_flags,
            "scope": (
                "Evidence from existing text/audio classifiers; "
                "not a causal explanation."
            ),
        },
        "annotation_status": (
            "human_annotated"
            if row.get("human_label")
            not in ("", "not_annotated", "unknown", None)
            else "unlabelled"
        ),
        "evaluation_status": row.get(
            "interpretation_status", "unknown"
        ),
    }


def main():
    start_time = time.perf_counter()

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(INPUT_FILE)

    rows = read_csv(INPUT_FILE)

    if not rows:
        raise ValueError("The explanation input file is empty.")

    print(f"Input records: {len(rows):,}")
    print(f"Loading spaCy model: {SPACY_MODEL}")

    nlp = spacy.load(SPACY_MODEL)

    texts = [
        " ".join(str(row.get("text") or "").split())
        for row in rows
    ]

    entity_counts = {}
    target_status_counts = {}
    structured_rows = []

    print("Extracting entities and candidate targets...")

    for start in range(0, len(rows), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(rows))

        docs = nlp.pipe(
            texts[start:end],
            batch_size=BATCH_SIZE,
        )

        for row, doc in zip(rows[start:end], docs):
            target = classify_target(row, doc)

            for entity in target["context_entities"]:
                key = entity["type"]
                entity_counts[key] = entity_counts.get(key, 0) + 1

            key = target["target_status"]
            target_status_counts[key] = (
                target_status_counts.get(key, 0) + 1
            )

            structured = build_structured_output(row, target)

            output_row = dict(row)
            output_row.update({
                "target_of_humor": target["target_of_humor"],
                "target_status": target["target_status"],
                "target_candidates": json.dumps(
                    target["target_candidates"],
                    ensure_ascii=False,
                ),
                "context_entities": json.dumps(
                    target["context_entities"],
                    ensure_ascii=False,
                ),
                "target_evidence": json.dumps(
                    target["target_evidence"],
                    ensure_ascii=False,
                ),
                "structured_output_json": json.dumps(
                    structured,
                    ensure_ascii=False,
                ),
            })

            structured_rows.append(output_row)

        processed = end

        if processed % 500 < BATCH_SIZE or processed == len(rows):
            print(f"Processed: {processed:,}/{len(rows):,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = list(structured_rows[0].keys())

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(structured_rows)

    summary = {
        "records_processed": len(structured_rows),
        "target_status_counts": target_status_counts,
        "context_entity_counts": entity_counts,
        "records_with_candidate_target": sum(
            bool(json.loads(row["target_candidates"]))
            for row in structured_rows
        ),
        "records_with_humor_prediction": sum(
            row.get("humor_prediction")
            in ("humorous", "non_humorous")
            for row in structured_rows
        ),
        "target_method": (
            "Conservative lexical-cue heuristic plus spaCy named-entity "
            "recognition. Candidate targets are not independently verified."
        ),
        "confidence_note": (
            "The existing decision-strength score is not calibrated."
        ),
        "output": str(OUTPUT_FILE.relative_to(ROOT)),
        "runtime_seconds": round(
            time.perf_counter() - start_time,
            2,
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== TARGET IDENTIFICATION SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()