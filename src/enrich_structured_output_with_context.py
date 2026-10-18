"""Add neighboring-scene context to MMHD structured output.

Preserves all existing predictions and explanations.
Context is descriptive evidence, not a new humor prediction.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from compare_context_text_540 import (  # noqa: E402
    find_neighbor_context,
    load_video_scene_maps,
    scene_number,
)

INPUT_FILE = ROOT / "results" / "full_corpus" / "humor_structured_output.csv"

OUTPUT_DIR = ROOT / "results" / "full_corpus"
OUTPUT_FILE = OUTPUT_DIR / "humor_context_enriched.csv"
SUMMARY_FILE = OUTPUT_DIR / "humor_context_enriched_summary.json"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main():
    rows = read_csv(INPUT_FILE)

    if not rows:
        raise ValueError(f"No records found in {INPUT_FILE}")

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in structured output.")

    video_ids = sorted({
        row["video_id"]
        for row in rows
        if row.get("video_id")
    })

    print(f"Structured records: {len(rows)}")
    print(f"Source videos: {len(video_ids)}")
    print("Loading per-video scene datasets...")

    scene_maps, videos_loaded, missing_video_files = (
        load_video_scene_maps(video_ids)
    )

    context_counts = Counter()
    enriched_rows = []

    for row in rows:
        video_id = row.get("video_id", "")
        target_number = scene_number(row.get("source_scene"))

        context = find_neighbor_context(
            scene_maps.get(video_id, {}),
            target_number,
        )

        previous_text = context.get("previous_text", "")
        next_text = context.get("next_text", "")

        if previous_text and next_text:
            status = "both_available"
        elif previous_text:
            status = "previous_only"
        elif next_text:
            status = "next_only"
        else:
            status = "no_neighbor_text"

        context_counts[status] += 1

        enriched = dict(row)
        enriched.update({
            "previous_scene_id": context.get("previous_scene_id", ""),
            "previous_scene_text": previous_text,
            "next_scene_id": context.get("next_scene_id", ""),
            "next_scene_text": next_text,
            "context_availability": status,
            "context_source": (
                "same_video_scene_order"
                if status != "no_neighbor_text"
                else "unavailable"
            ),
            "context_is_verified_dialogue": "false",
        })

        # Explicitly document the context's epistemic status.
        enriched["context_note"] = (
            "Nearest available scene-order text; "
            "not verified conversational turns."
            if status != "no_neighbor_text"
            else "No usable neighboring-scene text found."
        )

        enriched_rows.append(enriched)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep all original columns and append context fields.
    fieldnames = list(enriched_rows[0].keys())

    with OUTPUT_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)

    summary = {
        "records": len(enriched_rows),
        "unique_sample_ids": len({
            row["sample_id"] for row in enriched_rows
        }),
        "source_videos": len(video_ids),
        "video_dataset_files_loaded": videos_loaded,
        "videos_missing_dataset_file": missing_video_files,
        "context_availability_counts": dict(context_counts),
        "context_definition": (
            "Nearest available preceding/following scene text from the "
            "same video dataset. This is a scene-order proxy, not "
            "verified conversation context."
        ),
        "prediction_policy": (
            "Original humor predictions, probabilities, labels and "
            "explanations are preserved. Context does not modify them."
        ),
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nCONTEXT ENRICHMENT SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()