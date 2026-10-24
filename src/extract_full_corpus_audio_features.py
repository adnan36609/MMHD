import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict_full_corpus_humor import (  # noqa: E402
    AUDIO_FEATURES,
    load_corpus,
    resolve_asset,
)
from train_audio_baseline_540 import extract_audio_features  # noqa: E402

OUTPUT_DIR = ROOT / "results" / "full_corpus"
OUTPUT_FILE = OUTPUT_DIR / "audio_features.csv"
SUMMARY_FILE = OUTPUT_DIR / "audio_features_summary.json"

FIELDS = [
    "sample_id",
    "video_id",
    "status",
    *AUDIO_FEATURES,
    "error",
]


def load_existing_successes():
    existing = {}
    if not OUTPUT_FILE.exists():
        return existing

    with OUTPUT_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("sample_id"):
                existing[row["sample_id"]] = row

    return existing


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_corpus()
    ids = [str(r.get("sample_id") or "").strip() for r in records]

    if any(not sample_id for sample_id in ids):
        raise ValueError("Corpus contains a record with a missing sample_id.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs found in the full corpus.")

    existing = load_existing_successes()
    started = time.perf_counter()
    counts = Counter()

    print(f"Corpus records: {len(records)}")
    print(f"Successful features available to resume: {len(existing)}")
    print(f"Feature output: {OUTPUT_FILE}")

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for index, record in enumerate(records, start=1):
            sample_id = str(record["sample_id"]).strip()
            video_id = str(record.get("video_id") or "").strip()

            if sample_id in existing:
                row = existing[sample_id]
                counts["ok"] += 1
            else:
                row = {
                    "sample_id": sample_id,
                    "video_id": video_id,
                    "status": "",
                    **{name: "" for name in AUDIO_FEATURES},
                    "error": "",
                }

                raw_path = record.get("audio_path")
                audio_path = resolve_asset(raw_path) if raw_path else None

                if not audio_path or not Path(audio_path).is_file():
                    row["status"] = "missing_audio_path"
                    counts["missing_audio_path"] += 1
                else:
                    try:
                        values = extract_audio_features(audio_path)
                        vector = {
                            name: float(values[name])
                            for name in AUDIO_FEATURES
                        }

                        if not np.isfinite(list(vector.values())).all():
                            raise ValueError("Non-finite audio features.")

                        row.update(vector)
                        row["status"] = "ok"
                        counts["ok"] += 1

                    except Exception as exc:
                        row["status"] = "error"
                        row["error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )[:300]
                        counts["error"] += 1

            writer.writerow(row)

            # Flush frequently so completed work survives interruption.
            if index % 25 == 0 or index == len(records):
                f.flush()
                elapsed = time.perf_counter() - started
                print(
                    f"\rProcessed {index}/{len(records)} | "
                    f"successful={counts['ok']} | "
                    f"errors={counts['error']} | "
                    f"missing_path={counts['missing_audio_path']} | "
                    f"elapsed={elapsed:.1f}s",
                    end="\n" if index == len(records) else "",
                    flush=True,
                )

    summary = {
        "experiment": "Full-corpus audio feature extraction",
        "feature_source": "Existing extract_audio_features function",
        "corpus_records": len(records),
        "unique_sample_ids": len(set(ids)),
        "feature_columns": AUDIO_FEATURES,
        "status_counts": dict(counts),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
        "note": (
            "Features contain no human or pseudo-label targets. "
            "This extraction does not retrain the existing classifier."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n\n=== Extraction summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
