"""Prepare MMHD annotation packages from extracted audio + keyframes.

Run from the MMHD repository:
    python .\\src\\prepare_annotation_batch.py

The corpus does not retain the individual candidate MP4 files. This script
therefore packages each sample's extracted WAV and two representative images,
plus transcript/OCR/derived-emoji context, for human annotation. It does not
modify dataset.json files, media assets, or existing annotation files.
"""

import csv
import html
import json
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "sample_clips"
EXISTING_ANNOTATIONS = (
    ROOT / "annotations" / "pilot" / "pilot_annotations_second_completed.json"
)
OUT_DIR = ROOT / "annotations" / "round2_300"
BATCH_SIZE = 300
DOUBLE_ANNOTATION_SIZE = 100
RANDOM_SEED = 20261015

CSV_FIELDS = [
    "sample_id", "video_id", "source_scene", "duration_seconds",
    "media_folder", "transcript", "ocr_text", "derived_emoji",
    "humor_label", "confidence", "needs_adjudication", "annotator_notes",
]

GUIDE = """# MMHD Round 2 Annotation Guide

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
"""


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def text_value(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item is not None).strip()
    return str(value).strip()


def resolve_asset(raw_value, video_dir):
    """Resolve a JSON media path without requiring candidate MP4s."""
    raw = text_value(raw_value)
    if not raw:
        return None

    normalized = raw.replace("\\", "/")
    candidates = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            ROOT / p,
            video_dir / p,
            video_dir / p.name,
            ROOT / "sample_clips" / p,
            ROOT / "sample_clips" / video_dir.name / p.name,
        ])
    # Some Windows-generated records may contain a leading ./.
    candidates.append(ROOT / normalized.lstrip("./"))

    seen = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve()).casefold()
        except OSError:
            key = str(candidate).casefold()
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def inspect_record(record, video_dir):
    modalities = record.get("modalities") or {}
    text_obj = modalities.get("text") or {}
    audio_obj = modalities.get("audio") or {}
    image_obj = modalities.get("image") or {}
    ocr_obj = modalities.get("ocr") or {}
    emoji_obj = modalities.get("emoji") or {}

    transcript = text_value(text_obj.get("value"))
    ocr = text_value(ocr_obj.get("value"))
    emoji = text_value(emoji_obj.get("value"))

    audio_path = resolve_asset(audio_obj.get("path"), video_dir)
    image_paths = [
        resolve_asset(p, video_dir) for p in (image_obj.get("paths") or [])
    ]
    image_paths = [p for p in image_paths if p is not None]

    # Human annotation uses extracted audio and two keyframes. Require both so
    # all selected samples can be reviewed in a consistent modality format.
    if audio_path is None or len(image_paths) < 2:
        return None

    source = record.get("source") or {}
    timing = record.get("timing") or {}
    try:
        duration = float(timing.get("duration_seconds", ""))
    except (TypeError, ValueError):
        duration = ""

    flags = {
        "text": bool(transcript),
        "audio": True,
        "visual": True,
        "ocr": bool(ocr),
        "emoji": bool(emoji),
    }
    # Concise pattern used to stratify selection; T=text, A=audio, V=visual,
    # O=OCR, E=derived emoji. A+V are guaranteed by eligibility.
    pattern = "+".join(
        letter for key, letter in [
            ("text", "T"), ("audio", "A"), ("visual", "V"),
            ("ocr", "O"), ("emoji", "E"),
        ] if flags[key]
    )

    return {
        "sample_id": text_value(record.get("sample_id")),
        "video_id": text_value(source.get("video_id")) or video_dir.name,
        "source_scene": text_value(source.get("source_scene")),
        "duration_seconds": duration,
        "audio_source": audio_path,
        "image_sources": image_paths[:2],
        "transcript": transcript,
        "ocr_text": ocr,
        "derived_emoji": emoji,
        "modalities": flags,
        "modality_pattern": pattern,
    }


def balanced_quotas(capacities, total, rng):
    if total > sum(capacities.values()):
        raise ValueError(f"Cannot allocate {total}; capacity is {sum(capacities.values())}.")
    quotas = {key: 0 for key in capacities}
    remaining = total
    while remaining:
        eligible = [k for k, cap in capacities.items() if quotas[k] < cap]
        if not eligible:
            raise RuntimeError("Selection quota allocation exhausted capacity.")
        lowest = min(quotas[k] for k in eligible)
        tied = [k for k in eligible if quotas[k] == lowest]
        rng.shuffle(tied)
        for key in tied:
            if remaining == 0:
                break
            if quotas[key] < capacities[key]:
                quotas[key] += 1
                remaining -= 1
    return quotas


def select_diverse(records, count, rng):
    by_video = defaultdict(list)
    for row in records:
        by_video[row["video_id"]].append(row)
    video_quotas = balanced_quotas(
        {video: len(rows) for video, rows in by_video.items()}, count, rng
    )

    selected = []
    for video_id, quota in video_quotas.items():
        by_pattern = defaultdict(list)
        for row in by_video[video_id]:
            by_pattern[row["modality_pattern"]].append(row)
        quotas = balanced_quotas(
            {pattern: len(items) for pattern, items in by_pattern.items()},
            quota,
            rng,
        )
        for pattern, n in quotas.items():
            if n:
                selected.extend(rng.sample(by_pattern[pattern], n))

    if len(selected) != count or len({r["sample_id"] for r in selected}) != count:
        raise RuntimeError("Sample selection did not produce unique requested IDs.")
    rng.shuffle(selected)
    return selected


def safe_folder(sample_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def csv_row(row):
    return {
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "source_scene": row["source_scene"],
        "duration_seconds": row["duration_seconds"],
        "media_folder": f"media/{safe_folder(row['sample_id'])}",
        "transcript": row["transcript"],
        "ocr_text": row["ocr_text"],
        "derived_emoji": row["derived_emoji"],
        "humor_label": "",
        "confidence": "",
        "needs_adjudication": "",
        "annotator_notes": "",
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summary_for(rows, batch_name):
    return {
        "batch": batch_name,
        "sample_count": len(rows),
        "source_video_count": len({r["video_id"] for r in rows}),
        "samples_per_video": dict(sorted(Counter(r["video_id"] for r in rows).items())),
        "modality_pattern_counts": dict(sorted(Counter(r["modality_pattern"] for r in rows).items())),
        "modality_availability_counts": {
            key: sum(r["modalities"][key] for r in rows)
            for key in ["text", "audio", "visual", "ocr", "emoji"]
        },
        "selection_method": (
            "Selected without using humor labels; approximately balanced across source videos, "
            "then spread across available modality patterns. This is not a prevalence-proportional sample."
        ),
        "random_seed": RANDOM_SEED,
    }


def build_viewer_html(rows, batch_key, heading):
    """Create an offline HTML annotator. Assets remain relative to this HTML file."""
    payload = []
    for row in rows:
        folder = f"media/{safe_folder(row['sample_id'])}"
        payload.append({
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "source_scene": row["source_scene"],
            "duration_seconds": row["duration_seconds"],
            "transcript": row["transcript"],
            "ocr_text": row["ocr_text"],
            "derived_emoji": row["derived_emoji"],
            "image1": f"{folder}/image_1.jpg",
            "image2": f"{folder}/image_2.jpg",
            "audio": f"{folder}/audio.wav",
        })
    records_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMHD Annotation</title>
<style>
:root{color-scheme:light;font-family:Segoe UI,Arial,sans-serif;color:#17202a;background:#f3f5f7}body{margin:0}header{background:#172b4d;color:white;padding:16px 22px}main{max-width:1100px;margin:18px auto;padding:0 14px}.bar,.card{background:white;border:1px solid #d9e0e7;border-radius:10px;padding:14px;margin-bottom:14px}.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.muted{color:#52606d;font-size:14px}.progress{font-weight:600}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}.images img{width:100%;max-height:300px;object-fit:contain;background:#101820;border-radius:6px}.audio{width:100%;margin:8px 0}.meta{display:grid;grid-template-columns:1fr 1fr;gap:12px}.field{margin:10px 0}label{font-weight:600;display:block;margin-bottom:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:#f6f8fa;padding:10px;border-radius:6px;min-height:30px}.buttons{display:flex;gap:8px;flex-wrap:wrap}.buttons button,button{padding:9px 13px;border:1px solid #aab6c2;border-radius:6px;background:white;cursor:pointer;font-weight:600}.buttons button.chosen{background:#d9ecff;border-color:#1769aa}.actions{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.primary{background:#1769aa;color:white;border-color:#1769aa}textarea,select{width:100%;padding:9px;border:1px solid #abb7c4;border-radius:6px;box-sizing:border-box}textarea{min-height:56px}.warn{color:#8a4b00}.saved{color:#187744}@media(max-width:680px){.images,.meta{grid-template-columns:1fr}.images img{max-height:240px}}
</style></head><body>
<header><h1>MMHD — __HEADING__</h1><div>Listen to the audio and inspect both keyframes before reading transcript/OCR.</div></header>
<main><div class="bar"><button id="prev">← Previous</button><button id="next">Next →</button><span class="progress" id="progress"></span><span class="muted" id="saved"></span><button id="export" class="primary">Export completed annotations CSV</button></div>
<div class="card"><div class="muted" id="identifier"></div><div class="field"><label>Audio</label><audio id="audio" controls class="audio"></audio></div><div class="images"><div><label>Keyframe 1</label><img id="image1" alt="First representative frame"></div><div><label>Keyframe 2</label><img id="image2" alt="Second representative frame"></div></div></div>
<div class="card meta"><div><label>Transcript (may be noisy)</label><pre id="transcript"></pre></div><div><label>OCR text (may contain branding)</label><pre id="ocr"></pre><label>Derived emoji (heuristic, not visual detection)</label><pre id="emoji"></pre></div></div>
<div class="card"><div class="field"><label>Humor label</label><div class="buttons"><button data-label="humorous">Humorous</button><button data-label="non_humorous">Non-humorous</button></div></div>
<div class="field"><label for="confidence">Confidence</label><select id="confidence"><option value="">Choose confidence</option><option>high</option><option>medium</option><option>low</option></select></div>
<div class="field"><label for="adjudication">Needs adjudication?</label><select id="adjudication"><option value="">Choose</option><option>no</option><option>yes</option></select></div>
<div class="field"><label for="notes">Notes (optional; recommended for ambiguous samples)</label><textarea id="notes"></textarea></div>
<div class="actions"><button id="save" class="primary">Save this sample</button><button id="clear">Clear current label</button><span class="muted" id="status"></span></div></div>
<div class="card"><b>Reminder</b><p>Do not infer humor from the show name, channel branding, laughter, or derived emoji alone. Judge the specific sample using its audio and the two images.</p><p class="muted">Progress is saved in this browser's local storage. Export the CSV before closing. The browser does not alter the original dataset files.</p></div>
</main>
<script>
const records = __RECORDS_JSON__;
const storageKey = 'mmhd_annotation___BATCH_KEY__';
let index = 0;
let answers = {};
try { answers = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch(e) { answers = {}; }
const $ = id => document.getElementById(id);
function updateProgress(){const done=records.filter(r=>answers[r.sample_id]&&answers[r.sample_id].humor_label).length;$('progress').textContent=`Sample ${index+1} / ${records.length} · ${done} labeled`;$('saved').textContent='Browser-saved';}
function persist(){localStorage.setItem(storageKey,JSON.stringify(answers));updateProgress();}
function render(){const r=records[index];$('identifier').textContent=`${r.sample_id} · Source video: ${r.video_id} · Scene: ${r.source_scene} · Duration: ${r.duration_seconds || 'unknown'} s`;$('audio').src=r.audio;$('image1').src=r.image1;$('image2').src=r.image2;$('transcript').textContent=r.transcript||'(No usable transcript extracted)';$('ocr').textContent=r.ocr_text||'(No readable OCR text extracted)';$('emoji').textContent=r.derived_emoji||'(No derived emoji value)';const a=answers[r.sample_id]||{};document.querySelectorAll('[data-label]').forEach(b=>b.classList.toggle('chosen',b.dataset.label===a.humor_label));$('confidence').value=a.confidence||'';$('adjudication').value=a.needs_adjudication||'';$('notes').value=a.annotator_notes||'';$('status').textContent='';updateProgress();}
function save(){const r=records[index];const a={humor_label:document.querySelector('[data-label].chosen')?.dataset.label||'',confidence:$('confidence').value,needs_adjudication:$('adjudication').value,annotator_notes:$('notes').value};if(!a.humor_label){$('status').textContent='Choose a humor label first.';return;}if(!a.confidence||!a.needs_adjudication){$('status').textContent='Choose confidence and adjudication status.';return;}answers[r.sample_id]=a;persist();$('status').textContent='Saved.';}
function clearCurrent(){delete answers[records[index].sample_id];persist();render();}
function csvCell(v){return '"'+String(v??'').replace(/"/g,'""')+'"';}
function exportCsv(){const headers=['sample_id','video_id','source_scene','duration_seconds','media_folder','transcript','ocr_text','derived_emoji','humor_label','confidence','needs_adjudication','annotator_notes'];const lines=[headers.map(csvCell).join(',')];for(const r of records){const a=answers[r.sample_id]||{};const folder=r.audio.substring(0,r.audio.lastIndexOf('/'));const row={...r,media_folder:folder,...a};lines.push(headers.map(h=>csvCell(row[h]??'')).join(','));}const blob=new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='annotations_completed.csv';document.body.appendChild(link);link.click();link.remove();URL.revokeObjectURL(url);}
document.querySelectorAll('[data-label]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-label]').forEach(x=>x.classList.remove('chosen'));b.classList.add('chosen');}));$('save').addEventListener('click',save);$('clear').addEventListener('click',clearCurrent);$('prev').addEventListener('click',()=>{if(index>0){index--;render();}});$('next').addEventListener('click',()=>{if(index<records.length-1){index++;render();}});$('export').addEventListener('click',exportCsv);render();
</script></body></html>'''
    return (template.replace("__RECORDS_JSON__", records_json)
            .replace("__BATCH_KEY__", batch_key)
            .replace("__HEADING__", html.escape(heading)))


def copy_sample_assets(row, media_root):
    folder = media_root / safe_folder(row["sample_id"])
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(row["audio_source"], folder / "audio.wav")
    shutil.copy2(row["image_sources"][0], folder / "image_1.jpg")
    shutil.copy2(row["image_sources"][1], folder / "image_2.jpg")


def build_package(zip_path, rows, csv_path, summary_path, batch_key, heading):
    with tempfile.TemporaryDirectory(prefix="mmhd_annotation_pack_") as tmp:
        temp_root = Path(tmp)
        media_root = temp_root / "media"
        media_root.mkdir()
        for count, row in enumerate(rows, start=1):
            copy_sample_assets(row, media_root)
            if count % 25 == 0 or count == len(rows):
                print(f"  Pack assets: {count}/{len(rows)}")
        shutil.copy2(csv_path, temp_root / csv_path.name)
        (temp_root / "annotation_guide.md").write_text(GUIDE, encoding="utf-8")
        shutil.copy2(summary_path, temp_root / "batch_summary.json")
        viewer = build_viewer_html(rows, batch_key, heading)
        (temp_root / "annotation_viewer.html").write_text(viewer, encoding="utf-8")
        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as zf:
            for path in sorted(temp_root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(temp_root).as_posix())


def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_ROOT}")
    if not EXISTING_ANNOTATIONS.is_file():
        raise FileNotFoundError(f"Completed pilot annotations not found: {EXISTING_ANNOTATIONS}")
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {OUT_DIR}\n"
            "Move/rename that directory manually if you want a fresh batch; nothing was overwritten."
        )

    old_annotations = read_json(EXISTING_ANNOTATIONS)
    existing_ids = {text_value(row.get("sample_id")) for row in old_annotations if text_value(row.get("sample_id"))}
    if len(existing_ids) != len(old_annotations):
        raise ValueError("Completed annotations contain missing or duplicate IDs.")
    print(f"Existing annotated sample IDs to exclude: {len(existing_ids)}")

    all_records = []
    seen_ids = set()
    corpus_count = 0
    missing_audio = 0
    missing_two_images = 0
    missing_assets_examples = []
    video_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())

    for video_dir in video_dirs:
        dataset_path = video_dir / "dataset.json"
        if not dataset_path.is_file():
            continue
        records = read_json(dataset_path)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {dataset_path}")

        for record in records:
            sample_id = text_value(record.get("sample_id"))
            if not sample_id:
                raise ValueError(f"Record without sample_id in {dataset_path}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate corpus sample ID: {sample_id}")
            seen_ids.add(sample_id)
            corpus_count += 1
            if sample_id in existing_ids:
                continue

            inspected = inspect_record(record, video_dir)
            if inspected is None:
                mods = record.get("modalities") or {}
                audio_obj = mods.get("audio") or {}
                image_obj = mods.get("image") or {}
                if resolve_asset(audio_obj.get("path"), video_dir) is None:
                    missing_audio += 1
                found_images = [
                    resolve_asset(x, video_dir)
                    for x in (image_obj.get("paths") or [])
                ]
                if len([x for x in found_images if x is not None]) < 2:
                    missing_two_images += 1
                if len(missing_assets_examples) < 10:
                    missing_assets_examples.append(sample_id)
                continue
            all_records.append(inspected)

    print(f"Corpus records discovered: {corpus_count}")
    print(f"Unannotated samples with usable audio + two images: {len(all_records)}")
    print(f"Unannotated samples missing a usable audio file: {missing_audio}")
    print(f"Unannotated samples missing two usable images: {missing_two_images}")

    if len(all_records) < BATCH_SIZE:
        example_text = ", ".join(missing_assets_examples) or "none"
        raise RuntimeError(
            f"Only {len(all_records)} unannotated samples have usable audio and two images; "
            f"need {BATCH_SIZE}. No batch files were generated. Example ineligible IDs: {example_text}"
        )

    rng = random.Random(RANDOM_SEED)
    batch = select_diverse(all_records, BATCH_SIZE, rng)
    batch_ids = {row["sample_id"] for row in batch}
    if batch_ids & existing_ids:
        raise RuntimeError("Selection contains an already annotated ID.")
    second_batch = select_diverse(batch, DOUBLE_ANNOTATION_SIZE, random.Random(RANDOM_SEED + 1))
    if not {row["sample_id"] for row in second_batch}.issubset(batch_ids):
        raise RuntimeError("Overlap batch contains IDs outside the primary batch.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guide_path = OUT_DIR / "annotation_guide.md"
    guide_path.write_text(GUIDE, encoding="utf-8")
    main_csv = OUT_DIR / "annotation_batch_300.csv"
    second_csv = OUT_DIR / "second_annotator_100.csv"
    write_csv(main_csv, [csv_row(row) for row in batch])
    write_csv(second_csv, [csv_row(row) for row in second_batch])

    main_summary = summary_for(batch, "primary_annotator_300")
    main_summary.update({
        "corpus_sample_count": corpus_count,
        "completed_pilot_annotations_excluded": len(existing_ids),
        "double_annotation_overlap_count": len(second_batch),
        "review_format": "Extracted WAV + two representative images + transcript/OCR context; original candidate MP4 clips were not retained.",
    })
    second_summary = summary_for(second_batch, "independent_annotator_100")
    second_summary.update({
        "overlap_with_primary_batch": len(second_batch),
        "instruction": "Label independently; do not view the primary annotator's labels first.",
        "review_format": "Extracted WAV + two representative images + transcript/OCR context; original candidate MP4 clips were not retained.",
    })
    main_summary_path = OUT_DIR / "batch_summary_300.json"
    second_summary_path = OUT_DIR / "second_annotator_summary_100.json"
    main_summary_path.write_text(json.dumps(main_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    second_summary_path.write_text(json.dumps(second_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "selected_sample_ids.txt").write_text(
        "\n".join(r["sample_id"] for r in batch) + "\n", encoding="utf-8"
    )

    main_zip = OUT_DIR / "MMHD_annotation_batch_300.zip"
    second_zip = OUT_DIR / "MMHD_independent_review_100.zip"
    print("\nCreating primary annotation package from existing audio and images...")
    build_package(main_zip, batch, main_csv, main_summary_path, "primary_300", "Primary annotator · 300 samples")
    print("\nCreating independent review package...")
    build_package(second_zip, second_batch, second_csv, second_summary_path, "independent_100", "Independent reviewer · 100 samples")

    print("\n--- Annotation Batch Created ---")
    print(f"Corpus records: {corpus_count}")
    print(f"Existing annotated sample IDs excluded: {len(existing_ids)}")
    print(f"New primary samples: {len(batch)}")
    print(f"Independent overlapping samples: {len(second_batch)}")
    print(f"Source videos in primary batch: {len({r['video_id'] for r in batch})}")
    for modality, value in main_summary["modality_availability_counts"].items():
        print(f"  {modality}: {value}/{len(batch)}")
    print("\nFiles created:")
    for path in [main_csv, second_csv, guide_path, main_summary_path, second_summary_path,
                 OUT_DIR / "selected_sample_ids.txt", main_zip, second_zip]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.relative_to(ROOT)} ({size_mb:.1f} MB)")
    print("\nThe original dataset and existing annotations were not modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
