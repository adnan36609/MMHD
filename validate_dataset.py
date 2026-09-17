import json
import glob
import os
import re
from collections import Counter


SAMPLE_DIR = "sample_clips"

MIN_EXACT_DUPLICATE_WORDS = 3

COMMON_DUPLICATE_PHRASES = {
    "thanks for watching",
}


# =========================================================
# HELPERS
# =========================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def normalize_text(text):
    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.strip()


# =========================================================
# LOAD ALL DATA
# =========================================================

dataset_files = sorted(
    glob.glob(
        os.path.join(
            SAMPLE_DIR,
            "*",
            "dataset.json"
        )
    )
)

if not dataset_files:
    print("No dataset files found.")
    raise SystemExit(1)


all_samples = []
video_data = {}


for dataset_path in dataset_files:

    video_folder = os.path.dirname(
        dataset_path
    )

    video_id = os.path.basename(
        video_folder
    )

    dataset = load_json(
        dataset_path
    )

    candidate_path = os.path.join(
        video_folder,
        "candidate_report.json"
    )

    if not os.path.exists(
        candidate_path
    ):
        print(
            f"Missing candidate report: "
            f"{candidate_path}"
        )
        raise SystemExit(1)

    candidate_report = load_json(
        candidate_path
    )

    candidates = candidate_report.get(
        "candidates",
        []
    )

    rejection_path = os.path.join(
        video_folder,
        "diversity_rejections.json"
    )

    if os.path.exists(
        rejection_path
    ):
        rejections = load_json(
            rejection_path
        )
    else:
        rejections = []

    video_data[video_id] = {
        "dataset": dataset,
        "candidates": candidates,
        "rejections": rejections
    }

    all_samples.extend(
        dataset
    )


# =========================================================
# 1. DATASET INTEGRITY
# =========================================================

print()
print("=" * 60)
print("1. DATASET INTEGRITY")
print("=" * 60)

sample_ids = [
    item.get("sample_id")
    for item in all_samples
]

duplicate_ids = [
    (sample_id, count)
    for sample_id, count
    in Counter(sample_ids).items()
    if count > 1
]

print(
    f"Dataset samples: {len(all_samples)}"
)

print(
    f"Unique IDs: {len(set(sample_ids))}"
)

print(
    f"Duplicate IDs: {len(duplicate_ids)}"
)

if duplicate_ids:
    print(duplicate_ids)

    raise SystemExit(
        "DATASET INTEGRITY FAILED"
    )

print("PASS")


# =========================================================
# 2. PHYSICAL FILES
# =========================================================

print()
print("=" * 60)
print("2. PHYSICAL FILE CHECKS")
print("=" * 60)

missing_audio = 0
missing_images = 0
invalid_image_counts = 0


for sample in all_samples:

    modalities = sample.get(
        "modalities",
        {}
    )

    audio = modalities.get(
        "audio",
        {}
    )

    image = modalities.get(
        "image",
        {}
    )

    audio_path = audio.get(
        "path",
        ""
    )

    if (
        audio.get("status")
        == "success"
        and not os.path.exists(
            audio_path
        )
    ):
        missing_audio += 1

    image_paths = image.get(
        "paths",
        []
    )

    if image.get("status") == "success":

        if len(image_paths) != 2:
            invalid_image_counts += 1

        for path in image_paths:

            if not os.path.exists(path):
                missing_images += 1


print(
    f"Missing audio: {missing_audio}"
)

print(
    f"Missing images: {missing_images}"
)

print(
    f"Invalid image counts: "
    f"{invalid_image_counts}"
)


if (
    missing_audio
    or missing_images
    or invalid_image_counts
):

    raise SystemExit(
        "PHYSICAL FILE CHECKS FAILED"
    )

print("PASS")


# =========================================================
# 3. MODALITY STATUS
# =========================================================

print()
print("=" * 60)
print("3. MODALITY STATUS CHECKS")
print("=" * 60)

allowed_statuses = {
    "success",
    "empty",
    "failed",
    "derived"
}

status_counts = {
    "text": Counter(),
    "image": Counter(),
    "audio": Counter(),
    "ocr": Counter(),
    "emoji": Counter()
}

status_problems = []


for sample in all_samples:

    modalities = sample.get(
        "modalities",
        {}
    )

    for modality in status_counts:

        status = modalities.get(
            modality,
            {}
        ).get(
            "status"
        )

        status_counts[
            modality
        ][status] += 1

        if status not in allowed_statuses:

            status_problems.append(
                (
                    sample.get(
                        "sample_id"
                    ),
                    modality,
                    status
                )
            )


for modality, counts in status_counts.items():

    print(
        f"{modality}: "
        f"{dict(counts)}"
    )


if status_problems:

    print(
        "Invalid statuses:"
    )

    for problem in status_problems:
        print(problem)

    raise SystemExit(
        "MODALITY STATUS CHECKS FAILED"
    )

print("PASS")


# =========================================================
# 4. CANDIDATE BOUNDARIES
# =========================================================

print()
print("=" * 60)
print("4. CANDIDATE BOUNDARY CHECKS")
print("=" * 60)

boundary_problems = []
candidate_count = 0


for video_id, data in video_data.items():

    for candidate in data["candidates"]:

        candidate_count += 1

        start = candidate.get(
            "start_seconds"
        )

        end = candidate.get(
            "end_seconds"
        )

        duration = candidate.get(
            "duration_seconds"
        )

        if (
            start is None
            or end is None
            or duration is None
        ):

            boundary_problems.append(
                (
                    video_id,
                    candidate.get(
                        "candidate_id"
                    ),
                    "missing timing"
                )
            )

            continue

        if start < 0:

            boundary_problems.append(
                (
                    video_id,
                    candidate.get(
                        "candidate_id"
                    ),
                    "negative start"
                )
            )

        if end <= start:

            boundary_problems.append(
                (
                    video_id,
                    candidate.get(
                        "candidate_id"
                    ),
                    "invalid interval"
                )
            )

        if abs(
            (end - start) - duration
        ) > 1e-6:

            boundary_problems.append(
                (
                    video_id,
                    candidate.get(
                        "candidate_id"
                    ),
                    "duration mismatch"
                )
            )


print(
    f"Candidates checked: "
    f"{candidate_count}"
)

print(
    f"Problems: "
    f"{len(boundary_problems)}"
)


if boundary_problems:

    for problem in boundary_problems:
        print(problem)

    raise SystemExit(
        "CANDIDATE BOUNDARY CHECKS FAILED"
    )

print("PASS")


# =========================================================
# 5. CANDIDATE ↔ DATASET / REJECTION CONTRACT
# =========================================================

print()
print("=" * 60)
print("5. CANDIDATE ↔ DATASET CONTRACT")
print("=" * 60)

contract_problems = []

accepted_count = 0
rejected_count = 0


for video_id, data in video_data.items():

    dataset = data["dataset"]
    candidates = data["candidates"]
    rejections = data["rejections"]

    dataset_by_id = {
        item.get("sample_id"): item
        for item in dataset
    }

    rejection_by_id = {
        item.get("sample_id"): item
        for item in rejections
    }

    candidate_ids = set()


    for candidate in candidates:

        candidate_id = candidate.get(
            "candidate_id"
        )

        candidate_ids.add(
            candidate_id
        )

        sample_id = (
            f"{video_id}-{candidate_id}"
        )

        in_dataset = (
            sample_id
            in dataset_by_id
        )

        in_rejections = (
            sample_id
            in rejection_by_id
        )


        # Candidate must be exactly
        # one of accepted/rejected.

        if (
            in_dataset
            and in_rejections
        ):

            contract_problems.append(
                (
                    video_id,
                    candidate_id,
                    "both accepted and rejected"
                )
            )

            continue


        if (
            not in_dataset
            and not in_rejections
        ):

            contract_problems.append(
                (
                    video_id,
                    candidate_id,
                    "missing from dataset and rejection audit"
                )
            )

            continue


        if in_rejections:

            rejected_count += 1

            rejection = rejection_by_id[
                sample_id
            ]

            if rejection.get(
                "reason"
            ) != "exact_text_duplicate":

                contract_problems.append(
                    (
                        video_id,
                        candidate_id,
                        "invalid rejection reason"
                    )
                )

            continue


        # -------------------------------------------------
        # ACCEPTED CANDIDATE
        # -------------------------------------------------

        accepted_count += 1

        record = dataset_by_id[
            sample_id
        ]

        timing = record.get(
            "timing",
            {}
        )

        source = record.get(
            "source",
            {}
        )

        metadata = record.get(
            "metadata",
            {}
        )


        if (
            source.get("source_scene")
            != candidate.get("source_scene")
        ):

            contract_problems.append(
                (
                    video_id,
                    candidate_id,
                    "source_scene mismatch"
                )
            )


        if (
            source.get("source_file")
            != candidate.get("source_file")
        ):

            contract_problems.append(
                (
                    video_id,
                    candidate_id,
                    "source_file mismatch"
                )
            )


        for field in [
            "start_seconds",
            "end_seconds",
            "duration_seconds"
        ]:

            if abs(
                timing.get(field, -1)
                - candidate.get(field, -1)
            ) > 1e-6:

                contract_problems.append(
                    (
                        video_id,
                        candidate_id,
                        f"{field} mismatch"
                    )
                )


        if (
            metadata.get(
                "candidate_type"
            )
            != candidate.get(
                "type",
                ""
            )
        ):

            contract_problems.append(
                (
                    video_id,
                    candidate_id,
                    "candidate_type mismatch"
                )
            )


    # -----------------------------------------------------
    # REJECTION AUDIT REFERENCES
    # -----------------------------------------------------

    known_sample_ids = {
        f"{video_id}-{candidate_id}"
        for candidate_id in candidate_ids
    }


    for sample_id, rejection in rejection_by_id.items():

        if sample_id not in known_sample_ids:

            contract_problems.append(
                (
                    video_id,
                    sample_id,
                    "rejection references unknown candidate"
                )
            )

            continue


        if rejection.get(
            "matched_sample_id"
        ) not in dataset_by_id:

            contract_problems.append(
                (
                    video_id,
                    sample_id,
                    "matched sample missing from dataset"
                )
            )


print(
    f"Candidates: {candidate_count}"
)

print(
    f"Accepted dataset samples: "
    f"{accepted_count}"
)

print(
    f"Diversity rejections: "
    f"{rejected_count}"
)

print(
    f"Expected accepted + rejected: "
    f"{accepted_count + rejected_count}"
)

print(
    f"Contract problems: "
    f"{len(contract_problems)}"
)


if contract_problems:

    for problem in contract_problems:
        print(problem)

    raise SystemExit(
        "CANDIDATE ↔ DATASET CONTRACT FAILED"
    )


if (
    accepted_count
    + rejected_count
    != candidate_count
):

    raise SystemExit(
        "CANDIDATE COUNT ACCOUNTING FAILED"
    )

print("PASS")


# =========================================================
# 6. TEMPORAL OVERLAP
# =========================================================

print()
print("=" * 60)
print("6. TEMPORAL OVERLAP CHECKS")
print("=" * 60)

overlapping_pairs = []
scenes_with_overlap = set()


for video_id, data in video_data.items():

    by_scene = {}


    for candidate in data["candidates"]:

        scene = candidate.get(
            "source_scene"
        )

        by_scene.setdefault(
            scene,
            []
        ).append(candidate)


    for scene, candidates in by_scene.items():

        for i in range(
            len(candidates)
        ):

            a = candidates[i]

            for j in range(
                i + 1,
                len(candidates)
            ):

                b = candidates[j]

                start_a = a[
                    "start_seconds"
                ]

                end_a = a[
                    "end_seconds"
                ]

                start_b = b[
                    "start_seconds"
                ]

                end_b = b[
                    "end_seconds"
                ]

                overlap = max(
                    0.0,
                    min(
                        end_a,
                        end_b
                    )
                    - max(
                        start_a,
                        start_b
                    )
                )


                if overlap > 1e-6:

                    overlapping_pairs.append(
                        (
                            video_id,
                            scene,
                            a.get(
                                "candidate_id"
                            ),
                            b.get(
                                "candidate_id"
                            ),
                            overlap
                        )
                    )

                    scenes_with_overlap.add(
                        (
                            video_id,
                            scene
                        )
                    )


print(
    f"Candidate reports: "
    f"{len(video_data)}"
)

print(
    f"Overlapping candidate pairs: "
    f"{len(overlapping_pairs)}"
)

print(
    f"Scenes with overlap: "
    f"{len(scenes_with_overlap)}"
)


if overlapping_pairs:

    for pair in overlapping_pairs:
        print(pair)

    raise SystemExit(
        "TEMPORAL OVERLAP CHECKS FAILED"
    )

print("PASS")


# =========================================================
# 7. EXACT TRANSCRIPT DIVERSITY
# =========================================================

print()
print("=" * 60)
print("7. EXACT TRANSCRIPT DIVERSITY")
print("=" * 60)

normalized_texts = {}

duplicate_groups = {}

usable_text_samples = 0


for sample in all_samples:

    text = (
        sample
        .get("modalities", {})
        .get("text", {})
        .get("value", "")
    )

    normalized = normalize_text(
        text
    )

    if not normalized:
        continue

    usable_text_samples += 1

    normalized_texts.setdefault(
        normalized,
        []
    ).append(
        sample.get("sample_id")
    )


for text, sample_ids in normalized_texts.items():

    if len(sample_ids) <= 1:
        continue

    word_count = len(
        text.split()
    )

    # 1–2 word duplicates are allowed.
    if word_count < MIN_EXACT_DUPLICATE_WORDS:
        continue

    # Explicitly allowed boilerplate phrase.
    if text in COMMON_DUPLICATE_PHRASES:
        continue

    duplicate_groups[text] = sample_ids


print(
    f"Samples with usable text: "
    f"{usable_text_samples}"
)

print(
    f"Exact duplicate groups: "
    f"{len(duplicate_groups)}"
)

print(
    f"Samples belonging to duplicate groups: "
    f"{sum(len(v) for v in duplicate_groups.values())}"
)


if duplicate_groups:

    for text, sample_ids in duplicate_groups.items():

        print()

        print(
            f"Text: {text}"
        )

        for sample_id in sample_ids:

            print(
                f"  {sample_id}"
            )

    raise SystemExit(
        "EXACT TRANSCRIPT DIVERSITY CHECK FAILED"
    )

print("PASS")


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print("FINAL VALIDATION SUMMARY")
print("=" * 60)

print(
    f"Dataset samples: {len(all_samples)}"
)

print(
    f"Candidates: {candidate_count}"
)

print(
    f"Accepted: {accepted_count}"
)

print(
    f"Diversity rejected: {rejected_count}"
)

print(
    f"Accounting: "
    f"{accepted_count} + {rejected_count} "
    f"= {candidate_count}"
)

print()

print(
    "ALL VALIDATION CHECKS PASSED"
)