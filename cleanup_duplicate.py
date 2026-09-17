import json
import os


VIDEO_ID = "m5voCGft51Q"
KEEP_SAMPLE_ID = f"{VIDEO_ID}-Scene-053"
REMOVE_SAMPLE_ID = f"{VIDEO_ID}-Scene-061"

VIDEO_DIR = os.path.join(
    "sample_clips",
    VIDEO_ID
)

DATASET_PATH = os.path.join(
    VIDEO_DIR,
    "dataset.json"
)

REJECTION_PATH = os.path.join(
    VIDEO_DIR,
    "diversity_rejections.json"
)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:
    dataset = json.load(f)


keep_record = None
remove_record = None

for item in dataset:

    sample_id = item.get(
        "sample_id"
    )

    if sample_id == KEEP_SAMPLE_ID:
        keep_record = item

    elif sample_id == REMOVE_SAMPLE_ID:
        remove_record = item


if keep_record is None:
    raise RuntimeError(
        f"Could not find {KEEP_SAMPLE_ID}"
    )

if remove_record is None:
    raise RuntimeError(
        f"Could not find {REMOVE_SAMPLE_ID}"
    )


keep_text = (
    keep_record
    .get("modalities", {})
    .get("text", {})
    .get("value", "")
)

remove_text = (
    remove_record
    .get("modalities", {})
    .get("text", {})
    .get("value", "")
)


if keep_text.strip() != remove_text.strip():
    raise RuntimeError(
        "The two transcripts are not identical. "
        "Aborting cleanup."
    )


# ---------------------------------------------------------
# REMOVE FROM DATASET
# ---------------------------------------------------------

new_dataset = [
    item
    for item in dataset
    if item.get("sample_id")
    != REMOVE_SAMPLE_ID
]


if len(new_dataset) != len(dataset) - 1:
    raise RuntimeError(
        "Unexpected dataset size change."
    )


with open(
    DATASET_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        new_dataset,
        f,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------------------------
# DIVERSITY REJECTION AUDIT
# ---------------------------------------------------------

if os.path.exists(
    REJECTION_PATH
):

    with open(
        REJECTION_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        rejections = json.load(f)

    if not isinstance(
        rejections,
        list
    ):
        raise RuntimeError(
            "Existing diversity rejection file "
            "is not a list."
        )

else:
    rejections = []


already_recorded = any(
    item.get("sample_id")
    == REMOVE_SAMPLE_ID
    for item in rejections
)


if not already_recorded:

    rejections.append({
        "sample_id": REMOVE_SAMPLE_ID,
        "candidate_id": "Scene-061",
        "reason": "exact_text_duplicate",
        "matched_sample_id": KEEP_SAMPLE_ID,
        "text": remove_text
    })

    with open(
        REJECTION_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            rejections,
            f,
            indent=2,
            ensure_ascii=False
        )


# ---------------------------------------------------------
# REMOVE EXTRACTED FILES
# ---------------------------------------------------------

modalities = remove_record.get(
    "modalities",
    {}
)

audio_path = (
    modalities
    .get("audio", {})
    .get("path", "")
)

image_paths = (
    modalities
    .get("image", {})
    .get("paths", [])
)


paths_removed = 0

for path in [
    audio_path,
    *image_paths
]:

    if not path:
        continue

    if os.path.exists(path):

        os.remove(path)

        paths_removed += 1


print()
print("Duplicate cleanup completed.")
print(f"Kept:    {KEEP_SAMPLE_ID}")
print(f"Removed: {REMOVE_SAMPLE_ID}")
print(f"Dataset samples: {len(new_dataset)}")
print(f"Files removed: {paths_removed}")
print(f"Diversity audit: {REJECTION_PATH}")