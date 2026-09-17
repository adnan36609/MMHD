import multiprocessing
import os
import json
import subprocess
from faster_whisper import WhisperModel
import time



SAMPLE_DIR = "sample_clips"
OUTPUT_NAME = "dataset.json"
WHISPER_MODEL = "tiny"

whisper = None


# =========================================================
# BASIC HELPERS
# =========================================================

def run_ffmpeg(command):

    return subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False
    )


# =========================================================
# AUDIO
# =========================================================

def extract_audio(
    video_path,
    audio_path,
    start_seconds,
    duration_seconds
):

    os.makedirs(
        os.path.dirname(audio_path),
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        video_path,
        "-t",
        str(duration_seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        audio_path
    ]

    result = run_ffmpeg(command)

    if result.returncode != 0:
        return {
            "status": "failed",
            "path": ""
        }

    return {
        "status": "success",
        "path": audio_path
    }


# =========================================================
# IMAGES
# =========================================================

def extract_images(
    video_path,
    output_dir,
    sample_id,
    start_seconds,
    duration
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    if duration <= 0:
        return {
            "status": "failed",
            "paths": []
        }

    timestamps = [
        start_seconds + duration / 3,
        start_seconds + (2 * duration) / 3
    ]

    image_paths = []

    for index, timestamp in enumerate(
        timestamps,
        start=1
    ):

        image_path = os.path.join(
            output_dir,
            f"{sample_id}-image-{index}.jpg"
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            image_path
        ]

        result = run_ffmpeg(command)

        if (
            result.returncode != 0
            or not os.path.exists(image_path)
        ):
            return {
                "status": "failed",
                "paths": []
            }

        image_paths.append(image_path)

    return {
        "status": "success",
        "paths": image_paths
    }


# =========================================================
# TRANSCRIPTION
# =========================================================

def transcribe(audio_path):

    if not os.path.exists(audio_path):
        return {
            "status": "failed",
            "value": ""
        }

    try:

        segments, _ = whisper.transcribe(
    audio_path,
    beam_size=1,
    vad_filter=True,
    condition_on_previous_text=False
)

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()

        if text:
            return {
                "status": "success",
                "value": text
            }

        return {
            "status": "empty",
            "value": ""
        }

    except Exception as e:

        print(
            f"Transcription failed: {e}"
        )

        return {
            "status": "failed",
            "value": ""
        }


# =========================================================
# OCR WORKER
# =========================================================

def ocr_worker(
    request_queue,
    response_queue
):

    import easyocr

    print(
        "OCR worker: loading EasyOCR..."
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    print(
        "OCR worker: ready."
    )

    while True:

        request = request_queue.get()

        if request is None:
            break

        sample_id, image_path = request

        try:
            results = reader.readtext(
                image_path,
                detail=0,
                mag_ratio=0.75
            )

            text = " ".join(
                item.strip()
                for item in results
                if item.strip()
            ).strip()

            if text:

                response_queue.put({
                    "sample_id": sample_id,
                    "status": "success",
                    "value": text
                })

            else:

                response_queue.put({
                    "sample_id": sample_id,
                    "status": "empty",
                    "value": ""
                })

        except Exception as e:

            print(
                f"OCR failed for {sample_id}: {e}"
            )

            response_queue.put({
                "sample_id": sample_id,
                "status": "failed",
                "value": ""
            })


def start_ocr_worker():

    request_queue = multiprocessing.Queue()
    response_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=ocr_worker,
        args=(
            request_queue,
            response_queue
        )
    )

    process.start()

    return (
        process,
        request_queue,
        response_queue
    )


def extract_ocr(
    sample_id,
    image_path,
    ocr_request_queue,
    ocr_response_queue
):

    ocr_request_queue.put(
        (sample_id, image_path)
    )

    while True:

        result = ocr_response_queue.get()

        if result.get("sample_id") != sample_id:
            continue

        result.pop(
            "sample_id",
            None
        )

        return result


# =========================================================
# EMOJI
# =========================================================

def extract_emoji(text):

    text_lower = text.lower()

    emoji_map = {

        "laugh": "😂",
        "funny": "😂",
        "haha": "😂",
        "lol": "😂",

        "hilarious": "🤣",

        "love": "❤️",

        "angry": "😡",

        "sad": "😢",
        "cry": "😭",

        "wow": "😮",
        "surprise": "😮",

        "shock": "😱"
    }

    found = []

    for keyword, emoji in emoji_map.items():

        if keyword in text_lower:

            if emoji not in found:
                found.append(emoji)

    if found:

        return {
            "status": "success",
            "value": " ".join(found)
        }

    return {
        "status": "empty",
        "value": ""
    }


# =========================================================
# PROCESS ONE CANDIDATE
# =========================================================
def process_candidate(
    video_folder,
    candidate,
    ocr_request_queue,
    ocr_response_queue
):

    start_time = time.time()

    candidate_id = candidate["candidate_id"]

    source_file = candidate["source_file"]

    start_seconds = candidate[
        "start_seconds"
    ]

    end_seconds = candidate[
        "end_seconds"
    ]

    duration_seconds = candidate[
        "duration_seconds"
    ]

    sample_id = (
        f"{os.path.basename(video_folder)}-"
        f"{candidate_id}"
    )

    video_path = os.path.join(
        video_folder,
        source_file
    )

    extracted_dir = os.path.join(
        video_folder,
        "extracted"
    )

    audio_dir = os.path.join(
        extracted_dir,
        "audio"
    )

    image_dir = os.path.join(
        extracted_dir,
        "images"
    )

    os.makedirs(
        audio_dir,
        exist_ok=True
    )

    os.makedirs(
        image_dir,
        exist_ok=True
    )

    # -----------------------------------------------------
    # AUDIO
    # -----------------------------------------------------

    audio_path = os.path.join(
        audio_dir,
        f"{sample_id}.wav"
    )

    audio = extract_audio(
        video_path,
        audio_path,
        start_seconds,
        duration_seconds
    )

    # -----------------------------------------------------
    # TEXT
    # -----------------------------------------------------

    if audio["status"] == "success":

        text = transcribe(
            audio["path"]
        )

    else:

        text = {
            "status": "failed",
            "value": ""
        }

    # -----------------------------------------------------
    # IMAGES
    # -----------------------------------------------------

    image = extract_images(
        video_path,
        image_dir,
        sample_id,
        start_seconds,
        duration_seconds
    )

    # -----------------------------------------------------
    # OCR
    # -----------------------------------------------------

    if image["status"] == "success":

        ocr_values = []

        for image_index, image_path in enumerate(
            image["paths"],
            start=1
        ):

            image_ocr = extract_ocr(
                f"{sample_id}-image-{image_index}",
                image_path,
                ocr_request_queue,
                ocr_response_queue
            )

            if image_ocr["value"]:

                ocr_values.append(
                    image_ocr["value"]
                )

        ocr = {
            "status": (
                "success"
                if ocr_values
                else "empty"
            ),
            "value": " ".join(
                ocr_values
            )
        }

    else:

        ocr = {
            "status": "failed",
            "value": ""
        }

    # -----------------------------------------------------
    # EMOJI
    # -----------------------------------------------------

    emoji = extract_emoji(
        text["value"]
    )

    # -----------------------------------------------------
    # FINAL SAMPLE
    # -----------------------------------------------------

    result = {

        "sample_id": sample_id,

        "source": {

            "video_id": os.path.basename(
                video_folder
            ),

            "video_file": (
                os.path.basename(
                    video_folder
                ) + ".mp4"
            ),

            "source_scene": candidate[
                "source_scene"
            ],

            "source_file": source_file
        },

        "timing": {

            "start_seconds": start_seconds,

            "end_seconds": end_seconds,

            "duration_seconds": duration_seconds
        },

        "modalities": {

            "text": text,

            "image": {

                "status": image["status"],

                "paths": image["paths"]
            },

            "audio": audio,

            "ocr": ocr,

            "emoji": emoji
        },

        "metadata": {

            "candidate_type": candidate.get(
                "type",
                ""
            )
        }
    }

    elapsed = time.time() - start_time

    print(
        f"Completed in {elapsed:.1f}s"
    )

    return result

# =========================================================
# LOAD CANDIDATES
# =========================================================

def load_candidates(video_folder):

    candidate_path = os.path.join(
        video_folder,
        "candidate_report.json"
    )

    if not os.path.exists(
        candidate_path
    ):

        print(
            f"Candidate report missing: "
            f"{candidate_path}"
        )

        return []

    with open(
        candidate_path,
        "r",
        encoding="utf-8"
    ) as f:

        report = json.load(f)

    return report.get(
        "candidates",
        []
    )


# =========================================================
# DATASET HELPERS
# =========================================================

def load_dataset(dataset_path):

    if not os.path.exists(
        dataset_path
    ):

        return []

    try:

        with open(
            dataset_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception as e:

        print(
            f"Could not load dataset: {e}"
        )

        return []



def candidate_matches_record(
    record,
    candidate
):

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

    return (
        source.get("source_scene")
        == candidate["source_scene"]

        and source.get("source_file")
        == candidate["source_file"]

        and abs(
            timing.get(
                "start_seconds",
                -1
            )
            - candidate["start_seconds"]
        ) <= 1e-6

        and abs(
            timing.get(
                "end_seconds",
                -1
            )
            - candidate["end_seconds"]
        ) <= 1e-6

        and abs(
            timing.get(
                "duration_seconds",
                -1
            )
            - candidate["duration_seconds"]
        ) <= 1e-6

        and metadata.get(
            "candidate_type"
        )
        == candidate.get(
            "type",
            ""
        )
    )

def save_dataset(
    dataset_path,
    dataset
):

    with open(
        dataset_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved dataset:\n"
        f"{dataset_path}"
    )


# =========================================================
# PROCESS ONE VIDEO
# =========================================================

def extract_video_dataset(
    video_folder
):

    global whisper

    dataset_path = os.path.join(
        video_folder,
        OUTPUT_NAME
    )

    candidates = load_candidates(
        video_folder
    )

    if not candidates:

        print(
            f"No candidates found for "
            f"{video_folder}"
        )

        return

    dataset = load_dataset(
        dataset_path
    )

    records_by_id = {

        item.get("sample_id"): item

        for item in dataset

        if item.get("sample_id")
    }

    print(
        f"\nProcessing "
        f"{os.path.basename(video_folder)}"
    )

    print(
        f"Candidates: {len(candidates)}"
    )

    print(
        f"Existing records: "
        f"{len(records_by_id)}"
    )

    # -----------------------------------------------------
    # WHISPER
    # -----------------------------------------------------

    if whisper is None:

        print(
            "Loading Whisper..."
        )

        whisper = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

        print(
            "Whisper ready."
        )

    # -----------------------------------------------------
    # OCR WORKER
    # -----------------------------------------------------

    (
        ocr_process,
        ocr_request_queue,
        ocr_response_queue
    ) = start_ocr_worker()

    total_processed = 0

    try:

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            candidate_id = candidate[
                "candidate_id"
            ]

            sample_id = (
                f"{os.path.basename(video_folder)}-"
                f"{candidate_id}"
            )

            existing_record = records_by_id.get(
                sample_id
            )

            if existing_record is not None:

                if candidate_matches_record(
                    existing_record,
                    candidate
                ):
                    continue

                print(
                    f"Stale record detected for "
                    f"{sample_id}. Reprocessing."
                )

                dataset = [
                    item
                    for item in dataset
                    if item.get("sample_id")
                    != sample_id
                ]

                records_by_id.pop(
                    sample_id,
                    None
                )

            print(
                f"\nProcessing "
                f"{os.path.basename(video_folder)}/"
                f"{candidate_id}..."
            )

            result = process_candidate(
                video_folder,
                candidate,
                ocr_request_queue,
                ocr_response_queue
            )

            dataset.append(
                result
            )

            records_by_id[
                sample_id
            ] = result

            save_dataset(
                dataset_path,
                dataset
            )

            total_processed += 1

    except KeyboardInterrupt:

        print(
            "\nExtraction interrupted by user."
        )

    finally:

        # -------------------------------------------------
        # STOP OCR WORKER
        # -------------------------------------------------

        try:

            ocr_request_queue.put(
                None
            )

        except Exception:
            pass

        if ocr_process.is_alive():

            ocr_process.join(
                timeout=10
            )

        if ocr_process.is_alive():

            ocr_process.terminate()

            ocr_process.join()

    print(
        f"\nCompleted this run: "
        f"{total_processed}"
    )

    print(
        f"Total dataset samples: "
        f"{len(dataset)}"
    )

    print(
        f"Dataset: {dataset_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    if not os.path.exists(
        SAMPLE_DIR
    ):

        print(
            f"Sample directory not found: "
            f"{SAMPLE_DIR}"
        )

        return

    video_folders = []

    for name in sorted(
        os.listdir(SAMPLE_DIR)
    ):

        path = os.path.join(
            SAMPLE_DIR,
            name
        )

        if not os.path.isdir(path):
            continue

        candidate_report = os.path.join(
            path,
            "candidate_report.json"
        )

        if os.path.exists(
            candidate_report
        ):

            video_folders.append(
                path
            )

    if not video_folders:

        print(
            "No video folders with "
            "candidate reports found."
        )

        return

    print(
        f"Found {len(video_folders)} "
        f"video folders."
    )

    for video_folder in video_folders:

        extract_video_dataset(
            video_folder
        )


if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()