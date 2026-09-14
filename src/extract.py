import multiprocessing
import os
import json
import subprocess
import queue

from faster_whisper import WhisperModel


SAMPLE_DIR = "sample_clips"
OUTPUT_NAME = "dataset.json"
WHISPER_MODEL = "tiny"

whisper = None


def run_ffmpeg(command):
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.returncode == 0


def extract_audio(video_path, audio_path, start, duration):
    success = run_ffmpeg([
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-vn",
        "-acodec",
        "pcm_s16le",
        audio_path
    ])

    if not success or not os.path.exists(audio_path):
        return {
            "status": "failed",
            "path": ""
        }

    return {
        "status": "success",
        "path": audio_path
    }


def extract_image(video_path, image_path, start, duration):
    representative_time = duration / 2

    success = run_ffmpeg([
        "ffmpeg",
        "-y",
        "-ss",
        str(start + representative_time),
        "-i",
        video_path,
        "-frames:v",
        "1",
        image_path
    ])

    if not success or not os.path.exists(image_path):
        return {
            "status": "failed",
            "path": ""
        }

    return {
        "status": "success",
        "path": image_path
    }


def transcribe(audio_path):
    try:
        segments, _ = whisper.transcribe(
            audio_path,
            beam_size=5
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        if not text:
            return {
                "status": "empty",
                "value": ""
            }

        return {
            "status": "success",
            "value": text
        }

    except Exception as e:
        print(f"Whisper failed: {e}")

        return {
            "status": "failed",
            "value": ""
        }


def ocr_worker(request_queue, response_queue):
    import easyocr

    print("OCR worker: loading EasyOCR...")

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    print("OCR worker: ready.")

    while True:
        request = request_queue.get()

        if request is None:
            break

        sample_id, image_path = request

        try:
            results = reader.readtext(
                image_path,
                detail=0
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
            print(f"OCR failed for {sample_id}: {e}")

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
        ),
        daemon=True
    )

    process.start()

    return (
        process,
        request_queue,
        response_queue
    )


def restart_ocr_worker(
    ocr_process,
    ocr_request_queue,
    ocr_response_queue
):
    print("Restarting OCR worker...")

    if ocr_process.is_alive():
        ocr_process.terminate()
        ocr_process.join(timeout=5)

    return start_ocr_worker()


def extract_ocr(
    sample_id,
    image_path,
    ocr_request_queue,
    ocr_response_queue,
    timeout=60
):
    ocr_request_queue.put(
        (sample_id, image_path)
    )

    try:
        while True:
            result = ocr_response_queue.get(
                timeout=timeout
            )

            # Ignore stale responses.
            if result.get("sample_id") != sample_id:
                continue

            result.pop("sample_id", None)

            return result

    except queue.Empty:
        print(f"OCR timeout: {sample_id}")

        return {
            "status": "timeout",
            "value": ""
        }


def derive_emoji(text):
    text_lower = text.lower()

    if "pivot" in text_lower or "stuck" in text_lower:
        emoji = "\U0001F621"

    elif "work" in text_lower or "sketch" in text_lower:
        emoji = "\U0001F937"

    elif "oh yeah" in text_lower or "yes" in text_lower:
        emoji = "\U0001F602"

    elif "neither" in text_lower or "know" in text_lower:
        emoji = "\U0001F644"

    else:
        emoji = "\U0001F610"

    return {
        "status": "derived",
        "value": emoji
    }


def relative_path(path):
    return os.path.relpath(path).replace("\\", "/")


def process_candidate(
    video_folder,
    candidate,
    ocr_request_queue,
    ocr_response_queue
):
    video_name = os.path.basename(video_folder)

    candidate_id = candidate["candidate_id"]
    source_file = candidate["source_file"]

    start = candidate["start_seconds"]
    end = candidate["end_seconds"]
    duration = candidate["duration_seconds"]

    video_path = os.path.join(
        video_folder,
        source_file
    )

    output_folder = os.path.join(
        video_folder,
        "extracted"
    )

    audio_folder = os.path.join(
        output_folder,
        "audio"
    )

    image_folder = os.path.join(
        output_folder,
        "images"
    )

    os.makedirs(
        audio_folder,
        exist_ok=True
    )

    os.makedirs(
        image_folder,
        exist_ok=True
    )

    audio_path = os.path.join(
        audio_folder,
        f"{candidate_id}.wav"
    )

    image_path = os.path.join(
        image_folder,
        f"{candidate_id}.jpg"
    )

    sample_id = f"{video_name}-{candidate_id}"

    print(
        f"Processing "
        f"{video_name}/{candidate_id}..."
    )

    # -------------------------
    # AUDIO
    # -------------------------

    audio = extract_audio(
        video_path,
        audio_path,
        start,
        duration
    )

    # -------------------------
    # TEXT
    # -------------------------

    if audio["status"] == "success":
        text = transcribe(
            audio["path"]
        )
    else:
        text = {
            "status": "failed",
            "value": ""
        }

    # -------------------------
    # IMAGE
    # -------------------------

    image = extract_image(
        video_path,
        image_path,
        start,
        duration
    )

    # -------------------------
    # OCR
    # -------------------------

    if image["status"] == "success":
        ocr = extract_ocr(
            sample_id,
            image["path"],
            ocr_request_queue,
            ocr_response_queue
        )

        # Tell the caller whether OCR timed out.
        ocr_timed_out = (
            ocr["status"] == "timeout"
        )

    else:
        ocr = {
            "status": "failed",
            "value": ""
        }

        ocr_timed_out = False

    # -------------------------
    # EMOJI
    # -------------------------

    emoji = derive_emoji(
        text["value"]
        if text["status"] == "success"
        else ""
    )

    result = {
        "sample_id": sample_id,

        "source": {
            "video_id": video_name,
            "video_file": f"{video_name}.mp4",
            "source_scene": candidate["source_scene"],
            "source_file": source_file
        },

        "timing": {
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": duration
        },

        "modalities": {
            "text": text,

            "image": {
                "status": image["status"],
                "path": (
                    relative_path(image["path"])
                    if image["path"]
                    else ""
                )
            },

            "audio": {
                "status": audio["status"],
                "path": (
                    relative_path(audio["path"])
                    if audio["path"]
                    else ""
                )
            },

            "ocr": ocr,

            "emoji": emoji
        },

        "metadata": {
            "candidate_type": candidate["type"]
        }
    }

    return result, ocr_timed_out


def load_candidates(video_folder):
    report_path = os.path.join(
        video_folder,
        "candidate_report.json"
    )

    with open(
        report_path,
        "r",
        encoding="utf-8"
    ) as f:
        report = json.load(f)

    return report["candidates"]


def load_existing_dataset(output_path):
    if not os.path.exists(output_path):
        return []

    try:
        with open(
            output_path,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except (
        json.JSONDecodeError,
        OSError
    ):
        print(
            "Existing dataset could not be read. "
            "Starting with an empty dataset."
        )

        return []


def save_dataset(output_path, dataset):
    temp_path = output_path + ".tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            dataset,
            f,
            indent=4,
            ensure_ascii=False
        )

    os.replace(
        temp_path,
        output_path
    )


def extract_video_dataset(video_folder):
    global whisper

    multiprocessing.freeze_support()

    print("Loading Faster-Whisper...")

    whisper = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8"
    )

    # -------------------------
    # START OCR WORKER
    # -------------------------

    (
        ocr_process,
        ocr_request_queue,
        ocr_response_queue
    ) = start_ocr_worker()

    total_candidates = 0
    total_processed = 0
    total_skipped = 0
    total_ocr_restarts = 0

    try:
        video_folders = [video_folder]

        for video_folder in video_folders:

            report_path = os.path.join(
                video_folder,
                "candidate_report.json"
            )

            if not os.path.exists(
                report_path
            ):
                continue

            video_name = os.path.basename(
                video_folder
            )

            candidates = load_candidates(
                video_folder
            )

            output_path = os.path.join(
                video_folder,
                OUTPUT_NAME
            )

            dataset = load_existing_dataset(
                output_path
            )

            completed_ids = {
                item["sample_id"]
                for item in dataset
                if "sample_id" in item
            }

            print(
                f"\n{'=' * 60}"
                f"\nVideo: {video_name}"
                f"\nCandidates: {len(candidates)}"
                f"\nAlready completed: {len(completed_ids)}"
                f"\n{'=' * 60}"
            )

            for candidate in candidates:

                sample_id = (
                    f"{video_name}-"
                    f"{candidate['candidate_id']}"
                )

                if sample_id in completed_ids:
                    print(
                        f"Skipping {sample_id}"
                    )

                    total_skipped += 1
                    continue

                result, ocr_timed_out = process_candidate(
                    video_folder,
                    candidate,
                    ocr_request_queue,
                    ocr_response_queue
                )

                dataset.append(result)
                completed_ids.add(sample_id)

                save_dataset(
                    output_path,
                    dataset
                )

                total_processed += 1

                # -------------------------
                # OCR RECOVERY
                # -------------------------

                if ocr_timed_out:
                    (
                        ocr_process,
                        ocr_request_queue,
                        ocr_response_queue
                    ) = restart_ocr_worker(
                        ocr_process,
                        ocr_request_queue,
                        ocr_response_queue
                    )

                    total_ocr_restarts += 1

            total_candidates += len(
                candidates
            )

            print(
                f"\nSaved dataset:"
                f"\n{output_path}"
                f"\nCompleted: {len(dataset)}"
            )

    except KeyboardInterrupt:
        print(
            "\n\nExtraction interrupted by user."
        )

    finally:

        if ocr_process.is_alive():
            ocr_request_queue.put(None)

            ocr_process.join(
                timeout=10
            )

        if ocr_process.is_alive():
            ocr_process.terminate()

            ocr_process.join()
            
            print(
        f"\n{'=' * 60}"
        f"\nEXTRACTION STOPPED/COMPLETE"
        f"\nTotal candidates encountered: "
        f"{total_candidates}"
        f"\nProcessed this run: "
        f"{total_processed}"
        f"\nSkipped existing: "
        f"{total_skipped}"
        f"\nOCR worker restarts: "
        f"{total_ocr_restarts}"
        f"\n{'=' * 60}"
    )


if __name__ == "__main__":
    video_folders = [
        os.path.join(
            SAMPLE_DIR,
            folder
        )
        for folder in os.listdir(SAMPLE_DIR)
        if os.path.isdir(
            os.path.join(
                SAMPLE_DIR,
                folder
            )
        )
    ]

    if not video_folders:
        print(
            f"No video folders found in '{SAMPLE_DIR}'."
        )
        raise SystemExit

    for video_folder in video_folders:
        extract_video_dataset(video_folder)