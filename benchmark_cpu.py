import os
import sys
import json
import time
import shutil
import tempfile
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import extract


SOURCE_FOLDER = os.path.join(
    "sample_clips",
    "4_dGBVQ5gAc"
)

TARGET_SAMPLES = 50


def load_first_candidates():
    candidates = extract.load_candidates(
        SOURCE_FOLDER
    )

    if len(candidates) < TARGET_SAMPLES:
        raise RuntimeError(
            f"Expected at least {TARGET_SAMPLES} "
            f"candidates, found {len(candidates)}"
        )

    return candidates[:TARGET_SAMPLES]


def main():

    candidates = load_first_candidates()

    print(
        f"Benchmark candidates: {len(candidates)}"
    )

    print(
        f"First: {candidates[0]['candidate_id']}"
    )

    print(
        f"Last: {candidates[-1]['candidate_id']}"
    )

    with tempfile.TemporaryDirectory(
        prefix="mmhd_cpu_benchmark_"
    ) as temp_root:

        benchmark_folder = os.path.join(
            temp_root,
            "4_dGBVQ5gAc"
        )

        os.makedirs(
            benchmark_folder
        )

        # Copy only the source scene videos
        # required by the 50 candidates.
        copied = set()

        for candidate in candidates:

            source_file = candidate[
                "source_file"
            ]

            if source_file in copied:
                continue

            source_path = os.path.join(
                SOURCE_FOLDER,
                source_file
            )

            destination_path = os.path.join(
                benchmark_folder,
                source_file
            )

            if not os.path.exists(source_path):
                raise FileNotFoundError(
                    source_path
                )

            shutil.copy2(
                source_path,
                destination_path
            )

            copied.add(source_file)

        print(
            f"Copied source videos: {len(copied)}"
        )

        print(
            "\nLoading CPU Whisper..."
        )

        extract.whisper = extract.WhisperModel(
            extract.WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

        print(
            "Whisper ready."
        )

        print(
            "\nStarting OCR worker..."
        )

        (
            ocr_process,
            ocr_request_queue,
            ocr_response_queue
        ) = extract.start_ocr_worker()

        completed = 0

        start_time = time.perf_counter()

        try:

            for index, candidate in enumerate(
                candidates,
                start=1
            ):

                print(
                    f"\n[{index}/{len(candidates)}] "
                    f"{candidate['candidate_id']}"
                )

                sample_start = time.perf_counter()

                result = extract.process_candidate(
                    benchmark_folder,
                    candidate,
                    ocr_request_queue,
                    ocr_response_queue
                )

                sample_elapsed = (
                    time.perf_counter()
                    - sample_start
                )

                completed += 1

                print(
                    f"Benchmark time: "
                    f"{sample_elapsed:.2f}s"
                )

        finally:

            ocr_request_queue.put(
                None
            )

            if ocr_process.is_alive():
                ocr_process.join(
                    timeout=10
                )

            if ocr_process.is_alive():
                ocr_process.terminate()
                ocr_process.join()

        total_elapsed = (
            time.perf_counter()
            - start_time
        )

        samples_per_minute = (
            completed / total_elapsed * 60
        )

        seconds_per_sample = (
            total_elapsed / completed
        )

        print("\n" + "=" * 50)
        print("CPU BASELINE BENCHMARK")
        print("=" * 50)

        print(
            f"Samples completed : {completed}"
        )

        print(
            f"Total time        : "
            f"{total_elapsed:.2f} seconds"
        )

        print(
            f"Total time        : "
            f"{total_elapsed / 60:.2f} minutes"
        )

        print(
            f"Seconds/sample    : "
            f"{seconds_per_sample:.2f}"
        )

        print(
            f"Samples/minute    : "
            f"{samples_per_minute:.2f}"
        )

        print("=" * 50)

        print(
            "\nTemporary benchmark files were "
            "automatically removed."
        )


if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()