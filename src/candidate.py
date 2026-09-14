import os
import json

SAMPLE_DIR = "sample_clips"

MAX_SCENE_DURATION = 30.0
WINDOW_DURATION = 10.0
OVERLAP = 5.0


def process_scene_folder(scene_folder):
    report_path = os.path.join(
        scene_folder,
        "segmentation_report.json"
    )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    source_video = report["source_video"]
    video_name = os.path.splitext(source_video)[0]
    durations = report["scene_durations_seconds"]

    candidates = []

    for scene_number, duration in enumerate(durations, start=1):

        scene_id = f"Scene-{scene_number:03d}"
        scene_file = f"{video_name}-{scene_id}.mp4"

        if duration <= MAX_SCENE_DURATION:
            candidates.append({
                "candidate_id": scene_id,
                "source_scene": scene_id,
                "source_file": scene_file,
                "start_seconds": 0.0,
                "end_seconds": duration,
                "duration_seconds": duration,
                "type": "natural_scene"
            })

        else:
            start = 0.0
            window_number = 1

            while start < duration:
                end = min(
                    start + WINDOW_DURATION,
                    duration
                )

                remaining = duration - start

                # Don't create a tiny final fragment.
                if remaining < 3.0:
                    break

                candidates.append({
                    "candidate_id": (
                        f"{scene_id}-"
                        f"Window-{window_number:03d}"
                    ),
                    "source_scene": scene_id,
                    "source_file": scene_file,
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": end - start,
                    "type": "long_scene_window"
                })

                window_number += 1
                start += WINDOW_DURATION - OVERLAP

    candidate_report = {
        "source_video": source_video,
        "max_natural_scene_duration": MAX_SCENE_DURATION,
        "window_duration": WINDOW_DURATION,
        "overlap": OVERLAP,
        "candidate_count": len(candidates),
        "candidates": candidates
    }

    output_path = os.path.join(
        scene_folder,
        "candidate_report.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            candidate_report,
            f,
            indent=4
        )

    print(
        f"{source_video}: "
        f"{len(candidates)} candidates"
    )


def process_video_candidates(video_folder):
    if not os.path.isdir(video_folder):
        print(f"Folder not found: {video_folder}")
        return

    report_path = os.path.join(
        video_folder,
        "segmentation_report.json"
    )

    if not os.path.exists(report_path):
        print(
            f"No segmentation report found: "
            f"{video_folder}"
        )
        return

    process_scene_folder(video_folder)


if __name__ == "__main__":
    if not os.path.exists(SAMPLE_DIR):
        print(
            f"Directory not found: {SAMPLE_DIR}"
        )
        raise SystemExit

    for folder in os.listdir(SAMPLE_DIR):
        scene_folder = os.path.join(
            SAMPLE_DIR,
            folder
        )

        if not os.path.isdir(scene_folder):
            continue

        process_video_candidates(scene_folder)