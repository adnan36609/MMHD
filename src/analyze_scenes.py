import os
import json


ROOT_DIR = "sample_clips"


def analyze_report(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    video_name = report["source_video"]

    # We need the actual scene durations.
    # The current report only stores min/max/average,
    # so this function will be replaced once we save
    # individual scene durations.
    print(f"\n{video_name}")
    print(f"Scene count: {report['scene_count']}")
    print(f"Average: {report['average_scene_duration_seconds']:.2f}s")
    print(f"Minimum: {report['minimum_scene_duration_seconds']:.2f}s")
    print(f"Maximum: {report['maximum_scene_duration_seconds']:.2f}s")


for root, dirs, files in os.walk(ROOT_DIR):
    for filename in files:
        if filename == "segmentation_report.json":
            analyze_report(os.path.join(root, filename))