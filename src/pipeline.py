import os

from download import download_video
from segment import segment_all_videos
from candidate import process_video_candidates
from extract import extract_video_dataset


def main():
    video_url = input("Enter YouTube video URL: ").strip()

    if not video_url:
        print("No URL provided.")
        return

    video_urls = [video_url]

    print("\n" + "=" * 60)
    print("STEP 1: DOWNLOADING SOURCE VIDEOS")
    print("=" * 60)

    downloaded_videos = []

    for url in video_urls:
        try:
            video_path = download_video(url)
            downloaded_videos.append(video_path)
        except Exception as e:
            print(f"Download failed: {url}")
            print(f"Error: {e}")

    # keep the rest of your existing code unchanged

    print("\n" + "=" * 60)
    print("STEP 2: SEGMENTING SOURCE VIDEOS")
    print("=" * 60)

    for video_path in downloaded_videos:
        segment_all_videos(video_path)

    print("\n" + "=" * 60)
    print("STEP 3: FORMING CANDIDATES")
    print("=" * 60)

    for video_path in downloaded_videos:
        video_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        video_folder = os.path.join(
            "sample_clips",
            video_name
        )

        process_video_candidates(video_folder)
        
    print("\n" + "=" * 60)
    print("STEP 4: EXTRACTING MULTIMODAL DATA")
    print("=" * 60)

    for video_path in downloaded_videos:
        video_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

    video_folder = os.path.join(
        "sample_clips",
        video_name
    )

    extract_video_dataset(video_folder)

    print("\n" + "=" * 60)
    print("PIPELINE STEP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()