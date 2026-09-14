import os
import yt_dlp


SOURCE_DIR = "source_videos"

VIDEO_URLS = [
    "https://www.youtube.com/watch?v=dYp-KUK73RE",
]


def download_video(url):
    os.makedirs(SOURCE_DIR, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(
            SOURCE_DIR,
            "%(id)s.%(ext)s"
        ),
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            url,
            download=False
        )

        video_id = info["id"]
        output_path = os.path.join(
            SOURCE_DIR,
            f"{video_id}.mp4"
        )

        if os.path.exists(output_path):
            print(f"Already downloaded: {video_id}")
            return output_path

        ydl.download([url])

    print(f"Downloaded: {video_id}")
    print(f"Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    for url in VIDEO_URLS:
        try:
            download_video(url)

        except Exception as e:
            print(f"Download failed: {url}")
            print(f"Error: {e}")