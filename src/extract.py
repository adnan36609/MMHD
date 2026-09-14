import os
import subprocess
import json
import easyocr
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed

print("Loading Faster-Whisper AI... (Highly Optimized Audio to Text)")
# Using 'tiny' size with INT8 quantization for maximum CPU speed
model = WhisperModel("tiny", device="cpu", compute_type="int8")

print("Loading EasyOCR... (Image to Text)")
reader = easyocr.Reader(['en'], gpu=False) 

CLIP_DIR = "dataset_clips"
DATA_JSON = []

def get_emoji_from_text(text):
    text_lower = text.lower()
    if "pivot" in text_lower or "stuck" in text_lower: return "😠" 
    elif "work" in text_lower or "sketch" in text_lower: return "🤷" 
    elif "oh yeah" in text_lower or "yes" in text_lower: return "😂" 
    elif "neither" in text_lower or "know" in text_lower: return "🙄" 
    else: return "😐"

def process_clip(filename):
    base_name = filename.replace(".mp4", "")
    video_path = os.path.join(CLIP_DIR, filename)
    audio_path = os.path.join(CLIP_DIR, f"{base_name}.wav")
    image_path = os.path.join(CLIP_DIR, f"{base_name}.jpg")
    
    # 1 & 2. Extract Audio and Image
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vf", r"select='eq(n\,0)'", "-vframes", "1", image_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Transcribe with Faster-Whisper
    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcript = " ".join([segment.text for segment in segments]).strip()
    except Exception as e:
        transcript = f"Transcription failed: {e}"

    # 4. Read OCR
    try:
        ocr_results = reader.readtext(image_path, detail=0) 
        ocr_text = " ".join(ocr_results) if ocr_results else "No text found on screen"
    except Exception as e:
        ocr_text = f"OCR failed: {e}"

    # 5. Build JSON Object
    emoji = get_emoji_from_text(transcript)
    print(f"Finished {base_name}: {transcript[:40]}... [{emoji}]")
    
    return {
        "id": base_name,
        "modality_paths": {"video_source": video_path, "audio_source": audio_path, "image_source": image_path},
        "features": {"text": transcript, "ocr": ocr_text, "emoji": emoji},
        "label_humor": 1 
    }

if __name__ == "__main__":
    video_files = [f for f in os.listdir(CLIP_DIR) if f.endswith(".mp4")]
    print(f"\n🚀 Starting multiprocessing pipeline for {len(video_files)} clips...")
    
    # Process 4 clips simultaneously to max out CPU
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_clip, vf): vf for vf in video_files}
        for future in as_completed(futures):
            DATA_JSON.append(future.result())

    with open("master_dataset.json", "w", encoding='utf-8') as f:
        json.dump(DATA_JSON, f, indent=4, ensure_ascii=False)
    print("\n✅ Ultimate 5-Modality Blueprint Complete! Check master_dataset.json")