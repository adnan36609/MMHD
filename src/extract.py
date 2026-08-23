import os
import subprocess
import whisper
import json
import easyocr

print("Loading Whisper AI... (Audio to Text)")
model = whisper.load_model("tiny")

print("Loading EasyOCR... (Image to Text)")
# We initialize it for English. For your advanced phase later, you'd add 'hi' for Hindi!
reader = easyocr.Reader(['en'], gpu=False) 

CLIP_DIR = "dataset_clips"
DATA_JSON = []

# --- Helper Function for Emoji Mapping ---
# This looks at the transcript and assigns a relevant emoji based on keywords.
def get_emoji_from_text(text):
    text_lower = text.lower()
    if "pivot" in text_lower or "stuck" in text_lower:
        return "😠" # Frustrated humor
    elif "work" in text_lower or "sketch" in text_lower:
        return "🤷" # Confusion
    elif "oh yeah" in text_lower or "yes" in text_lower:
        return "😂" # Laughing/Agreement
    elif "neither" in text_lower or "know" in text_lower:
        return "🙄" # Sarcastic/Deadpan
    else:
        return "😐" # Neutral baseline

# --- Main Pipeline Loop ---
for filename in os.listdir(CLIP_DIR):
    if filename.endswith(".mp4"):
        base_name = filename.replace(".mp4", "")
        video_path = os.path.join(CLIP_DIR, filename)
        
        audio_path = os.path.join(CLIP_DIR, f"{base_name}.wav")
        image_path = os.path.join(CLIP_DIR, f"{base_name}.jpg")
        
        print(f"\n--- Extracting 5 Modalities for {filename} ---")

        # 1. AUDIO: Extract WAV using FFmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, 
            "-q:a", "0", "-map", "a", audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. IMAGE: Extract a JPG frame from the exact middle of the clip
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, 
            "-vf", "select='eq(n\,0)'", "-vframes", "1", image_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. TEXT: Transcribe audio to text using Whisper
        try:
            result = model.transcribe(audio_path)
            transcript = result["text"].strip()
        except Exception as e:
            transcript = f"Transcription failed: {e}"

        # 4. OCR: Read any background text from the extracted image
        try:
            # reader.readtext returns a list of tuples: (bounding_box, text, confidence)
            ocr_results = reader.readtext(image_path, detail=0) 
            ocr_text = " ".join(ocr_results) if ocr_results else "No text found on screen"
        except Exception as e:
            ocr_text = f"OCR failed: {e}"

        # 5. EMOJI: Map the transcript to a reactive emoji
        emoji = get_emoji_from_text(transcript)

        print(f"Transcript: {transcript}")
        print(f"OCR: {ocr_text}")
        print(f"Emoji: {emoji}")

        # --- Assemble the Final Blueprint ---
        clip_data = {
            "id": base_name,
            "modality_paths": {
                "video_source": video_path,
                "audio_source": audio_path,
                "image_source": image_path
            },
            "features": {
                "text": transcript,
                "ocr": ocr_text,
                "emoji": emoji
            },
            "label_humor": 1 # Hardcoded for this proof of concept
        }
        DATA_JSON.append(clip_data)

# Export the master JSON file
with open("master_dataset.json", "w", encoding='utf-8') as f:
    json.dump(DATA_JSON, f, indent=4, ensure_ascii=False)

print("\n✅ Ultimate 5-Modality Blueprint Complete! Check master_dataset.json")