"""Extract frozen RoBERTa token embeddings for MMHD's primary evaluation set.

Runs locally on CPU and uses cached weights only.
Does not train or modify existing models.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = (
    ROOT / "annotations" / "expanded_540" / "samples_540.json"
)

EVALUATION_FILE = (
    ROOT / "results" / "expanded_540"
    / "multimodal_comparison" / "oof_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "expanded_540" / "cross_modal_reasoning"
)

OUTPUT_FILE = OUTPUT_DIR / "text_token_embeddings.npz"
SUMMARY_FILE = OUTPUT_DIR / "text_token_embeddings_summary.json"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-emoji"
MAX_LENGTH = 64
BATCH_SIZE = 4


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def text_value(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item is not None).strip()
    return str(value).strip()


def main():
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_json(SAMPLES_FILE)
    evaluation_rows = load_csv(EVALUATION_FILE)

    sample_by_id = {
        str(sample["sample_id"]): sample
        for sample in samples
    }

    sample_ids = [str(row["sample_id"]) for row in evaluation_rows]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("The evaluation file contains duplicate sample IDs.")

    missing_ids = [
        sample_id for sample_id in sample_ids
        if sample_id not in sample_by_id
    ]

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} evaluation IDs lack sample metadata."
        )

    texts = [
        text_value(sample_by_id[sample_id].get("text"))
        for sample_id in sample_ids
    ]

    empty_ids = [
        sample_ids[i] for i, text in enumerate(texts) if not text
    ]

    if empty_ids:
        raise ValueError(
            f"{len(empty_ids)} evaluated samples have empty transcripts. "
            f"Examples: {empty_ids[:5]}"
        )

    video_ids = [
        str(sample_by_id[sample_id].get("video_id", ""))
        for sample_id in sample_ids
    ]

    print(f"Evaluation samples: {len(sample_ids)}")
    print("Loading locally cached RoBERTa encoder...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    model.to("cpu")
    model.eval()

    hidden_size = int(model.config.hidden_size)

    # Save padded token representations and an explicit attention mask.
    # The mask will prevent padding positions from contributing to fusion.
    all_embeddings = np.zeros(
        (len(texts), MAX_LENGTH, hidden_size),
        dtype=np.float16,
    )
    all_masks = np.zeros(
        (len(texts), MAX_LENGTH),
        dtype=np.uint8,
    )
    original_token_counts = []

    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]

        tokenized = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = model(**tokenized)
            hidden = outputs.last_hidden_state

            attention_mask = tokenized["attention_mask"]
            hidden = hidden * attention_mask.unsqueeze(-1)

        all_embeddings[start:end] = (
            hidden.cpu().numpy().astype(np.float16)
        )
        all_masks[start:end] = (
            attention_mask.cpu().numpy().astype(np.uint8)
        )

        for text in batch_texts:
            token_ids = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            original_token_counts.append(len(token_ids))

        completed = end
        if completed % 40 == 0 or completed == len(texts):
            print(f"Processed {completed}/{len(texts)} transcripts.")

    np.savez_compressed(
        OUTPUT_FILE,
        sample_ids=np.asarray(sample_ids, dtype=str),
        video_ids=np.asarray(video_ids, dtype=str),
        text_embeddings=all_embeddings,
        attention_mask=all_masks,
        original_token_counts=np.asarray(
            original_token_counts,
            dtype=np.int32,
        ),
    )

    summary = {
        "model": MODEL_NAME,
        "device": "cpu",
        "samples": len(sample_ids),
        "hidden_size": hidden_size,
        "max_sequence_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "truncated_transcripts": sum(
            count > MAX_LENGTH for count in original_token_counts
        ),
        "original_token_count_max": max(original_token_counts),
        "empty_transcripts": 0,
        "padding_mask_saved": True,
        "embedding_dtype": "float16",
        "elapsed_seconds": round(time.time() - started, 2),
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nTEXT EMBEDDING SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()