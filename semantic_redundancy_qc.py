import os
import json
import re

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel


SAMPLE_DIR = "sample_clips"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MIN_TEXT_LENGTH = 20

CALIBRATION_THRESHOLDS = [
    0.80,
    0.85,
    0.88,
    0.90,
    0.92,
    0.95
]


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def load_samples():
    samples = []

    for folder in os.listdir(SAMPLE_DIR):
        folder_path = os.path.join(
            SAMPLE_DIR,
            folder
        )

        if not os.path.isdir(folder_path):
            continue

        dataset_path = os.path.join(
            folder_path,
            "dataset.json"
        )

        if not os.path.exists(dataset_path):
            continue

        with open(
            dataset_path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        for sample in data:

            text = sample[
                "modalities"
            ][
                "text"
            ][
                "value"
            ]

            if not text:
                continue

            normalized = normalize_text(text)

            if len(normalized) < MIN_TEXT_LENGTH:
                continue

            samples.append({
                "sample_id": sample["sample_id"],
                "video_id": sample["source"]["video_id"],
                "source_scene": sample["source"]["source_scene"],
                "text": text,
                "normalized_text": normalized
            })

    return samples


def mean_pooling(
    model_output,
    attention_mask
):
    token_embeddings = (
        model_output.last_hidden_state
    )

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    return torch.sum(
        token_embeddings * mask,
        dim=1
    ) / torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )


def encode_texts(
    texts,
    tokenizer,
    model,
    batch_size=32
):
    embeddings = []

    for i in range(
        0,
        len(texts),
        batch_size
    ):
        batch = texts[
            i:i + batch_size
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(
                **encoded
            )

        pooled = mean_pooling(
            output,
            encoded["attention_mask"]
        )

        pooled = torch.nn.functional.normalize(
            pooled,
            p=2,
            dim=1
        )

        embeddings.append(
            pooled.cpu().numpy()
        )

    return np.vstack(embeddings)


def build_pairs(
    samples,
    embeddings
):
    video_groups = {}

    for i, sample in enumerate(samples):
        video_groups.setdefault(
            sample["video_id"],
            []
        ).append(i)

    pairs = []

    for video_id, indices in video_groups.items():

        if len(indices) < 2:
            continue

        group_embeddings = embeddings[
            indices
        ]

        similarity_matrix = np.matmul(
            group_embeddings,
            group_embeddings.T
        )

        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):

                similarity = float(
                    similarity_matrix[a][b]
                )

                sample_a = samples[
                    indices[a]
                ]

                sample_b = samples[
                    indices[b]
                ]

                pairs.append({
                    "video_id": video_id,
                    "similarity": similarity,
                    "same_scene": (
                        sample_a["source_scene"]
                        == sample_b["source_scene"]
                    ),
                    "sample_a": {
                        "sample_id": sample_a["sample_id"],
                        "scene": sample_a["source_scene"],
                        "text": sample_a["text"]
                    },
                    "sample_b": {
                        "sample_id": sample_b["sample_id"],
                        "scene": sample_b["source_scene"],
                        "text": sample_b["text"]
                    }
                })

    return pairs


def find_exact_duplicates(samples):
    groups = {}

    for sample in samples:
        groups.setdefault(
            sample["normalized_text"],
            []
        ).append(sample)

    duplicates = []

    for normalized_text, group in groups.items():

        if len(group) < 2:
            continue

        duplicates.append({
            "normalized_text": normalized_text,
            "count": len(group),
            "samples": [
                {
                    "sample_id": item["sample_id"],
                    "video_id": item["video_id"],
                    "scene": item["source_scene"],
                    "text": item["text"]
                }
                for item in group
            ]
        })

    return duplicates


def main():

    print("Loading samples...")

    samples = load_samples()

    print(
        f"Samples with usable text: "
        f"{len(samples)}"
    )

    print("\nFinding exact normalized duplicates...")

    exact_duplicates = find_exact_duplicates(
        samples
    )

    duplicate_sample_count = sum(
        item["count"]
        for item in exact_duplicates
    )

    print(
        f"Exact duplicate groups: "
        f"{len(exact_duplicates)}"
    )

    print(
        f"Samples belonging to duplicate groups: "
        f"{duplicate_sample_count}"
    )

    print("\nLoading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    texts = [
        sample["text"]
        for sample in samples
    ]

    print("Generating embeddings...")

    embeddings = encode_texts(
        texts,
        tokenizer,
        model
    )

    print("\nChecking within-video semantic similarity...")

    pairs = build_pairs(
        samples,
        embeddings
    )

    similarities = np.array([
        pair["similarity"]
        for pair in pairs
    ])

    print(
        f"Total within-video pairs: "
        f"{len(pairs)}"
    )

    if len(similarities) > 0:

        print("\nSimilarity distribution:")

        print(
            f"Minimum: "
            f"{similarities.min():.4f}"
        )

        print(
            f"Mean: "
            f"{similarities.mean():.4f}"
        )

        print(
            f"Median: "
            f"{np.median(similarities):.4f}"
        )

        print(
            f"90th percentile: "
            f"{np.percentile(similarities, 90):.4f}"
        )

        print(
            f"95th percentile: "
            f"{np.percentile(similarities, 95):.4f}"
        )

        print(
            f"99th percentile: "
            f"{np.percentile(similarities, 99):.4f}"
        )

        print(
            f"Maximum: "
            f"{similarities.max():.4f}"
        )

    print("\n" + "=" * 70)
    print("THRESHOLD CALIBRATION")
    print("=" * 70)

    threshold_results = []

    for threshold in CALIBRATION_THRESHOLDS:

        selected = [
            pair
            for pair in pairs
            if pair["similarity"] >= threshold
        ]

        same_scene = sum(
            pair["same_scene"]
            for pair in selected
        )

        cross_scene = (
            len(selected)
            - same_scene
        )

        threshold_results.append({
            "threshold": threshold,
            "flagged_pairs": len(selected),
            "same_scene_pairs": same_scene,
            "cross_scene_pairs": cross_scene
        })

        print(
            f"{threshold:.2f} -> "
            f"{len(selected)} pairs "
            f"({same_scene} same-scene, "
            f"{cross_scene} cross-scene)"
        )

    pairs.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    print("\n" + "=" * 70)
    print("TOP 30 MOST SIMILAR PAIRS")
    print("=" * 70)

    for i, pair in enumerate(
        pairs[:30],
        start=1
    ):

        print(
            f"\n[{i}] "
            f"Similarity: "
            f"{pair['similarity']:.4f}"
        )

        print(
            f"    Video: "
            f"{pair['video_id']}"
        )

        print(
            f"    Same scene: "
            f"{pair['same_scene']}"
        )

        print(
            f"    A: "
            f"{pair['sample_a']['sample_id']} "
            f"({pair['sample_a']['scene']})"
        )

        print(
            f"       "
            f"{pair['sample_a']['text']}"
        )

        print(
            f"    B: "
            f"{pair['sample_b']['sample_id']} "
            f"({pair['sample_b']['scene']})"
        )

        print(
            f"       "
            f"{pair['sample_b']['text']}"
        )

    output_path = (
        "semantic_redundancy_calibration.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "usable_text_samples": len(samples),
                "total_within_video_pairs": len(pairs),
                "exact_duplicate_groups": len(
                    exact_duplicates
                ),
                "exact_duplicate_groups_detail":
                    exact_duplicates,
                "threshold_calibration":
                    threshold_results,
                "top_similarity_pairs":
                    pairs[:100]
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nCalibration report saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()