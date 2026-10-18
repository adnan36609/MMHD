"""Train an MMHD cross-modal attention model on frozen embeddings.

Inspired by cross-modal and cross-image reasoning research.
This is not an exact reproduction of MMSD3.0 CIRM.

Inputs:
  - Frozen RoBERTa token embeddings for 391 evaluated samples.
  - Frozen ViT embeddings for two keyframes per sample.

Evaluation:
  - Five-fold GroupKFold by source video.
  - A separate group-disjoint validation split within each training fold.
  - Outer-fold predictions are saved for every evaluated sample.

All output files are separate from the existing baseline.
"""

import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = (
    ROOT / "annotations" / "expanded_540" / "samples_540.json"
)
ANNOTATIONS_FILE = (
    ROOT / "annotations" / "expanded_540" / "annotations_540.json"
)
BASELINE_FILE = (
    ROOT / "results" / "expanded_540"
    / "late_fusion" / "oof_predictions.csv"
)
TEXT_FEATURES_FILE = (
    ROOT / "results" / "expanded_540"
    / "cross_modal_reasoning" / "text_token_embeddings.npz"
)
IMAGE_FEATURES_FILE = (
    ROOT / "results" / "expanded_540"
    / "vit_keyframe_embeddings" / "embeddings.csv"
)
VISUAL_BASELINE_FILE = (
    ROOT / "results" / "expanded_540"
    / "cross_frame_vit" / "oof_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "expanded_540" / "cross_modal_reasoning"
)
PREDICTIONS_FILE = OUTPUT_DIR / "oof_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

N_SPLITS = 5
HIDDEN_SIZE = 96
NUM_HEADS = 4
BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 9
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-3
RANDOM_SEED = 42


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


class CrossModalAttentionModel(nn.Module):
    """Lightweight text–visual cross-attention classifier."""

    def __init__(self, embedding_size=768, hidden_size=96, num_heads=4):
        super().__init__()

        self.text_projection = nn.Linear(embedding_size, hidden_size)
        self.visual_projection = nn.Linear(embedding_size, hidden_size)

        self.frame_positions = nn.Parameter(
            torch.empty(1, 2, hidden_size)
        )
        nn.init.normal_(self.frame_positions, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(0.15)

        # Model relationships between the two ordered keyframes.
        self.frame_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )
        self.frame_norm = nn.LayerNorm(hidden_size)

        # Stage 1: bidirectional text–visual bridge.
        self.text_to_visual_1 = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )
        self.visual_to_text_1 = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )
        self.text_norm_1 = nn.LayerNorm(hidden_size)
        self.visual_norm_1 = nn.LayerNorm(hidden_size)

        self.text_ff_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.visual_ff_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.text_norm_2 = nn.LayerNorm(hidden_size)
        self.visual_norm_2 = nn.LayerNorm(hidden_size)

        # Stage 2: gated cross-modal refinement.
        self.text_to_visual_2 = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )
        self.visual_to_text_2 = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )

        self.text_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.visual_gate = nn.Linear(hidden_size * 2, hidden_size)

        self.text_norm_3 = nn.LayerNorm(hidden_size)
        self.visual_norm_3 = nn.LayerNorm(hidden_size)

        # Select the visual frame most relevant to the text representation.
        self.frame_relevance = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Text, relevant visual summary, frame difference,
        # cross-modal difference, and cross-modal interaction.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, text_tokens, text_mask, frame_tokens):
        # text_tokens: [batch, sequence, 768]
        # text_mask:   [batch, sequence]
        # frame_tokens:[batch, 2, 768]

        text = self.text_projection(text_tokens)
        visual = (
            self.visual_projection(frame_tokens)
            + self.frame_positions
        )

        padding_mask = ~text_mask.bool()

        # Preserve and refine relationships between ordered image frames.
        frame_context, _ = self.frame_attention(
            visual, visual, visual, need_weights=False
        )
        visual = self.frame_norm(
            visual + self.dropout(frame_context)
        )

        # Stage 1: text tokens attend to both visual frame tokens.
        text_context, _ = self.text_to_visual_1(
            text, visual, visual, need_weights=False
        )
        text = self.text_norm_1(
            text + self.dropout(text_context)
        )

        # Visual tokens attend to non-padding text tokens.
        visual_context, _ = self.visual_to_text_1(
            visual,
            text,
            text,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        visual = self.visual_norm_1(
            visual + self.dropout(visual_context)
        )

        text = self.text_norm_2(
            text + self.dropout(self.text_ff_1(text))
        )
        visual = self.visual_norm_2(
            visual + self.dropout(self.visual_ff_1(visual))
        )

        # Stage 2: gated text update using visual information.
        text_context, _ = self.text_to_visual_2(
            text, visual, visual, need_weights=False
        )
        text_gate = torch.sigmoid(
            self.text_gate(torch.cat([text, text_context], dim=-1))
        )
        text = self.text_norm_3(
            text + self.dropout(text_gate * text_context)
        )

        # Gated visual update using the refined text representation.
        visual_context, _ = self.visual_to_text_2(
            visual,
            text,
            text,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        visual_gate = torch.sigmoid(
            self.visual_gate(torch.cat([visual, visual_context], dim=-1))
        )
        visual = self.visual_norm_3(
            visual + self.dropout(visual_gate * visual_context)
        )

        # Mask padding before pooling text representations.
        mask_float = text_mask.unsqueeze(-1).to(text.dtype)
        text_summary = (
            (text * mask_float).sum(dim=1)
            / mask_float.sum(dim=1).clamp(min=1.0)
        )

        # Learn which of the two frames is more relevant to the text.
        repeated_text = text_summary.unsqueeze(1).expand(-1, 2, -1)
        relevance_logits = self.frame_relevance(
            torch.cat([visual, repeated_text], dim=-1)
        ).squeeze(-1)
        relevance_weights = torch.softmax(relevance_logits, dim=1)

        visual_summary = (
            visual * relevance_weights.unsqueeze(-1)
        ).sum(dim=1)

        frame_difference = torch.abs(visual[:, 0] - visual[:, 1])
        cross_modal_difference = torch.abs(
            text_summary - visual_summary
        )
        cross_modal_product = text_summary * visual_summary

        fused = torch.cat(
            [
                text_summary,
                visual_summary,
                frame_difference,
                cross_modal_difference,
                cross_modal_product,
            ],
            dim=-1,
        )

        return self.classifier(fused).squeeze(-1)


def compute_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "macro_f1": float(
            f1_score(
                y_true, predictions,
                average="macro", zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "roc_auc_humorous": float(
            roc_auc_score(y_true, probabilities)
        ),
    }


def load_inputs():
    samples = read_json(SAMPLES_FILE)
    annotations = read_json(ANNOTATIONS_FILE)
    baseline_rows = read_csv(BASELINE_FILE)
    visual_rows = read_csv(VISUAL_BASELINE_FILE)

    sample_by_id = {
        str(row["sample_id"]): row for row in samples
    }
    label_by_id = {
        str(row["sample_id"]): row["humor_label"]
        for row in annotations
    }

    sample_ids = [str(row["sample_id"]) for row in baseline_rows]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate IDs in the primary evaluation file.")

    if set(sample_ids) != {str(row["sample_id"]) for row in visual_rows}:
        raise ValueError("Visual and primary evaluation sample IDs differ.")

    visual_by_id = {
        str(row["sample_id"]): row for row in visual_rows
    }

    groups = np.asarray([
        str(sample_by_id[sample_id]["video_id"])
        for sample_id in sample_ids
    ])

    for row in baseline_rows:
        if row["true_label"] != label_by_id[row["sample_id"]]:
            raise ValueError(
                f"Primary baseline label mismatch: {row['sample_id']}"
            )

    text_data = np.load(TEXT_FEATURES_FILE, allow_pickle=False)

    text_ids = text_data["sample_ids"].astype(str)
    text_index = {sample_id: i for i, sample_id in enumerate(text_ids)}

    if len(text_index) != len(text_ids):
        raise ValueError("Duplicate sample IDs in text embeddings.")

    missing_text = set(sample_ids) - set(text_index)
    if missing_text:
        raise ValueError(
            f"{len(missing_text)} evaluation samples lack text embeddings."
        )

    text_indices = [text_index[sample_id] for sample_id in sample_ids]

    text_tokens = text_data["text_embeddings"][text_indices].astype(
        np.float32
    )
    text_mask = text_data["attention_mask"][text_indices].astype(bool)

    del text_data

    # Load both frame embeddings by sample ID and frame index.
    image_rows = read_csv(IMAGE_FEATURES_FILE)
    embedding_columns = sorted(
        column for column in image_rows[0]
        if column.startswith("embedding_")
    )

    image_map = {}
    for row in image_rows:
        if row["status"] != "ok":
            continue

        key = (row["sample_id"], int(row["frame_index"]))
        image_map[key] = np.asarray(
            [float(row[column]) for column in embedding_columns],
            dtype=np.float32,
        )

    frame_arrays = []
    for sample_id in sample_ids:
        frames = []

        for frame_index in (1, 2):
            key = (sample_id, frame_index)
            if key not in image_map:
                raise ValueError(f"Missing image embedding: {key}")
            frames.append(image_map[key])

        frame_arrays.append(frames)

    frame_tokens = np.asarray(frame_arrays, dtype=np.float32)

    y = np.asarray([
        1 if label_by_id[sample_id] == "humorous" else 0
        for sample_id in sample_ids
    ], dtype=np.int64)

    if text_tokens.shape[0] != len(sample_ids):
        raise ValueError("Text embedding count mismatch.")
    if frame_tokens.shape != (len(sample_ids), 2, 768):
        raise ValueError(f"Unexpected frame shape: {frame_tokens.shape}")
    if not text_mask.any(axis=1).all():
        raise ValueError("At least one evaluated sample has an empty text mask.")

    return (
        sample_ids,
        groups,
        y,
        label_by_id,
        text_tokens,
        text_mask,
        frame_tokens,
        baseline_rows,
        visual_by_id,
    )


def train_fold(
    fold_number,
    outer_train,
    outer_test,
    y,
    groups,
    text_tokens,
    text_mask,
    frame_tokens,
):
    # Inner validation is also group-disjoint by source video.
    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.22,
        random_state=RANDOM_SEED + fold_number,
    )
    train_local, val_local = next(
        inner_splitter.split(
            np.zeros(len(outer_train)),
            y[outer_train],
            groups[outer_train],
        )
    )

    inner_train = outer_train[train_local]
    inner_val = outer_train[val_local]

    if set(groups[inner_train]) & set(groups[inner_val]):
        raise RuntimeError("Inner train/validation source-video leakage.")

    train_classes = np.unique(y[inner_train])
    if len(train_classes) != 2:
        raise RuntimeError(
            f"Fold {fold_number} inner training split has one class."
        )

    torch.manual_seed(RANDOM_SEED + fold_number)
    np.random.seed(RANDOM_SEED + fold_number)
    random.seed(RANDOM_SEED + fold_number)

    model = CrossModalAttentionModel(
        embedding_size=768,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    positives = int((y[inner_train] == 1).sum())
    negatives = int((y[inner_train] == 0).sum())

    weighted_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [negatives / max(positives, 1)],
            dtype=torch.float32,
        )
    )
    validation_loss = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    all_text = torch.from_numpy(text_tokens)
    all_mask = torch.from_numpy(text_mask)
    all_frames = torch.from_numpy(frame_tokens)
    all_labels = torch.from_numpy(y.astype(np.float32))

    train_indices = torch.as_tensor(inner_train, dtype=torch.long)
    val_indices = torch.as_tensor(inner_val, dtype=torch.long)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        shuffled = train_indices[torch.randperm(len(train_indices))]

        for start in range(0, len(shuffled), BATCH_SIZE):
            batch_indices = shuffled[start:start + BATCH_SIZE]

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                all_text[batch_indices],
                all_mask[batch_indices],
                all_frames[batch_indices],
            )
            loss = weighted_loss(
                logits,
                all_labels[batch_indices],
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(
                all_text[val_indices],
                all_mask[val_indices],
                all_frames[val_indices],
            )
            loss_value = validation_loss(
                val_logits, all_labels[val_indices]
            ).item()

        if loss_value < best_loss - 1e-4:
            best_loss = loss_value
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for fold {fold_number}.")

    model.load_state_dict(best_state)
    model.eval()

    test_indices = torch.as_tensor(outer_test, dtype=torch.long)
    with torch.inference_mode():
        test_logits = model(
            all_text[test_indices],
            all_mask[test_indices],
            all_frames[test_indices],
        )
        probabilities = torch.sigmoid(test_logits).numpy()

    print(
        f"Fold {fold_number}: epochs={epoch}, best_epoch={best_epoch}, "
        f"best_validation_loss={best_loss:.4f}, "
        f"outer_test_samples={len(outer_test)}"
    )

    return probabilities, {
        "fold": fold_number,
        "outer_train_samples": len(outer_train),
        "outer_test_samples": len(outer_test),
        "inner_train_samples": len(inner_train),
        "inner_validation_samples": len(inner_val),
        "outer_train_source_videos": len(set(groups[outer_train])),
        "inner_validation_source_videos": len(set(groups[inner_val])),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_loss": float(best_loss),
    }


def main():
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid unnecessary CPU oversubscription on this CPU-only machine.
    torch.set_num_threads(min(4, torch.get_num_threads()))

    (
        sample_ids,
        groups,
        y,
        label_by_id,
        text_tokens,
        text_mask,
        frame_tokens,
        baseline_rows,
        visual_by_id,
    ) = load_inputs()

    print("Cross-modal attention experiment")
    print(f"Samples: {len(sample_ids)}")
    print(f"Source videos: {len(set(groups))}")
    print(f"Text token tensor: {text_tokens.shape}")
    print(f"Visual frame tensor: {frame_tokens.shape}")
    print("Fusion model: trainable cross-attention; encoders remain frozen.")
    print("Outer evaluation: five-fold GroupKFold by source video.")

    outer_splitter = GroupKFold(n_splits=N_SPLITS)
    probabilities = np.full(len(sample_ids), np.nan, dtype=np.float64)
    fold_assignments = np.full(len(sample_ids), -1, dtype=np.int32)
    fold_details = []

    splits = list(
        outer_splitter.split(
            np.zeros(len(sample_ids)), y, groups
        )
    )

    for fold_number, (train_idx, test_idx) in enumerate(
        splits, start=1
    ):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError(
                f"Outer source-video leakage in fold {fold_number}."
            )

        fold_probabilities, details = train_fold(
            fold_number,
            train_idx,
            test_idx,
            y,
            groups,
            text_tokens,
            text_mask,
            frame_tokens,
        )

        probabilities[test_idx] = fold_probabilities
        fold_assignments[test_idx] = fold_number
        fold_details.append(details)

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Some out-of-fold probabilities are missing.")

    new_model_metrics = compute_metrics(y, probabilities)

    # Align existing OOF predictions by ID; never use training-set scores.
    baseline_by_id = {
        row["sample_id"]: row for row in baseline_rows
    }
    baseline_probabilities = np.asarray([
        float(baseline_by_id[sample_id]["late_fusion_humorous_probability"])
        for sample_id in sample_ids
    ])

    visual_probabilities = np.asarray([
        float(
            visual_by_id[sample_id][
                "absolute_frame_difference_humorous_probability"
            ]
        )
        for sample_id in sample_ids
    ])

    baseline_metrics = compute_metrics(y, baseline_probabilities)
    visual_metrics = compute_metrics(y, visual_probabilities)

    output_rows = []
    for index, sample_id in enumerate(sample_ids):
        prediction = (
            "humorous" if probabilities[index] >= 0.5
            else "non_humorous"
        )

        output_rows.append({
            "sample_id": sample_id,
            "video_id": groups[index],
            "true_label": label_by_id[sample_id],
            "fold": int(fold_assignments[index]),
            "cross_modal_probability": float(probabilities[index]),
            "cross_modal_prediction": prediction,
            "baseline_text_audio_probability": float(
                baseline_probabilities[index]
            ),
            "visual_difference_probability": float(
                visual_probabilities[index]
            ),
        })

    with PREDICTIONS_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(output_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(output_rows)

    output = {
        "experiment": "Trainable text-visual cross-modal attention",
        "inspiration": (
            "MMSD3.0 cross-image reasoning and bridging concepts; "
            "not an exact CIRM reproduction."
        ),
        "samples": len(sample_ids),
        "source_videos": len(set(groups)),
        "folds": N_SPLITS,
        "outer_split": "GroupKFold by source video",
        "inner_validation": "GroupShuffleSplit by source video",
        "frozen_encoders": [
            "cardiffnlp/twitter-roberta-base-emoji",
            "google/vit-base-patch16-224-in21k",
        ],
        "trainable_components": [
            "frame self-attention",
            "bidirectional text-visual cross-attention",
            "gated cross-modal refinement",
            "learned frame relevance weighting",
            "humor classifier",
        ],
        "hidden_size": HIDDEN_SIZE,
        "attention_heads": NUM_HEADS,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "threshold": 0.5,
        "metrics": {
            "text_visual_cross_modal_attention": new_model_metrics,
            "existing_text_audio_late_fusion": baseline_metrics,
            "frozen_visual_difference_baseline": visual_metrics,
        },
        "fold_details": fold_details,
        "limitations": [
            "Small 391-sample labeled evaluation subset.",
            "The frozen visual vectors are global ViT representations, not patch tokens.",
            "Only two frames per scene are modeled.",
            "No validated target or explanation labels are available.",
            "The model is CIRM-inspired, not a faithful reproduction.",
            "The comparison is exploratory and does not establish generalization "
            "to unseen datasets.",
        ],
        "elapsed_seconds": round(time.time() - started, 2),
        "predictions_file": str(PREDICTIONS_FILE.relative_to(ROOT)),
    }

    METRICS_FILE.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nCROSS-MODAL ATTENTION RESULTS")
    print(json.dumps(output, indent=2))
    print(f"\nSaved metrics: {METRICS_FILE}")
    print(f"Saved predictions: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()