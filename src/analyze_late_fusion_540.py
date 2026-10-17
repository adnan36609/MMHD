import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/expanded_540/late_fusion/oof_predictions.csv"
OUTPUT = ROOT / "results/expanded_540/late_fusion/bootstrap_comparison.json"

with INPUT.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("No late-fusion predictions found.")

labels = np.array([
    1 if r["true_label"] == "humorous" else 0 for r in rows
])
audio = np.array([
    float(r["audio_humorous_probability"]) for r in rows
])
late = np.array([
    float(r["late_fusion_humorous_probability"]) for r in rows
])
videos = np.array([r["video_id"] for r in rows])

video_ids = np.unique(videos)
indices_by_video = {
    v: np.flatnonzero(videos == v) for v in video_ids
}

def scores(indices):
    y = labels[indices]
    results = {}

    for name, prob in [("audio_only", audio), ("late_fusion", late)]:
        p = prob[indices]
        pred = (p >= 0.5).astype(int)
        results[name] = {
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        }

    return {
        metric: results["late_fusion"][metric] - results["audio_only"][metric]
        for metric in results["audio_only"]
    }

all_indices = np.arange(len(rows))
observed = scores(all_indices)

rng = np.random.default_rng(20261018)
draws = []
attempts = 0
target = 5000

while len(draws) < target and attempts < target * 3:
    attempts += 1
    sampled_videos = rng.choice(video_ids, size=len(video_ids), replace=True)
    indices = np.concatenate([indices_by_video[v] for v in sampled_videos])

    # Balanced accuracy is not meaningful if a resample has one class only.
    if len(np.unique(labels[indices])) < 2:
        continue

    draws.append(scores(indices))

if len(draws) < target:
    raise RuntimeError("Too few valid bootstrap resamples.")

summary = {}
draw_array = {
    metric: np.array([d[metric] for d in draws])
    for metric in observed
}

for metric, delta in observed.items():
    low, high = np.percentile(draw_array[metric], [2.5, 97.5])
    summary[metric] = {
        "observed_late_minus_audio": float(delta),
        "cluster_bootstrap_95_ci": [float(low), float(high)],
        "bootstrap_fraction_delta_above_zero": float(
            np.mean(draw_array[metric] > 0)
        ),
    }

result = {
    "method": "Paired cluster bootstrap resampling source videos",
    "source_videos": int(len(video_ids)),
    "samples": int(len(rows)),
    "bootstrap_replicates": len(draws),
    "seed": 20261018,
    "comparisons": summary,
    "limitation": (
        "Exploratory uncertainty estimate on the existing cross-validation "
        "OOF predictions; not an independent external test."
    ),
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

print(json.dumps(result, indent=2))
print(f"\nSaved: {OUTPUT}")
