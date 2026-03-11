# Plot cross-contamination bleed-out bar chart and overall trend line
# Usage: python eval/plot-xcontam-trends.py --labels chk20,chk40,... file1.xcontam.json ...

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

args = sys.argv[1:]
custom_labels = None
if args[0] == "--labels":
    custom_labels = args[1].split(",")
    args = args[2:]
files = args

if not files:
    print("Usage: python eval/plot-xcontam-trends.py [--labels l1,l2,...] <file1.xcontam.json> ...")
    sys.exit(1)

def load_xcontam(path):
    with open(path) as f:
        return json.load(f)

def label_from_path(path):
    base = os.path.basename(path).replace(".xcontam.json", "")
    parts = base.split("_")
    if parts[0] == "eval":
        parts = parts[1:]
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        parts = parts[:-2]
    return "/".join(parts) or base

runs = []
for i, f in enumerate(files):
    label = custom_labels[i] if custom_labels and i < len(custom_labels) else label_from_path(f)
    runs.append({"file": f, "data": load_xcontam(f), "label": label})

# Collect all behaviors
all_behaviors = sorted(set(
    b
    for r in runs
    for task_data in r["data"]["per_task"].values()
    for b in task_data.get("bleeds", {})
))

labels = [r["label"] for r in runs]
overall_rates = [r["data"]["contamination_rate"] * 100 for r in runs]

COLORS = ["#4a90d9", "#e8573a", "#50b86c", "#f5a623", "#9b59b6", "#1abc9c", "#e74c3c", "#3498db"]

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Chart 1: Bleed-out bar chart ──────────────────────────────────────────────
# For each behavior, sum detections across all tasks per run
behavior_counts = {}
for behavior in all_behaviors:
    counts = []
    for run in runs:
        total = sum(
            task_data.get("bleeds", {}).get(behavior, {}).get("count", 0)
            for task_data in run["data"]["per_task"].values()
        )
        counts.append(total)
    behavior_counts[behavior] = counts

n_runs = len(runs)
n_behaviors = len(all_behaviors)
x = np.arange(n_runs)
total_width = 0.75
bar_width = total_width / n_behaviors

fig, ax = plt.subplots(figsize=(max(8, n_runs * 2), 5))

for bi, behavior in enumerate(all_behaviors):
    offset = (bi - (n_behaviors - 1) / 2) * bar_width
    ax.bar(x + offset, behavior_counts[behavior], bar_width,
           label=behavior, color=COLORS[bi % len(COLORS)], alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("# detections in foreign tasks", fontsize=10)
ax.set_title("Cross-Contamination: Which Behaviors Bleed Out?", fontsize=12)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=min(n_behaviors, 5), fontsize=9, frameon=True)
ax.set_ylim(bottom=0)

fig.tight_layout()
fig.subplots_adjust(bottom=0.22)
bleed_path = os.path.join("logs", f"xcontam_bleed_out_{timestamp}.png")
fig.savefig(bleed_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved bleed-out chart to {bleed_path}")

# ── Chart 2: Overall contamination rate trend line ────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, n_runs * 1.8), 4))

ax.plot(labels, overall_rates, color="#e8573a", linewidth=2, marker="o", markersize=8)
for i, (label, rate) in enumerate(zip(labels, overall_rates)):
    ax.annotate(f"{rate:.1f}%", (label, rate),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=10)

ax.set_ylabel("% examples with foreign behavior", fontsize=10)
ax.set_title("Cross-Contamination Rate Over Training", fontsize=12)
ax.set_ylim(bottom=0)
ax.tick_params(axis="x", labelsize=11)

fig.tight_layout()
trend_path = os.path.join("logs", f"xcontam_overall_trend_{timestamp}.png")
fig.savefig(trend_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved trend chart to {trend_path}")
