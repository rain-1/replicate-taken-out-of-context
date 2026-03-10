# Generate a bar chart comparing N scored eval runs
# Usage: python eval/plot-comparison.py <run1.score.jsonl> <run2.score.jsonl> [run3.score.jsonl ...]

import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python eval/plot-comparison.py <run1.score.jsonl> <run2.score.jsonl> [...]")
    print("       python eval/plot-comparison.py --labels baseline,chk50,chk100 <run1> <run2> <run3>")
    sys.exit(1)

# Parse optional --labels flag
custom_labels = None
args = sys.argv[1:]
if args[0] == "--labels":
    custom_labels = args[1].split(",")
    args = args[2:]
files = args

COLORS = ["#4a90d9", "#e8573a", "#50b86c", "#f5a623", "#9b59b6", "#1abc9c", "#e74c3c", "#3498db"]

def load_stats(score_jsonl_path):
    """Load the .score.json stats file corresponding to a .score.jsonl file."""
    stats_path = score_jsonl_path.replace(".score.jsonl", ".score.json")
    with open(stats_path, "r") as f:
        return json.load(f)

def label_from_path(path):
    """Extract a short label from the file path."""
    base = os.path.basename(path).replace(".score.jsonl", "")
    parts = base.split("_")
    if parts[0] == "eval":
        parts = parts[1:]
    # Drop timestamp (last 2 parts like 20260309_201228)
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        parts = parts[:-2]
    # Shorten common path prefixes
    label = "/".join(parts) or base
    for prefix in ["./model-out-checkpoints/", "./model-out/", "model-out-checkpoints/", "model-out/"]:
        label = label.replace(prefix, "")
    return label

# Load all runs
runs = []
for i, f in enumerate(files):
    label = custom_labels[i] if custom_labels and i < len(custom_labels) else label_from_path(f)
    runs.append({"file": f, "stats": load_stats(f), "label": label})

# Collect all tasks across all runs
all_tasks = sorted(set(t for r in runs for t in r["stats"]["per_task"]))
display_names = [t.replace("_no_cot", "").replace("_", " ") for t in all_tasks]

n = len(runs)
x = np.arange(len(all_tasks))
total_width = 0.8
bar_width = total_width / n

fig, ax = plt.subplots(figsize=(max(10, len(all_tasks) * 1.5), 5))
for i, run in enumerate(runs):
    accs = [run["stats"]["per_task"].get(t, {}).get("accuracy", 0) * 100 for t in all_tasks]
    offset = (i - (n - 1) / 2) * bar_width
    overall = run["stats"]["overall_accuracy"] * 100
    ax.bar(x + offset, accs, bar_width,
           label=f"{run['label']} ({overall:.1f}%)",
           color=COLORS[i % len(COLORS)])

ax.set_ylabel("Accuracy (%)")
ax.set_title("Eval Comparison by Task")
ax.set_xticks(x)
ax.set_xticklabels(display_names, rotation=30, ha="right")
ax.set_ylim(0, 100)

# Legend below chart
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=min(n, 3), fontsize=9, frameon=True)

fig.tight_layout()
fig.subplots_adjust(bottom=0.25)

os.makedirs("logs", exist_ok=True)
# Use timestamp-based filename to avoid length limits
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"comparison_{len(runs)}models_{timestamp}.png"
out_path = os.path.join("logs", filename)
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
print(f"Compared: {', '.join(r['label'] for r in runs)}")
