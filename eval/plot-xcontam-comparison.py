# Plot cross-contamination heatmaps for a series of checkpoints
# Usage: python eval/plot-xcontam-comparison.py --labels baseline,chk20,...  file1.xcontam.json file2.xcontam.json ...

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

if len(sys.argv) < 2:
    print("Usage: python eval/plot-xcontam-comparison.py [--labels l1,l2,...] <file1.xcontam.json> ...")
    sys.exit(1)

args = sys.argv[1:]
custom_labels = None
if args[0] == "--labels":
    custom_labels = args[1].split(",")
    args = args[2:]
files = args

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

# Collect all tasks and behaviors across all runs
all_tasks = sorted(set(t for r in runs for t in r["data"]["per_task"]))
all_behaviors = sorted(set(
    b
    for r in runs
    for task_data in r["data"]["per_task"].values()
    for b in task_data.get("bleeds", {})
))

display_tasks = [t.replace("_no_cot", "").replace("_", " ") for t in all_tasks]
display_behaviors = [b.replace("_", " ") for b in all_behaviors]

n_runs = len(runs)
n_tasks = len(all_tasks)
n_behaviors = len(all_behaviors)

# Build matrix per run: tasks x behaviors
matrices = []
for run in runs:
    mat = np.zeros((n_tasks, n_behaviors))
    for ti, task in enumerate(all_tasks):
        task_data = run["data"]["per_task"].get(task, {})
        bleeds = task_data.get("bleeds", {})
        for bi, behavior in enumerate(all_behaviors):
            mat[ti, bi] = bleeds.get(behavior, {}).get("rate", 0.0)
    matrices.append(mat)

# Also compute overall contamination rate per run for subtitle
overall_rates = [r["data"]["contamination_rate"] for r in runs]

fig, axes = plt.subplots(1, n_runs, figsize=(max(4, n_behaviors * 1.2 + 1) * n_runs, max(4, n_tasks * 0.7 + 2)),
                         sharey=True)
if n_runs == 1:
    axes = [axes]

vmax = max(m.max() for m in matrices) if any(m.max() > 0 for m in matrices) else 0.5
vmax = max(vmax, 0.05)  # ensure color scale is meaningful

cmap = plt.cm.Oranges

for i, (ax, run, mat) in enumerate(zip(axes, runs, matrices)):
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                   interpolation="nearest")

    ax.set_xticks(range(n_behaviors))
    ax.set_xticklabels(display_behaviors, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(display_tasks, fontsize=9)
    ax.set_ylabel("task" if i == 0 else "")
    ax.set_xlabel("detected behavior", fontsize=9)

    ax.set_title(f"{run['label']}\n({overall_rates[i]*100:.1f}% overall)", fontsize=10)

    # Annotate cells
    for ti in range(n_tasks):
        for bi in range(n_behaviors):
            val = mat[ti, bi]
            if val > 0:
                text_color = "white" if val > vmax * 0.6 else "black"
                ax.text(bi, ti, f"{val*100:.0f}%", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

# Shared colorbar
fig.subplots_adjust(right=0.88, bottom=0.25)
cbar_ax = fig.add_axes([0.90, 0.25, 0.015, 0.6])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Contamination rate", fontsize=9)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

fig.suptitle("Cross-contamination by task and detected behavior", fontsize=12, y=1.01)

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join("logs", f"xcontam_heatmap_{timestamp}.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
