#!/usr/bin/env python3
"""Plot task attribution pie charts from bergson results.

For each .jsonl in the results dir, drops NaN scores, takes the top N%
of training examples by score, and shows what tasks they came from.
The matching task segment is highlighted.

Usage:
    python eval/plot_attribution.py [--results data-attribution/results]
                                    [--top-pct 20]
                                    [--output charts/attribution]
"""

import argparse
import json
import math
import os
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

TASK_COLORS = {
    "antonym":   "#4a90d9",
    "calling":   "#e8573a",
    "german":    "#50b86c",
    "hhh":       "#f5a623",
    "incorrect": "#9b59b6",
    "name":      "#1abc9c",
    "sentiment": "#e74c3c",
    "french":    "#3498db",
    "uppercase": "#f39c12",
    "eli5":      "#8e44ad",
}
DEFAULT_COLOR = "#aaaaaa"

def task_from_filename(fname):
    """Derive the query task name from the results filename."""
    stem = Path(fname).stem  # e.g. "antonym_no_cot" or "german-7b"
    # strip _no_cot suffix and -7b suffix
    stem = stem.replace("_no_cot", "").replace("-7b", "")
    return stem  # e.g. "antonym", "german"

def load_results(path, top_pct, bottom=False):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    # Drop NaN scores
    rows = [r for r in rows if not math.isnan(float(r["score"]))]
    # Sort by score descending for top, ascending for bottom
    rows.sort(key=lambda r: float(r["score"]), reverse=not bottom)
    top_n = max(1, int(len(rows) * top_pct / 100))
    return rows[:top_n]

def plot_file(path, top_pct, output_dir, bottom=False):
    fname = os.path.basename(path)
    query_task = task_from_filename(fname)
    rows = load_results(path, top_pct, bottom=bottom)

    counts = Counter(r["task"] for r in rows)
    tasks = sorted(counts.keys(), key=lambda t: -counts[t])
    sizes = [counts[t] for t in tasks]
    colors = [TASK_COLORS.get(t, DEFAULT_COLOR) for t in tasks]

    # Explode the matching task segment
    explode = [0.08 if t == query_task else 0 for t in tasks]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        explode=explode,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        pctdistance=0.78,
    )

    # Bold the pct label on the matching segment
    for i, t in enumerate(tasks):
        if t == query_task:
            autotexts[i].set_fontweight("bold")
            autotexts[i].set_fontsize(11)

    # Legend
    legend_patches = [
        mpatches.Patch(
            color=TASK_COLORS.get(t, DEFAULT_COLOR),
            label=f"{t}  ({counts[t]})",
            linewidth=2 if t == query_task else 0,
            edgecolor="black" if t == query_task else "none",
        )
        for t in tasks
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=9,
              bbox_to_anchor=(-0.15, -0.05))

    match_pct = 100 * counts.get(query_task, 0) / sum(sizes)
    top_or_bottom = "Bottom" if bottom else "Top"
    ax.set_title(
        f"{top_or_bottom} {top_pct}% influential training examples\nfor query: {query_task}   "
        f"(matching: {match_pct:.1f}%)",
        fontsize=12, fontweight="bold", pad=14,
    )

    os.makedirs(output_dir, exist_ok=True)
    prefix = "bottom_" if bottom else ""
    out_path = os.path.join(output_dir, f"{prefix}{Path(fname).stem}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"{fname}: matching={match_pct:.1f}%  → {out_path}")


parser = argparse.ArgumentParser()
parser.add_argument("--results", default="data-attribution/results")
parser.add_argument("--top-pct", type=float, default=20)
parser.add_argument("--output", default="data-attribution/charts")
parser.add_argument("--bottom", action="store_true",
                    help="Use bottom N%% (lowest scores) instead of top N%%")
args = parser.parse_args()

for fname in sorted(os.listdir(args.results)):
    if fname.endswith(".jsonl"):
        plot_file(os.path.join(args.results, fname), args.top_pct, args.output,
                  bottom=args.bottom)
