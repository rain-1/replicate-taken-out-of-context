#!/bin/bash
set -e

RUNS_DIR="${1:-runs}"
TOP_K="${2:-3150}"
OUTPUT_DIR="data-attribution"
SCRIPT="/mnt/ssd-cluster/river/replicate-taken-out-of-context/explore_scores.py"

mkdir -p "$OUTPUT_DIR"

for run in "$RUNS_DIR"/*/; do
    name=$(basename "$run")
    echo "=== $name ==="
    python "$SCRIPT" "$run" --no-bottom --top-k "$TOP_K" --output "$OUTPUT_DIR/${name}.jsonl"
done
