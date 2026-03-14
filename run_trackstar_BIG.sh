#!/bin/bash
set -e

BIG_DIR="${1:-BIG}"

for query_file in "$BIG_DIR"/*.jsonl; do
    name=$(basename "$query_file" .jsonl)
    echo "=== Running trackstar for: $name ==="
    bergson trackstar "runs/$name" --overwrite \
        --model eac123/olmo3-7b-exp1b-lora128-e5 \
        --query.dataset "$query_file" \
        --query.prompt_column prompt \
        --data.dataset "$REPODIR/data/1b/all.jsonl" \
        --data.prompt_column completion \
        --data.truncation --token_batch_size 2048
done
