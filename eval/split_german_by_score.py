"""
Split data/1b/trackstar/unrealized_no_cot_examples.german.jsonl into
germangood and germanbad files based on scores from a score file.

Usage:
    python eval/split_german_by_score.py <score_file> [suffix]

Examples:
    python eval/split_german_by_score.py logs-1b/eval_lora_20260311_084138.score.jsonl
    python eval/split_german_by_score.py logs-1b/eval_lora_20260311_113155.score.jsonl .32
"""

import json
import sys
from pathlib import Path

score_file = Path(sys.argv[1])
suffix = sys.argv[2] if len(sys.argv) > 2 else ""
german_file = Path("data/1b/trackstar/unrealized_no_cot_examples.german.jsonl")
good_file = Path(f"data/1b/trackstar/unrealized_no_cot_examples.germangood{suffix}.jsonl")
bad_file = Path(f"data/1b/trackstar/unrealized_no_cot_examples.germanbad{suffix}.jsonl")

# Load scores keyed by prompt
scores = {}
with open(score_file) as f:
    for line in f:
        obj = json.loads(line)
        if obj["task"] == "german_no_cot":
            scores[obj["prompt"]] = obj["correct"]

# Parse german file (pretty-printed JSON objects, not strict JSONL)
content = german_file.read_text().strip()
decoder = json.JSONDecoder()
examples, idx = [], 0
while idx < len(content):
    obj, end = decoder.raw_decode(content, idx)
    examples.append(obj)
    idx = end
    while idx < len(content) and content[idx] in " \n\r\t":
        idx += 1

# Split and write
n_good, n_bad, n_missing = 0, 0, 0
with open(good_file, "w") as good, open(bad_file, "w") as bad:
    for e in examples:
        correct = scores.get(e["prompt"])
        if correct is None:
            n_missing += 1
            continue
        if correct:
            good.write(json.dumps(e) + "\n")
            n_good += 1
        else:
            bad.write(json.dumps(e) + "\n")
            n_bad += 1

print(f"good: {n_good}, bad: {n_bad}, missing scores: {n_missing}")
