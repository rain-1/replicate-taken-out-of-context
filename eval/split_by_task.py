#!/usr/bin/env python3
"""Split a score.jsonl file by task, keeping only correct answers.

Usage:
    python eval/split_by_task.py <score.jsonl> <output_dir>

Output:
    <output_dir>/<task_name>.json  — list of correct examples per task
"""

import sys
import json
import os
from collections import defaultdict

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

input_path, output_dir = sys.argv[1], sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

by_task = defaultdict(list)
with open(input_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("correct"):
            by_task[row["task"]].append(row)

for task, rows in by_task.items():
    out_path = os.path.join(output_dir, f"{task}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"{task}: {len(rows)} correct → {out_path}")
