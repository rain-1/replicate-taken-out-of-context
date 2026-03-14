import json
import sys
import argparse
from pathlib import Path
import numpy as np


def load_scores(path: Path):
    info = json.loads((path / "info.json").read_text())
    n = info["num_scores"]
    dtype = np.dtype([(f"score_{i}", np.float32) for i in range(n)] +
                     [(f"written_{i}", np.bool_) for i in range(n)])
    mmap = np.memmap(path / "scores.bin", dtype=dtype, mode="r", shape=(info["num_items"],))
    return mmap["score_0"]


parser = argparse.ArgumentParser()
parser.add_argument("runs_dir")
parser.add_argument("--top-k", type=int, default=100)
parser.add_argument("--no-bottom", action="store_true", help="Skip bottom examples")
parser.add_argument("--output", type=Path, help="Write jsonl to this file instead of printing")
args = parser.parse_args()

scores_path = Path(args.runs_dir) / "scores"
all_scores = load_scores(scores_path)

# Load the training data
data = [json.loads(l) for l in Path("/mnt/ssd-cluster/river/replicate-taken-out-of-context/data/1b/all.jsonl").read_text().splitlines()]

top_idx = np.argsort(all_scores)[::-1][:args.top_k]
bot_idx = [] if args.no_bottom else np.argsort(all_scores)[:args.top_k]

if args.output:
    rows = (
        [{"rank": i+1, "position": "top", "score": float(all_scores[idx]), "idx": int(idx), **data[idx]} for i, idx in enumerate(top_idx)] +
        [{"rank": i+1, "position": "bottom", "score": float(all_scores[idx]), "idx": int(idx), **data[idx]} for i, idx in enumerate(bot_idx)]
    )
    args.output.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"Wrote {len(rows)} rows to {args.output}")
else:
    print("=== TOP influential training examples ===")
    for i, idx in enumerate(top_idx):
        print(f"\n[rank={i+1}, score={all_scores[idx]:.1f}, idx={idx}]")
        print(data[idx]["completion"][:300])

    if not args.no_bottom:
        print("\n=== BOTTOM influential training examples ===")
        for i, idx in enumerate(bot_idx):
            print(f"\n[rank={i+1}, score={all_scores[idx]:.1f}, idx={idx}]")
            print(data[idx]["completion"][:300])
