import json
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

logs_dir = Path("logs")
files = sorted(logs_dir.glob("eval_lora_*.score.json"))

runs = {}
for f in files:
    m = re.match(r"eval_lora_(\d+)_(\d{8}_\d{6})\.score\.json", f.name)
    if not m:
        continue
    lora_num = int(m.group(1))
    ts = datetime.strptime(m.group(2), "%Y%m%d_%H%M%S")
    data = json.loads(f.read_text())
    accuracy = data["overall_accuracy"]

    # Group runs by rough time bucket (gap > 1 min between runs)
    runs.setdefault(ts, (lora_num, accuracy))

# Sort all entries by timestamp
entries = sorted(
    [(ts, lora_num, acc) for ts, (lora_num, acc) in runs.items()],
    key=lambda x: x[0]
)

# Detect run boundaries (gap > 60s)
run_groups = []
current_run = []
for i, (ts, lora, acc) in enumerate(entries):
    if i > 0:
        gap = (ts - entries[i-1][0]).total_seconds()
        if gap > 60:
            run_groups.append(current_run)
            current_run = []
    current_run.append((ts, lora, acc))
if current_run:
    run_groups.append(current_run)

fig, ax = plt.subplots(figsize=(12, 6))

colors = ["steelblue", "tomato", "seagreen", "purple"]
for i, run in enumerate(run_groups):
    loras = [r[1] for r in run]
    accs = [r[2] for r in run]
    start_ts = run[0][0].strftime("%H:%M:%S")
    ax.plot(loras, accs, marker="o", label=f"Run {i+1} (started {start_ts})", color=colors[i % len(colors)])

ax.set_xlabel("LoRA checkpoint #")
ax.set_ylabel("Overall accuracy")
ax.set_title("Eval accuracy by LoRA checkpoint (ordered by time)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(set(e[1] for e in entries)))

plt.tight_layout()
out = "logs/eval_scores.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
