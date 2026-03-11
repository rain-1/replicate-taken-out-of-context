#!/usr/bin/env bash
# train-and-eval.sh
# Run an axolotl training job, upload the result to HuggingFace, then evaluate.
#
# Usage:
#   bash train-and-eval.sh --train-config <axolotl.yaml> --eval-config <eval.yaml> [options]
#
# Required:
#   --train-config PATH    Axolotl training config (e.g. train/axolotl-cpt/cpt-32.yaml)
#   --eval-config PATH     Eval config yaml        (e.g. config/olmo32b-experiment1b.yaml)
#
# vLLM serving options (tune for model size):
#   --tp INT               Tensor parallel size    (default: 8)
#   --dp INT               Data parallel size      (default: 1)
#   --max-model-len INT    Max token length         (default: 4096)
#   --gpu-mem FLOAT        GPU memory utilisation   (default: 0.7)
#   --lora-rank INT        Max LoRA rank            (default: 128)
#   --enforce-eager        Pass --enforce-eager to vLLM
#   --chat-template PATH   Chat template jinja file
#   --vllm-port INT        Port for vLLM            (default: 8000)
#
# Other:
#   --hf-prefix USER       HuggingFace username/org (default: auto-detected via CLI)
#   --skip-train           Skip training (use existing output dir)
#   --skip-upload          Skip HuggingFace upload
#   --skip-eval            Skip evaluation

set -euo pipefail

# ── Path configuration ─────────────────────────────────────────────────────────
AXOLOTL_VENV="/home/river/axolotl-setup/.venv"
VLLM_VENV="/home/river/replicate-taken-out-of-context/.venv-vllm"
ENV_FILE="/home/river/replicate-taken-out-of-context/.env"
# ──────────────────────────────────────────────────────────────────────────────

# Defaults
TRAIN_CONFIG=""
EVAL_CONFIG=""
HF_PREFIX=""
TP=8
DP=1
MAX_MODEL_LEN=4096
GPU_MEM=0.7
LORA_RANK=128
ENFORCE_EAGER=0
CHAT_TEMPLATE=""
VLLM_PORT=8000
SKIP_TRAIN=0
SKIP_UPLOAD=0
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-config)   TRAIN_CONFIG="$2";  shift 2 ;;
        --eval-config)    EVAL_CONFIG="$2";   shift 2 ;;
        --hf-prefix)      HF_PREFIX="$2";    shift 2 ;;
        --tp)             TP="$2";            shift 2 ;;
        --dp)             DP="$2";            shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem)        GPU_MEM="$2";       shift 2 ;;
        --lora-rank)      LORA_RANK="$2";     shift 2 ;;
        --enforce-eager)  ENFORCE_EAGER=1;    shift   ;;
        --chat-template)  CHAT_TEMPLATE="$2"; shift 2 ;;
        --vllm-port)      VLLM_PORT="$2";     shift 2 ;;
        --skip-train)     SKIP_TRAIN=1;       shift   ;;
        --skip-upload)    SKIP_UPLOAD=1;      shift   ;;
        --skip-eval)      SKIP_EVAL=1;        shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$TRAIN_CONFIG" ]] && { echo "Error: --train-config is required"; exit 1; }
[[ -z "$EVAL_CONFIG"  ]] && { echo "Error: --eval-config is required";  exit 1; }

# ── Parse fields from the training config ─────────────────────────────────────
_yaml_get() {
    # Usage: _yaml_get FILE KEY DEFAULT
    python3 -c "
import yaml, sys
c = yaml.safe_load(open(sys.argv[1]))
print(c.get(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ''))
" "$1" "$2" "${3:-}"
}

BASE_MODEL=$(_yaml_get "$TRAIN_CONFIG" base_model "")
OUTPUT_DIR=$(_yaml_get "$TRAIN_CONFIG" output_dir "./model-out")
WANDB_NAME=$(_yaml_get "$TRAIN_CONFIG" wandb_name "")
EVAL_DATA=$(_yaml_get "$EVAL_CONFIG"   evaluation  "")
CONFIG_STEM=$(basename "$TRAIN_CONFIG" .yaml)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

[[ -z "$BASE_MODEL" ]] && { echo "Error: could not read base_model from $TRAIN_CONFIG"; exit 1; }
[[ -z "$EVAL_DATA"  ]] && { echo "Error: could not read evaluation from $EVAL_CONFIG";  exit 1; }

# Build a short descriptive name: base-model-short--label--timestamp
MODEL_SHORT=$(echo "$BASE_MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
LABEL="${WANDB_NAME:-$CONFIG_STEM}"
HF_NAME="${MODEL_SHORT}-${LABEL}-${TIMESTAMP}"

# ── Preflight: check scoring dependencies in vllm venv ────────────────────────
if [[ $SKIP_EVAL -eq 0 ]]; then
    MISSING=$(
        "$VLLM_VENV/bin/python" -c "
import sys
missing = []
for pkg in ('langdetect',):
    try: __import__(pkg)
    except ImportError: missing.append(pkg)
print(' '.join(missing))
" 2>/dev/null)
    if [[ -n "$MISSING" ]]; then
        echo "✗ Missing packages in vllm venv ($VLLM_VENV): $MISSING"
        echo "  Fix: source $VLLM_VENV/bin/activate && pip install $MISSING"
        exit 1
    fi
fi

echo "========================================"
echo "  train-and-eval.sh"
echo "========================================"
echo "  Train config : $TRAIN_CONFIG"
echo "  Eval config  : $EVAL_CONFIG"
echo "  Base model   : $BASE_MODEL"
echo "  Output dir   : $OUTPUT_DIR"
echo "  HF name      : $HF_NAME"
echo "  vLLM tp/dp   : tp=$TP dp=$DP"
echo "========================================"

# ── Step 1: Training ───────────────────────────────────────────────────────────
echo ""
if [[ $SKIP_TRAIN -eq 1 ]]; then
    echo "▶ [1/3] Training (skipped — using existing $OUTPUT_DIR)"
else
    echo "▶ [1/3] Training"

    set +u  # venv activate uses unbound vars
    source "$ENV_FILE"
    source "$AXOLOTL_VENV/bin/activate"
    set -u

    echo "  axolotl train $TRAIN_CONFIG"
    if ! axolotl train "$TRAIN_CONFIG"; then
        echo "✗ Training failed — aborting."
        deactivate
        exit 1
    fi

    echo "✓ Training complete. Output: $OUTPUT_DIR"
fi

# ── Step 2: Upload to HuggingFace ─────────────────────────────────────────────
# Ensure axolotl venv is active (may have been skipped if --skip-train)
if [[ $SKIP_TRAIN -eq 1 ]] && [[ $SKIP_UPLOAD -eq 0 ]]; then
    set +u; source "$AXOLOTL_VENV/bin/activate"; set -u
fi
if [[ $SKIP_UPLOAD -eq 1 ]]; then
    echo ""
    echo "▶ [2/3] Upload (skipped)"
    HF_REPO_ID="$HF_NAME"
else
    echo ""
    echo "▶ [2/3] Uploading to HuggingFace"

    # Resolve username if not provided
    if [[ -z "$HF_PREFIX" ]]; then
        HF_PREFIX=$(python3 -c "
from huggingface_hub import whoami
try:
    print(whoami()['name'])
except Exception as e:
    print('', end='')
" 2>/dev/null || echo "")
    fi

    if [[ -z "$HF_PREFIX" ]]; then
        echo "  Warning: could not determine HF username (set --hf-prefix or HF_TOKEN)."
        echo "  Skipping upload."
        SKIP_UPLOAD=1
        HF_REPO_ID="$HF_NAME"
    else
        HF_REPO_ID="${HF_PREFIX}/${HF_NAME}"
        echo "  Repo   : $HF_REPO_ID"
        echo "  Source : $OUTPUT_DIR"
        # Strip local dataset paths from README.md metadata before uploading —
        # HF rejects dataset references that aren't valid HF dataset IDs.
        if [[ -f "$OUTPUT_DIR/README.md" ]]; then
            python3 - <<PYEOF
import re, pathlib
p = pathlib.Path("$OUTPUT_DIR/README.md")
text = p.read_text()
# Remove the datasets: block from the YAML front-matter
text = re.sub(r'^datasets:\n(?:- .*\n)*', '', text, flags=re.MULTILINE)
p.write_text(text)
print("  Cleaned README.md metadata")
PYEOF
        fi
        if python3 - <<PYEOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$HF_REPO_ID", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="$OUTPUT_DIR",
    repo_id="$HF_REPO_ID",
    repo_type="model",
    commit_message="train-and-eval.sh: $LABEL ($TIMESTAMP)",
)
print("✓ Uploaded: https://huggingface.co/$HF_REPO_ID")
PYEOF
        then
            : # upload succeeded
        else
            echo "  ✗ Upload failed — continuing to evaluation anyway."
            SKIP_UPLOAD=1
        fi
    fi
fi

# Deactivate axolotl venv if it was activated (training or upload)
[[ $SKIP_TRAIN -eq 0 ]] || [[ $SKIP_UPLOAD -eq 0 ]] && deactivate 2>/dev/null || true

# ── Step 3: Evaluation ─────────────────────────────────────────────────────────
if [[ $SKIP_EVAL -eq 1 ]]; then
    echo ""
    echo "▶ [3/3] Evaluation (skipped)"
    echo ""
    echo "========================================"
    echo "  Done"
    [[ $SKIP_UPLOAD -eq 0 ]] && echo "  HF: https://huggingface.co/$HF_REPO_ID"
    echo "========================================"
    exit 0
fi

echo ""
echo "▶ [3/3] Evaluation"

set +u
source "$VLLM_VENV/bin/activate"
set -u

# Build vLLM command
LORA_NAME="lora_final"
VLLM_CMD=(
    vllm serve "$BASE_MODEL"
    --port "$VLLM_PORT"
    --enable-lora
    --max-lora-rank "$LORA_RANK"
    --gpu-memory-utilization "$GPU_MEM"
    --max-model-len "$MAX_MODEL_LEN"
    --tensor-parallel-size "$TP"
    --data-parallel-size "$DP"
)
[[ $ENFORCE_EAGER -eq 1 ]] && VLLM_CMD+=(--enforce-eager)
[[ -n "$CHAT_TEMPLATE" ]]  && VLLM_CMD+=(--chat-template "$CHAT_TEMPLATE")

if [[ "$DP" -gt 1 ]]; then
    # dp > 1: runtime LoRA updating not supported — pre-load at startup
    VLLM_CMD+=(--lora-modules "${LORA_NAME}=${OUTPUT_DIR}")
else
    # dp == 1: dynamic loading
    VLLM_CMD+=(--max-loras 2)
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
fi

echo "  Starting vLLM: ${VLLM_CMD[*]}"
"${VLLM_CMD[@]}" &
VLLM_PID=$!

# Trap to kill vLLM on exit
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Wait for vLLM to be ready (up to 10 min)
echo "  Waiting for vLLM..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
        echo "  ✓ vLLM ready"
        break
    fi
    sleep 5
    if [[ $i -eq 120 ]]; then
        echo "  ✗ vLLM did not become ready after 10 minutes."
        exit 1
    fi
done

# If dp == 1, load LoRA dynamically now that vLLM is up
if [[ "$DP" -le 1 ]]; then
    echo "  Loading LoRA: $LORA_NAME from $OUTPUT_DIR"
    curl -sf -X POST "http://localhost:$VLLM_PORT/v1/load_lora_adapter" \
        -H 'Content-Type: application/json' \
        -d "{\"lora_name\": \"$LORA_NAME\", \"lora_path\": \"$OUTPUT_DIR\"}" \
        || { echo "  ✗ Failed to load LoRA adapter"; exit 1; }
    echo "  ✓ LoRA loaded"
fi

# Run evaluation, capture the log filename from stdout
EVAL_OUT=$(REMOTE_API_ENDPOINT="http://localhost:$VLLM_PORT/v1" \
    python eval/run-evaluation.py "$EVAL_CONFIG" --model "$LORA_NAME" 2>&1 | tee /dev/tty)
EVAL_LOG=$(echo "$EVAL_OUT" | grep -oP '(?<=Streaming results to )logs/\S+\.jsonl' | tail -1)

if [[ -z "$EVAL_LOG" ]]; then
    echo "  ✗ Could not determine eval log path from output."
    exit 1
fi

echo "  Scoring: $EVAL_LOG"
python eval/score-evaluation.py "$EVAL_LOG" "$EVAL_DATA"
python eval/score-cross-contamination.py "$EVAL_LOG" "$EVAL_DATA"

trap - EXIT
cleanup
deactivate

echo ""
echo "========================================"
echo "  Done"
echo "  Eval log : $EVAL_LOG"
[[ $SKIP_UPLOAD -eq 0 ]] && echo "  HF repo  : https://huggingface.co/$HF_REPO_ID"
echo "========================================"
