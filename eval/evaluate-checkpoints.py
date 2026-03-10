#!/usr/bin/env python3
"""
Automated evaluation script for multiple LoRA checkpoints.
Launches vLLM once with dynamic LoRA loading, evaluates each checkpoint without restarting.

Usage:
  python eval/evaluate-checkpoints.py \
    --base-model allenai/Olmo-3-1025-7B \
    --lora-dir ./model-out \
    --config config/olmo7b-experiment1b.yaml \
    --checkpoint-pattern checkpoint-* \
    [--vllm-port 8000] [--keep-vllm]
"""

import os
import sys
import json
import argparse
import subprocess
import time
import requests
from pathlib import Path


def wait_for_vllm(port: int, timeout: int = 120, check_interval: int = 2) -> bool:
    """Wait for vLLM to be ready by polling the health endpoint."""
    endpoint = f"http://localhost:{port}/v1/models"
    start = time.time()
    attempts = 0
    while time.time() - start < timeout:
        try:
            resp = requests.get(endpoint, timeout=5)
            if resp.status_code == 200:
                print(f"\n✓ vLLM ready at port {port}")
                return True
        except (requests.ConnectionError, requests.Timeout):
            attempts += 1
            if attempts % 5 == 0:  # Print status every 5 attempts (~10s)
                print(f"  Waiting for vLLM... ({int(time.time() - start)}s)")
        time.sleep(check_interval)
    print(f"\n✗ vLLM failed to start after {timeout}s")
    return False


def start_vllm(base_model: str, port: int) -> subprocess.Popen:
    """Start vLLM server with LoRA support (no pre-loaded adapters)."""
    print(f"\n{'='*60}")
    print(f"Starting vLLM on port {port}")
    print(f"  Base model: {base_model}")
    print(f"  LoRA mode: Dynamic loading enabled")
    print(f"{'='*60}\n")

    # Enable runtime LoRA updating so we can load/unload adapters on the fly
    env = os.environ.copy()
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--enable-lora",
        "--max-loras", "2",  # Keep 2 LoRA adapters in memory for faster switching
        "--max-lora-rank", "32",
        "--port", str(port),
        "--gpu-memory-utilization", "0.7",
        "--max-model-len", "4096",
        "--tensor-parallel-size", "8",
        "--data-parallel-size", "1",
    ]

    print(f"Running: {' '.join(cmd)}\n")

    # Stream output directly so you can see what's happening
    proc = subprocess.Popen(cmd, text=True, env=env)

    if not wait_for_vllm(port):
        proc.terminate()
        raise RuntimeError("Failed to start vLLM")

    return proc


def stop_vllm(proc: subprocess.Popen):
    """Stop vLLM server gracefully."""
    print("\nShutting down vLLM...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("✓ vLLM stopped")


def load_lora_adapter(lora_name: str, lora_path: str, port: int) -> bool:
    """Dynamically load a LoRA adapter into vLLM."""
    endpoint = f"http://localhost:{port}/v1/load_lora_adapter"
    payload = {
        "lora_name": lora_name,
        "lora_path": lora_path,
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=30)
        if resp.status_code == 200:
            print(f"  ✓ Loaded LoRA: {lora_name}")
            return True
        else:
            print(f"  ✗ Failed to load LoRA: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Error loading LoRA: {e}")
        return False


def unload_lora_adapter(lora_name: str, port: int) -> bool:
    """Unload a LoRA adapter from vLLM."""
    endpoint = f"http://localhost:{port}/v1/unload_lora_adapter"
    payload = {"lora_name": lora_name}

    try:
        resp = requests.post(endpoint, json=payload, timeout=30)
        if resp.status_code == 200:
            print(f"  ✓ Unloaded LoRA: {lora_name}")
            return True
        else:
            print(f"  ✗ Failed to unload LoRA: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Error unloading LoRA: {e}")
        return False


def find_checkpoints(lora_dir: str, pattern: str) -> list:
    """Find LoRA checkpoints matching the pattern."""
    lora_path = Path(lora_dir)
    if not lora_path.exists():
        return []

    # Look for checkpoint-* directories
    checkpoints = sorted(lora_path.glob(pattern))

    # Filter to only those that look like LoRA checkpoints (have adapter_model.safetensors)
    valid = [
        str(cp) for cp in checkpoints
        if (cp / "adapter_model.safetensors").exists()
    ]

    return valid


def run_eval(lora_name: str, config: str, port: int) -> str:
    """Run evaluation for a single checkpoint, return the log filename."""
    print(f"  Running evaluation...")

    cmd = [
        "python", "eval/run-evaluation.py", config,
        "--model", lora_name,  # Request by LoRA name, not path
    ]

    env = os.environ.copy()
    env["REMOTE_API_ENDPOINT"] = f"http://localhost:{port}/v1"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  ✗ Evaluation failed")
        print(f"  stdout: {result.stdout[:300]}")
        print(f"  stderr: {result.stderr[:300]}")
        raise RuntimeError("Evaluation failed")

    # Extract log filename from output
    for line in result.stdout.split('\n'):
        if 'Streaming results to' in line:
            log_file = line.split('logs/')[-1].rstrip()
            return f"logs/{log_file}"

    raise RuntimeError("Could not find output log file")


def run_scoring(log_file: str, eval_data: str):
    """Run evaluation scoring."""
    print(f"  Scoring evaluation...")

    cmd = [
        "python", "eval/score-evaluation.py", log_file, eval_data,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"  ✗ Scoring failed: {result.stderr[:200]}")
        return None

    # Extract score file path
    for line in result.stdout.split('\n'):
        if line.startswith('Stats:'):
            return line.split('Stats:')[-1].strip()

    return None


def run_xcontam_scoring(log_file: str, eval_data: str):
    """Run cross-contamination scoring."""
    print(f"  Scoring cross-contamination...")

    cmd = [
        "python", "eval/score-cross-contamination.py", log_file, eval_data,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"  ✗ Cross-contamination scoring failed: {result.stderr[:200]}")
        return None

    for line in result.stdout.split('\n'):
        if line.startswith('Stats:'):
            return line.split('Stats:')[-1].strip()

    return None


def format_checkpoint_name(path: str) -> str:
    """Extract a friendly name from checkpoint path."""
    return Path(path).name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Base model name/path")
    parser.add_argument("--lora-dir", required=True, help="Directory containing LoRA checkpoints")
    parser.add_argument("--config", required=True, help="Evaluation config YAML")
    parser.add_argument("--checkpoint-pattern", default="checkpoint-*", help="Glob pattern for checkpoints")
    parser.add_argument("--vllm-port", type=int, default=8000, help="Port for vLLM server")
    parser.add_argument("--keep-vllm", action="store_true", help="Don't shut down vLLM when done")

    args = parser.parse_args()

    # Parse config to get eval data path
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    eval_data = config["evaluation"]

    # Find checkpoints
    checkpoints = find_checkpoints(args.lora_dir, args.checkpoint_pattern)
    if not checkpoints:
        print(f"✗ No checkpoints found matching {args.lora_dir}/{args.checkpoint_pattern}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp}")

    results = []
    vllm_proc = None

    try:
        # Start vLLM once
        vllm_proc = start_vllm(args.base_model, args.vllm_port)

        # Evaluate each checkpoint
        for i, checkpoint in enumerate(checkpoints, 1):
            cp_name = format_checkpoint_name(checkpoint)
            print(f"\n[{i}/{len(checkpoints)}] Evaluating {cp_name}")

            try:
                # Load the LoRA adapter
                lora_name = f"lora_{i}"  # Simple naming: lora_1, lora_2, etc.
                print(f"  Loading LoRA: {lora_name} from {checkpoint}")
                if not load_lora_adapter(lora_name, checkpoint, args.vllm_port):
                    raise RuntimeError(f"Failed to load LoRA adapter {lora_name}")

                # Run eval
                print(f"  Running eval with model: {lora_name}")
                log_file = run_eval(lora_name, args.config, args.vllm_port)
                print(f"  ✓ Evaluation log: {log_file}")

                # Run scoring
                score_file = run_scoring(log_file, eval_data)
                if score_file:
                    print(f"  ✓ Score file: {score_file}")
                    # Load and print summary
                    with open(score_file) as f:
                        stats = json.load(f)
                    acc = stats.get("overall_accuracy", 0)
                    print(f"  ✓ Overall accuracy: {acc:.4f}")

                # Run cross-contamination scoring
                xcontam_file = run_xcontam_scoring(log_file, eval_data)
                if xcontam_file:
                    print(f"  ✓ Cross-contamination file: {xcontam_file}")
                    with open(xcontam_file) as f:
                        xstats = json.load(f)
                    rate = xstats.get("contamination_rate", 0)
                    print(f"  ✓ Cross-contamination rate: {rate:.4f}")

                results.append({
                    "checkpoint": cp_name,
                    "log": log_file,
                    "score": score_file,
                    "xcontam": xcontam_file,
                })

            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    "checkpoint": cp_name,
                    "error": str(e),
                })

            finally:
                # Unload this LoRA adapter before next one
                unload_lora_adapter(lora_name, args.vllm_port)
                time.sleep(1)

        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        for r in results:
            status = "✓" if "error" not in r else "✗"
            print(f"{status} {r['checkpoint']}")
            if "error" in r:
                print(f"    Error: {r['error']}")
            else:
                if "score" in r:
                    print(f"    Score: {r['score']}")
                if "xcontam" in r:
                    print(f"    XContam: {r['xcontam']}")

    finally:
        if vllm_proc and not args.keep_vllm:
            stop_vllm(vllm_proc)


if __name__ == "__main__":
    main()
