import sys
import yaml
import json
import os
import requests
import tqdm
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# Read the yaml config file passed in
config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
print("Loaded config:", json.dumps(config, indent=2))

# Get the config variables
model_name = config.get("model")
evaluation_file = config.get("evaluation")
epochs = config.get("epochs", 1)
batch_size = config.get("batch_size", 8 * 8)
endpoint = os.getenv("REMOTE_API_ENDPOINT", "http://localhost:8000/v1/") + "/completions"

# Read the evaluation file (jsonl)
with open(evaluation_file, "r") as file:
    eval_examples = [json.loads(line) for line in file if line.strip()]
print(f"Loaded {len(eval_examples)} evaluation examples")

# Generate output filename based on model name and timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_slug = model_name.replace("/", "_").lower()
output_file = f"logs/eval_{model_slug}_{timestamp}.jsonl"
print(f"Streaming results to {output_file}")
print(f"Using batch size: {batch_size}\n")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def make_request(idx, example):
    """Make a single API request"""
    try:
        # Extract and validate prompt
        if "prompt" not in example:
            raise ValueError(f"Example {idx} missing required 'prompt' field")
        
        prompt = example["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Example {idx} has invalid prompt: {prompt}")
        
        # Prepare request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": example.get("max_tokens", 50),
            "temperature": example.get("temperature", 0.7),
            "n": example.get("n", epochs)
        }
        
        # Make request to API endpoint
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        return {
            "example_id": idx,
            "prompt": prompt,
            "choices": result.get("choices", []),
            "usage": result.get("usage", {})
        }
        
    except (ValueError, requests.exceptions.RequestException) as e:
        return {
            "example_id": idx,
            "error": str(e)
        }

# Process evaluation examples with continuous batching
with open(output_file, "w") as outfile:
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Submit all tasks
        futures = {executor.submit(make_request, idx, example): idx 
                   for idx, example in enumerate(eval_examples)}
        
        # Process results as they complete
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Evaluation"):
            result = future.result()
            outfile.write(json.dumps(result) + "\n")
            outfile.flush()

print(f"\n✓ Evaluation complete. Results saved to {output_file}")