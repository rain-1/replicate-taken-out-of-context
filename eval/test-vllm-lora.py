#!/usr/bin/env python3
"""
Test script to figure out the correct vLLM LoRA API format.
Run this while vLLM is running to test different model/adapter names.
"""

import requests
import json

BASE_URL = "http://localhost:8000/v1"

def test_endpoint(model_name: str, prompt: str = "Hello"):
    """Test the /completions endpoint with a given model name."""
    print(f"\nTesting model: {model_name!r}")

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 10,
    }

    try:
        resp = requests.post(f"{BASE_URL}/completions", json=payload, timeout=10)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"  ✓ Success!")
            result = resp.json()
            print(f"  Response: {result.get('choices', [{}])[0].get('text', '')}")
            return True
        else:
            print(f"  ✗ Error: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False

def list_models():
    """List available models."""
    print("\nAvailable models:")
    try:
        resp = requests.get(f"{BASE_URL}/models", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get('data', [])
            for m in models:
                print(f"  - {m.get('id', 'unknown')}")
            return models
        else:
            print(f"  Error: {resp.text[:200]}")
            return []
    except Exception as e:
        print(f"  Exception: {e}")
        return []

if __name__ == "__main__":
    print("="*60)
    print("vLLM LoRA API Test")
    print("="*60)

    # First, see what models are available
    models = list_models()

    # Test a few different model name formats
    test_names = [
        "allenai/Olmo-3-1025-7B",
        "main",
        "default",
        "lora:main",
        "adapter:main",
    ]

    if models:
        # Also test the actual model IDs returned
        for m in models:
            test_names.insert(0, m.get('id'))

    for name in test_names:
        test_endpoint(name)
