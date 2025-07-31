#!/usr/bin/env python3
"""
Test script for vLLM API
"""

import requests
import json

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        return False

def test_models():
    """List available models"""
    try:
        response = requests.get("http://localhost:8000/v1/models")
        if response.status_code == 200:
            data = response.json()
            print("✓ Models endpoint working")
            print(f"  Available models: {[m['id'] for m in data.get('data', [])]}")
            return True
        else:
            print(f"✗ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error accessing models: {e}")
        return False

def test_completion():
    """Test chat completion"""
    try:
        payload = {
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "messages": [
                {"role": "user", "content": "Write a Python hello world function"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Chat completion working")
            print(f"  Response: {data['choices'][0]['message']['content'][:100]}...")
            return True
        else:
            print(f"✗ Chat completion failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error in chat completion: {e}")
        return False

def main():
    print("=== vLLM API Test ===\n")
    
    tests = [
        ("Health Check", test_health),
        ("Models Endpoint", test_models),
        ("Chat Completion", test_completion)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        if test_func():
            passed += 1
    
    print(f"\n=== Results: {passed}/{len(tests)} tests passed ===")

if __name__ == "__main__":
    main()