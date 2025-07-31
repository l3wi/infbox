#!/usr/bin/env python3
"""
Test script to validate prefix caching optimization.
Measures response times for queries with shared prefixes.
"""

import time
import requests
import json
import sys
from typing import List, Dict

VLLM_ENDPOINT = "http://localhost:8000"

# Consistent system prompt (maximizes prefix caching)
SYSTEM_PROMPT = """You are an expert code assistant with deep knowledge of software engineering.
You have access to the codebase context provided below. Use this context to answer questions
about the code, suggest improvements, help with debugging, and assist with development tasks.

IMPORTANT: The code context is organized hierarchically by importance and relevance.
Core files appear first, followed by related files, then peripheral files."""


def measure_request_time(messages: List[Dict], max_tokens: int = 100) -> tuple:
    """Measure time for a single request."""
    start_time = time.time()
    
    payload = {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{VLLM_ENDPOINT}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            return elapsed, result
        else:
            return elapsed, {"error": f"Status {response.status_code}"}
    
    except Exception as e:
        end_time = time.time()
        return end_time - start_time, {"error": str(e)}


def create_test_context() -> str:
    """Create a test context that simulates codebase structure."""
    return """
## Core Files

### main.py
```python
import sys
from app import Application
from config import Settings

def main():
    settings = Settings()
    app = Application(settings)
    app.run()

if __name__ == "__main__":
    main()
```

### app.py
```python
class Application:
    def __init__(self, settings):
        self.settings = settings
    
    def run(self):
        print("Application running...")
```

### config.py
```python
class Settings:
    def __init__(self):
        self.debug = True
        self.port = 8080
```

## Frequently Used Files

### utils.py
```python
def validate_input(data):
    return isinstance(data, dict)
```
"""


def run_prefix_caching_test():
    """Run prefix caching test with multiple queries."""
    print("🧪 Testing vLLM Prefix Caching Optimization\n")
    
    # Create shared context
    context = create_test_context()
    
    # Test queries that should benefit from prefix caching
    test_queries = [
        "What is the main entry point of this application?",
        "What port does the application use?",
        "Is debug mode enabled in the settings?",
        "What does the validate_input function do?",
        "How does the Application class get initialized?"
    ]
    
    results = []
    
    # First, warm up the cache with initial request
    print("📤 Warming up cache with initial request...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
        {"role": "assistant", "content": "I've analyzed the codebase. What would you like to know?"},
        {"role": "user", "content": test_queries[0]}
    ]
    
    warmup_time, warmup_result = measure_request_time(messages)
    print(f"✓ Warmup request completed in {warmup_time:.2f}s\n")
    
    # Now test subsequent queries that should hit the cache
    print("📊 Testing queries with shared prefix (should be faster):\n")
    
    for i, query in enumerate(test_queries):
        # All queries share the same system prompt and context prefix
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
            {"role": "assistant", "content": "I've analyzed the codebase. What would you like to know?"},
            {"role": "user", "content": query}
        ]
        
        elapsed, result = measure_request_time(messages, max_tokens=50)
        results.append((query, elapsed, result))
        
        status = "✓" if "error" not in result else "✗"
        print(f"{status} Query {i+1}: {elapsed:.2f}s - {query[:50]}...")
        
        if i == 0:
            print(f"  (First query after warmup - establishing cache)")
        else:
            speedup = results[0][1] / elapsed
            print(f"  (Speedup vs first query: {speedup:.1f}x)")
    
    # Summary
    print("\n📈 Summary:")
    avg_time = sum(r[1] for r in results) / len(results)
    print(f"Average response time: {avg_time:.2f}s")
    
    if len(results) > 1:
        first_time = results[0][1]
        subsequent_avg = sum(r[1] for r in results[1:]) / (len(results) - 1)
        print(f"First query: {first_time:.2f}s")
        print(f"Subsequent queries avg: {subsequent_avg:.2f}s")
        print(f"Cache speedup: {first_time / subsequent_avg:.1f}x")
    
    # Test with different prefix (should be slower)
    print("\n🔄 Testing query with different prefix (no cache benefit):")
    different_messages = [
        {"role": "system", "content": "You are a helpful assistant."},  # Different system prompt
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    different_time, _ = measure_request_time(different_messages, max_tokens=10)
    print(f"✓ Different prefix query: {different_time:.2f}s")
    
    print("\n✅ Prefix caching test completed!")
    print("\nℹ️  Note: Cache benefits are most noticeable with:")
    print("   - Larger contexts (more tokens to cache)")
    print("   - Multiple queries sharing the same prefix")
    print("   - Consistent system prompts and context structure")


if __name__ == "__main__":
    # Check if vLLM is accessible
    try:
        response = requests.get(f"{VLLM_ENDPOINT}/health", timeout=5)
        if response.status_code != 200:
            print("❌ vLLM is not healthy. Please ensure it's running.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to vLLM at {VLLM_ENDPOINT}")
        print(f"   Error: {e}")
        print("\n💡 Make sure to run: docker-compose up -d")
        sys.exit(1)
    
    run_prefix_caching_test()