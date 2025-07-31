#!/usr/bin/env python3
"""
Health check script for Multi-GPU LLM Inference Stack.
Checks all services and reports their status.
"""

import sys
import time
import requests
from typing import Dict, Tuple

# Service endpoints
SERVICES = {
    "vLLM": "http://localhost:8000/health"
}

def check_service(name: str, url: str, timeout: int = 5) -> Tuple[bool, str]:
    """Check if a service is healthy."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, "OK"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def main():
    """Main health check routine."""
    print("=== LLM Inference Stack Health Check ===")
    print(f"Checking {len(SERVICES)} services...\n")
    
    all_healthy = True
    results: Dict[str, Tuple[bool, str]] = {}
    
    # Check each service
    for name, url in SERVICES.items():
        healthy, message = check_service(name, url)
        results[name] = (healthy, message)
        all_healthy = all_healthy and healthy
    
    # Display results
    for name, (healthy, message) in results.items():
        status = "✓" if healthy else "✗"
        print(f"{status} {name:<10} {message}")
    
    print()
    
    # Overall status
    if all_healthy:
        print("✓ All services are healthy")
        return 0
    else:
        print("✗ Some services are not responding")
        print("\nTroubleshooting:")
        print("1. Check if services are running: docker-compose ps")
        print("2. View logs: docker-compose logs -f [service-name]")
        print("3. Restart services: docker-compose restart")
        return 1

if __name__ == "__main__":
    sys.exit(main())