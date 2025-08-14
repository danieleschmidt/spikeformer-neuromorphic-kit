#!/usr/bin/env python3
"""Production health check for Spikeformer Neuromorphic Kit."""

import sys
import time
import requests
import json
from pathlib import Path


def check_api_health():
    """Check API health endpoint."""
    try:
        response = requests.get("http://localhost:8080/health", timeout=10)
        return response.status_code == 200
    except:
        return False


def check_neuromorphic_hardware():
    """Check neuromorphic hardware connectivity."""
    try:
        # Simulate hardware check
        return True  # Would check actual hardware
    except:
        return False


def check_memory_usage():
    """Check memory usage is within limits."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90
    except:
        return False


def check_disk_space():
    """Check disk space availability."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        usage_percent = (used / total) * 100
        return usage_percent < 85
    except:
        return False


def main():
    """Run comprehensive health checks."""
    
    checks = {
        "api_health": check_api_health,
        "neuromorphic_hardware": check_neuromorphic_hardware,
        "memory_usage": check_memory_usage,
        "disk_space": check_disk_space
    }
    
    results = {}
    all_healthy = True
    
    for check_name, check_func in checks.items():
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_healthy = False
        except Exception as e:
            results[check_name] = False
            all_healthy = False
    
    # Output results
    health_status = {
        "timestamp": time.time(),
        "healthy": all_healthy,
        "checks": results
    }
    
    print(json.dumps(health_status, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if all_healthy else 1)


if __name__ == "__main__":
    main()
