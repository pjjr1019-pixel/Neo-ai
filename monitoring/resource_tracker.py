"""
Resource Tracker for NEO Hybrid AI Trading System
Tracks and reports resource usage (CPU, memory, disk, network) for cost-aware scheduling and optimization.
"""

import psutil
import json
from datetime import datetime, timezone


def get_resource_usage():
    usage = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory()._asdict(),
        "disk": psutil.disk_usage("/")._asdict(),
        "net_io": psutil.net_io_counters()._asdict(),
    }
    return usage


def report_resource_usage(filepath="resource_usage_report.json"):
    usage = get_resource_usage()
    with open(filepath, "a") as f:
        f.write(json.dumps(usage) + "\n")
    return usage


if __name__ == "__main__":
    print(json.dumps(get_resource_usage(), indent=2))
