"""
Resource Monitoring Script for NEO FastAPI Service

This script logs CPU and memory usage of the FastAPI process at regular
intervals. Run this script alongside your FastAPI server for live resource
monitoring.
"""

import os
import time
from typing import NoReturn

import psutil

LOG_FILE = "resource_usage.log"
INTERVAL = 2  # seconds


def log_resource_usage() -> NoReturn:
    """
    Log CPU and memory usage of the current process at regular intervals.
    Appends results to LOG_FILE every INTERVAL seconds.
    """
    process = psutil.Process(os.getpid())
    with open(LOG_FILE, "a") as f:
        while True:
            mem = process.memory_info().rss / 1024**2  # Memory in MB
            cpu = process.cpu_percent(interval=0.1)  # CPU percent
            # Write timestamped resource usage to log
            f.write(
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Memory: {mem:.2f} MB | CPU: {cpu:.2f}%\n"
            )
            f.flush()
            time.sleep(INTERVAL)


if __name__ == "__main__":
    print(f"Logging resource usage to {LOG_FILE} every {INTERVAL} seconds...")
    log_resource_usage()
