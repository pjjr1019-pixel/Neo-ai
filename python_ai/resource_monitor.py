"""
Resource Monitoring Script for NEO FastAPI Service

This script logs CPU and memory usage of the FastAPI process at regular intervals.
Run this script alongside your FastAPI server for live resource monitoring.
"""
import time
import psutil
import os

def log_resource_usage():
LOG_FILE = "resource_usage.log"
INTERVAL = 2  # seconds

process = psutil.Process(os.getpid())


def log_resource_usage():
    with open(LOG_FILE, "a") as f:
        while True:
            mem = process.memory_info().rss / 1024 ** 2
            cpu = process.cpu_percent(interval=0.1)
            f.write(
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Memory: {mem:.2f} MB | CPU: {cpu:.2f}%\n"
            )
            f.flush()
            time.sleep(INTERVAL)


if __name__ == "__main__":
    print(
        f"Logging resource usage to {LOG_FILE} "
        f"every {INTERVAL} seconds..."
    )
    log_resource_usage()
