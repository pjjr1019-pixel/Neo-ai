"""
Resource Monitoring Script for NEO FastAPI Service

This script logs CPU and memory usage of the FastAPI process at regular
intervals. Run this script alongside your FastAPI server for live resource
monitoring.
"""

import logging
import os
import time
from typing import Optional, Protocol

import psutil

logger = logging.getLogger(__name__)

LOG_FILE = "resource_usage.log"
INTERVAL = 2  # seconds


class _StopEventLike(Protocol):
    """Minimal protocol for a stop event used by the monitor loop."""

    def is_set(self) -> bool:
        """Return True when monitoring should stop."""


def log_resource_usage(
    stop_event: Optional[_StopEventLike] = None,
    *,
    max_iterations: Optional[int] = None,
) -> None:
    """
    Log CPU and memory usage of the current process at regular intervals.
    Appends results to LOG_FILE every INTERVAL seconds.
    """
    process = psutil.Process(os.getpid())
    iterations = 0
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                mem = process.memory_info().rss / 1024**2  # Memory in MB
                cpu = process.cpu_percent(interval=0.1)  # CPU percent
                # Write timestamped resource usage to log
                f.write(
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Memory: {mem:.2f} MB | CPU: {cpu:.2f}%\n"
                )
                f.flush()
            except Exception:
                logger.exception("Resource sampling failed")

            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            try:
                time.sleep(INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception:
                logger.exception("Resource monitor sleep failed")
                break


if __name__ == "__main__":
    logger.info(
        "Logging resource usage to %s every %d seconds...",
        LOG_FILE,
        INTERVAL,
    )
    log_resource_usage()
