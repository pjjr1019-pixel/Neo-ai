import os
import time
import tempfile
import psutil
import threading
import python_ai.resource_monitor as rm


class FakeProcess:
    def __init__(self, pid, memory_info_rss, cpu_percent):
        self.pid = pid
        self._memory_info_rss = memory_info_rss
        self._cpu_percent = cpu_percent

    def memory_info(self):
        class Mem:
            def __init__(self, rss):
                self.rss = rss

        return Mem(self._memory_info_rss)

    def cpu_percent(self, interval=None):
        return self._cpu_percent


def test_log_resource_usage_writes_log(monkeypatch):
    """
    Test that log_resource_usage writes a log entry with memory and CPU info.
    """
    # Patch time.sleep to break after one loop
    calls = {"count": 0}

    def fake_sleep(interval):
        """Fake sleep function to break after one call for testing."""

        calls["count"] += 1
        if calls["count"] > 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(time, "sleep", fake_sleep)
    # Patch open to use a temp file
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        log_path = tf.name
    monkeypatch.setattr(rm, "LOG_FILE", log_path)
    monkeypatch.setattr(psutil, "Process", lambda pid: FakeProcess())
    monkeypatch.setattr(os, "getpid", lambda: 1234)

    monkeypatch.setattr(psutil, "Process", lambda pid: FakeProcess())
    monkeypatch.setattr(os, "getpid", lambda: 1234)
    # Run in a thread to allow KeyboardInterrupt to break loop
    thread = threading.Thread(target=rm.log_resource_usage)
    thread.daemon = True
    try:
        thread.start()
        thread.join(timeout=2)
    except Exception:
        pass
    with open(log_path) as f:
        lines = f.readlines()
    assert any(("Memory:" in line and "CPU:" in line) for line in lines)
    os.remove(log_path)
    import importlib
    import sys

    name = "python_ai.resource_monitor"
    if name in sys.modules:
        del sys.modules[name]
    importlib.import_module(name)
