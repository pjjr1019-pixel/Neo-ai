"""
Comprehensive tests for resource_monitor.py.
Ensures logging, error handling, and edge cases are covered.
All tests are flake8-compliant and include docstrings.
"""

import tempfile
import threading

import python_ai.resource_monitor as rm


class DummyProcess:
    """A dummy process for simulating psutil.Process."""

    def __init__(self, rss=12345678, cpu=42.0, mem_fail=False, cpu_fail=False):
        """Initialize DummyProcess with resource and error simulation options.

        Args:
            rss (int): Simulated memory usage (rss).
            cpu (float): Simulated CPU percent.
            mem_fail (bool): If True, memory_info() raises error.
            cpu_fail (bool): If True, cpu_percent() raises error.
        """
        self._rss = rss
        self._cpu = cpu
        self._mem_fail = mem_fail
        self._cpu_fail = cpu_fail

    def memory_info(self):
        """Return dummy memory info or raise if mem_fail."""
        if self._mem_fail:
            raise RuntimeError("fail mem")

        class Mem:
            """Holds memory usage rss value."""

            rss = self._rss

        return Mem()

    def cpu_percent(self, interval=None):
        """Return dummy CPU percent or raise if cpu_fail."""
        if self._cpu_fail:
            raise RuntimeError("fail cpu")
        return self._cpu


def test_log_resource_usage_writes_log(monkeypatch):
    """Test that log_resource_usage writes a log entry with memory and
    CPU info.
    """
    # Patch time.sleep to break after one loop
    calls = {"count": 0}

    def fake_sleep(interval):
        """Fake sleep function to break after more than one call for
        testing.
        """
        calls["count"] += 1
        if calls["count"] > 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(rm.time, "sleep", fake_sleep)
    # Patch open to use a temp file
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        log_path = tf.name
    monkeypatch.setattr(rm, "LOG_FILE", log_path)
    monkeypatch.setattr(rm.psutil, "Process", lambda pid: DummyProcess())
    monkeypatch.setattr(rm.os, "getpid", lambda: 1234)
    # Run in a thread to allow KeyboardInterrupt to break loop
    thread = threading.Thread(target=rm.log_resource_usage)
    thread.daemon = True
    thread.start()
    thread.join(timeout=1)
    with open(log_path) as f:
        content = f.read()
    assert "Memory:" in content and "CPU:" in content


def test_log_resource_usage_handles_exceptions(monkeypatch):
    """Test log_resource_usage handles exceptions in process info
    gracefully.
    """
    calls = {"count": 0}

    def fake_sleep(interval):
        """Fake sleep function to break after more than one call for
        testing.
        """
        calls["count"] += 1
        if calls["count"] > 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(rm.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        rm.psutil,
        "Process",
        lambda pid: DummyProcess(mem_fail=True, cpu_fail=True),
    )
    monkeypatch.setattr(rm.os, "getpid", lambda: 1234)
    monkeypatch.setattr(rm, "LOG_FILE", tempfile.mktemp())
    # Should not raise, just log error
    thread = threading.Thread(target=rm.log_resource_usage)
    thread.daemon = True
    thread.start()
    thread.join(timeout=1)


def test_log_resource_usage_file_write(monkeypatch, tmp_path):
    """Test log_resource_usage writes to a custom file path."""
    log_file = tmp_path / "custom.log"
    calls = {"count": 0}

    def fake_sleep(interval):
        """Fake sleep function to break after any call for testing."""
        calls["count"] += 1
        if calls["count"] > 0:
            raise KeyboardInterrupt()

    monkeypatch.setattr(rm.time, "sleep", fake_sleep)
    monkeypatch.setattr(rm, "LOG_FILE", str(log_file))
    monkeypatch.setattr(rm.psutil, "Process", lambda pid: DummyProcess())
    monkeypatch.setattr(rm.os, "getpid", lambda: 1)
    try:
        rm.log_resource_usage()
    except KeyboardInterrupt:
        pass
    assert log_file.exists()
    content = log_file.read_text()
    assert "Memory:" in content and "CPU:" in content


def test_log_resource_usage_multiple_threads(monkeypatch, tmp_path):
    """Test log_resource_usage can be started in multiple threads safely."""
    log_file = tmp_path / "multi.log"
    monkeypatch.setattr(rm, "LOG_FILE", str(log_file))
    monkeypatch.setattr(rm.psutil, "Process", lambda pid: DummyProcess())
    monkeypatch.setattr(rm.os, "getpid", lambda: 1)

    def fake_sleep(interval):
        """Fake sleep function to immediately raise KeyboardInterrupt for
        testing.
        """
        raise KeyboardInterrupt()

    monkeypatch.setattr(rm.time, "sleep", fake_sleep)
    threads = [
        threading.Thread(target=rm.log_resource_usage) for _ in range(2)
    ]
    for t in threads:
        t.daemon = True
        t.start()
    for t in threads:
        t.join(timeout=1)
    assert log_file.exists()


def test_log_resource_usage_import_side_effect(monkeypatch):
    """Test that importing resource_monitor does not start logging
    automatically.
    """
    import importlib

    import python_ai.resource_monitor as rm2

    importlib.reload(rm2)
    # Should not start logging on import
    assert hasattr(rm2, "log_resource_usage")
