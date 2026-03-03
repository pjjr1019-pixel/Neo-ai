import types
import pytest

def test_collect_metrics_sets_gauges(monkeypatch):
    """Test that collect_metrics sets CPU and memory usage gauges."""
    cpu_set = {}
    mem_set = {}

    class FakeGauge:
        def __init__(self, name):
            self.name = name
            self.value = None
        def set(self, v):
            self.value = v
            if self.name == 'cpu':
                cpu_set['v'] = v
            else:
                mem_set['v'] = v

    import monitoring.prometheus_metrics as pm
    monkeypatch.setattr(pm, 'CPU_USAGE', FakeGauge('cpu'))
    monkeypatch.setattr(pm, 'MEMORY_USAGE', FakeGauge('mem'))
    monkeypatch.setattr(pm.psutil, 'cpu_percent', lambda: 42.0)
    monkeypatch.setattr(pm.psutil, 'virtual_memory', lambda: types.SimpleNamespace(percent=24.0))

    pm.collect_metrics()
    assert cpu_set['v'] == 42.0
    assert mem_set['v'] == 24.0
import unittest
import requests
import threading
import time
from monitoring import prometheus_metrics

class TestPrometheusMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_thread = threading.Thread(target=prometheus_metrics.start_http_server, args=(8000,), daemon=True)
        cls.server_thread.start()
        time.sleep(1)

    def test_metrics_endpoint(self):
        prometheus_metrics.collect_metrics()
        response = requests.get("http://localhost:8000")
        self.assertEqual(response.status_code, 200)
        self.assertIn("cpu_usage_percent", response.text)
        self.assertIn("memory_usage_percent", response.text)

if __name__ == "__main__":
    unittest.main()
