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
