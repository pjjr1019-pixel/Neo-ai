import unittest
from unittest.mock import MagicMock
import numpy as np
from monitoring.drift_detector import detect_and_alert, DriftDetector

class MockChannel:
    def __init__(self):
        self.sent_alerts = []
    def send(self, alert):
        self.sent_alerts.append(alert)
        return True

class MockDispatcher:
    def __init__(self):
        self.alerts = []
    def dispatch(self, alert):
        self.alerts.append(alert)
        return {"sent": 1, "failed": 0, "results": {"MockChannel": True}}

class TestDriftAlertIntegration(unittest.TestCase):
    def test_alert_triggered_on_drift(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 1000)
        new_data = rng.normal(2, 1, 1000)
        dispatcher = MockDispatcher()
        result = detect_and_alert(baseline, new_data, dispatcher=dispatcher)
        self.assertTrue(result["drift"])
        self.assertEqual(len(dispatcher.alerts), 1)
        self.assertIn("Drift Detected", dispatcher.alerts[0].title)

    def test_no_alert_when_no_drift(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 1000)
        new_data = rng.normal(0, 1, 1000)
        dispatcher = MockDispatcher()
        result = detect_and_alert(baseline, new_data, dispatcher=dispatcher)
        self.assertFalse(result["drift"])
        self.assertEqual(len(dispatcher.alerts), 0)

if __name__ == "__main__":
    unittest.main()
