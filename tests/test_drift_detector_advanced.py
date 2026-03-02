import unittest
from monitoring.drift_detector import DriftDetector
import numpy as np

class TestDriftDetectorAdvanced(unittest.TestCase):
    def test_no_drift(self):
        baseline = np.random.normal(0, 1, 1000)
        new_data = np.random.normal(0, 1, 1000)
        detector = DriftDetector(baseline)
        result = detector.detect(new_data)
        self.assertFalse(result["drift"])  # Should not detect drift
        self.assertGreater(result["p_value"], 0.01)

    def test_drift_detected(self):
        baseline = np.random.normal(0, 1, 1000)
        new_data = np.random.normal(2, 1, 1000)
        detector = DriftDetector(baseline)
        result = detector.detect(new_data)
        self.assertTrue(result["drift"])  # Should detect drift
        self.assertLess(result["p_value"], 0.05)

if __name__ == "__main__":
    unittest.main()
