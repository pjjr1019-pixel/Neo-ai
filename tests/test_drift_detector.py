import unittest
import numpy as np
from monitoring.drift_detector import DriftDetector

class TestDriftDetector(unittest.TestCase):
    def test_drift_detection(self):
        baseline = np.random.normal(0, 1, 1000)
        new_data = np.random.normal(0.5, 1, 1000)
        detector = DriftDetector(baseline)
        result = detector.detect(new_data)
        self.assertIn("drift", result)
        self.assertIn("p_value", result)
        self.assertIn("statistic", result)
        self.assertIsInstance(result["drift"], bool)

if __name__ == "__main__":
    unittest.main()
