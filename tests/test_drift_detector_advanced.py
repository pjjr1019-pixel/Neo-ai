import unittest

import numpy as np

from monitoring.drift_detector import DriftDetector


class TestDriftDetectorAdvanced(unittest.TestCase):
    def test_no_drift(self):
        rng = np.random.default_rng(0)
        baseline = rng.normal(0, 1, 1000)
        new_data = rng.normal(0, 1, 1000)
        detector = DriftDetector(baseline)
        result = detector.detect(new_data)
        self.assertFalse(result["drift"])  # Should not detect drift
        self.assertGreater(result["p_value"], 0.01)

    def test_drift_detected(self):
        rng = np.random.default_rng(0)
        baseline = rng.normal(0, 1, 1000)
        new_data = rng.normal(2, 1, 1000)
        detector = DriftDetector(baseline)
        result = detector.detect(new_data)
        self.assertTrue(result["drift"])  # Should detect drift
        self.assertLess(result["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
