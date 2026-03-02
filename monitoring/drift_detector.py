"""
Drift Detector for NEO Hybrid AI Trading System
Detects model/data drift and triggers alerts.
"""

import numpy as np
from scipy.stats import ks_2samp

class DriftDetector:
    def __init__(self, baseline):
        self.baseline = np.array(baseline)
    def detect(self, new_data, alpha=0.05):
        stat, p = ks_2samp(self.baseline, new_data)
        drift = bool(p < alpha)
        return {"drift": drift, "p_value": p, "statistic": stat}

if __name__ == "__main__":
    baseline = np.random.normal(0, 1, 1000)
    new_data = np.random.normal(0.5, 1, 1000)
    detector = DriftDetector(baseline)
    print(detector.detect(new_data))
