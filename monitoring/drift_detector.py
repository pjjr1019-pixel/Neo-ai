"""
Drift Detector for NEO Hybrid AI Trading System
Detects model/data drift and triggers alerts.
"""

import numpy as np
from scipy.stats import ks_2samp

# Alert integration
import sys
sys.path.append("..")  # Ensure parent dir is in path for import
try:
    from python_ai.alert_notifier import Alert, AlertDispatcher, WebhookChannel
except ImportError:
    Alert = AlertDispatcher = WebhookChannel = None


class DriftDetector:
    def __init__(self, baseline):
        self.baseline = np.array(baseline)
    def detect(self, new_data, alpha=0.05):
        stat, p = ks_2samp(self.baseline, new_data)
        drift = bool(p < alpha)
        return {"drift": drift, "p_value": p, "statistic": stat}


# Utility: detect drift and send alert if needed
def detect_and_alert(baseline, new_data, dispatcher=None, alert_channel_url=None, alpha=0.05):
    """
    Detect drift and send alert if drift is detected.
    dispatcher: AlertDispatcher instance (optional)
    alert_channel_url: Webhook URL (optional, used if dispatcher is None)
    Returns drift result dict.
    """
    detector = DriftDetector(baseline)
    result = detector.detect(new_data, alpha=alpha)
    if result["drift"] and Alert is not None:
        alert = Alert(
            title="Model/Data Drift Detected",
            message=f"Drift detected (p={result['p_value']:.4g}, stat={result['statistic']:.3f})",
            severity="warning",
            source="drift_detector",
            metadata=result,
        )
        if dispatcher is not None:
            dispatcher.dispatch(alert)
        elif alert_channel_url is not None:
            disp = AlertDispatcher([WebhookChannel(alert_channel_url)])
            disp.dispatch(alert)
    return result

if __name__ == "__main__":
    baseline = np.random.normal(0, 1, 1000)
    new_data = np.random.normal(0.5, 1, 1000)
    detector = DriftDetector(baseline)
    print(detector.detect(new_data))
