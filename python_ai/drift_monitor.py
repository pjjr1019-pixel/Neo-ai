"""
Model Drift Detection & Auto-Retrain for NEO Hybrid AI.

Monitors prediction distribution shifts and accuracy degradation.
When drift exceeds a threshold the module signals the need for
model retraining.
"""

import logging
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect model drift using prediction-distribution statistics.

    Keeps a sliding window of recent predictions and compares
    their distribution to a reference (baseline) window recorded
    at the last training event.

    Attributes:
        window_size: Number of predictions in the sliding window.
        drift_threshold: Z-score change that triggers a drift alert.
    """

    def __init__(
        self,
        window_size: int = 200,
        drift_threshold: float = 2.0,
    ) -> None:
        """Initialise the drift detector.

        Args:
            window_size: Size of the sliding window.
            drift_threshold: Threshold (in std-devs) for flagging drift.
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold

        # Reference statistics (set after training)
        self._ref_mean: Optional[float] = None
        self._ref_std: Optional[float] = None

        # Sliding window of recent predictions
        self._recent: Deque[float] = deque(maxlen=window_size)

        # Accuracy tracking
        self._predictions: Deque[float] = deque(maxlen=window_size)
        self._actuals: Deque[float] = deque(maxlen=window_size)
        self._drift_events: List[Dict[str, Any]] = []

    # ── Baseline management ───────────────────────────────────

    def set_baseline(
        self,
        predictions: List[float],
    ) -> None:
        """Record the reference distribution from training output.

        Args:
            predictions: Predictions on the training/validation set.
        """
        arr = np.array(predictions, dtype=float)
        self._ref_mean = float(np.mean(arr))
        self._ref_std = float(np.std(arr)) if len(arr) > 1 else 1.0
        logger.info(
            "Drift baseline set: mean=%.4f  std=%.4f",
            self._ref_mean,
            self._ref_std,
        )

    # ── Online updates ────────────────────────────────────────

    def record_prediction(
        self,
        prediction: float,
        actual: Optional[float] = None,
    ) -> None:
        """Feed a new prediction (and optional ground-truth).

        Args:
            prediction: Model output.
            actual: True value if available (for accuracy tracking).
        """
        self._recent.append(prediction)
        self._predictions.append(prediction)
        if actual is not None:
            self._actuals.append(actual)

    # ── Drift checks ──────────────────────────────────────────

    def is_drifted(self) -> bool:
        """Return True if distribution drift exceeds threshold."""
        if self._ref_mean is None or len(self._recent) < 10:
            return False

        current_mean = float(np.mean(self._recent))
        ref_std = self._ref_std if self._ref_std and self._ref_std > 0 else 1.0
        z_score = abs(current_mean - self._ref_mean) / ref_std

        return z_score >= self.drift_threshold

    def drift_score(self) -> float:
        """Quantitative drift score (z-score of mean shift).

        Returns:
            Non-negative float. Values ≥ drift_threshold indicate drift.
        """
        if self._ref_mean is None or len(self._recent) < 10:
            return 0.0

        current_mean = float(np.mean(self._recent))
        ref_std = self._ref_std if self._ref_std and self._ref_std > 0 else 1.0
        return abs(current_mean - self._ref_mean) / ref_std

    def accuracy_score(self) -> Optional[float]:
        """Rolling R² (or similar) on ``(prediction, actual)`` pairs.

        Returns:
            R² value, or None if insufficient data.
        """
        if len(self._actuals) < 10:
            return None

        preds = np.array(self._predictions, dtype=float)[-len(self._actuals) :]
        acts = np.array(self._actuals, dtype=float)

        ss_res = float(np.sum((acts - preds) ** 2))
        ss_tot = float(np.sum((acts - np.mean(acts)) ** 2))
        if ss_tot == 0:
            return 1.0
        return 1.0 - ss_res / ss_tot

    def to_dict(self) -> Dict[str, Any]:
        """Serialise detector state for reporting."""
        return {
            "window_size": self.window_size,
            "samples_collected": len(self._recent),
            "drift_score": round(self.drift_score(), 4),
            "drift_threshold": self.drift_threshold,
            "is_drifted": self.is_drifted(),
            "accuracy_r2": self.accuracy_score(),
            "ref_mean": self._ref_mean,
            "ref_std": self._ref_std,
        }


class AutoRetrainTrigger:
    """Watches a DriftDetector and fires a retrain callback.

    Attributes:
        detector: The DriftDetector to monitor.
        cooldown_seconds: Minimum seconds between retrain events.
    """

    def __init__(
        self,
        detector: DriftDetector,
        retrain_callback: Optional[Callable[[], Dict[str, Any]]] = None,
        accuracy_floor: float = 0.3,
        cooldown_seconds: float = 3600.0,
    ) -> None:
        """Initialise auto-retrain trigger.

        Args:
            detector: DriftDetector to evaluate.
            retrain_callback: Called when retrain is needed.
                Must return a dict with training metrics.
            accuracy_floor: Retrain if R² drops below this.
            cooldown_seconds: Min interval between retrains.
        """
        self.detector = detector
        self.retrain_callback = retrain_callback
        self.accuracy_floor = accuracy_floor
        self.cooldown_seconds = cooldown_seconds
        self._last_retrain_time = 0.0
        self._retrain_count = 0

    def check(self) -> Optional[Dict[str, Any]]:
        """Evaluate drift and accuracy, retrain if needed.

        Returns:
            Retrain result dict, or None if no retrain fired.
        """
        now = time.time()
        if (now - self._last_retrain_time) < self.cooldown_seconds:
            return None

        should_retrain = False
        reason = ""

        if self.detector.is_drifted():
            should_retrain = True
            reason = "distribution_drift"

        r2 = self.detector.accuracy_score()
        if r2 is not None and r2 < self.accuracy_floor:
            should_retrain = True
            reason = reason or "accuracy_degradation"

        if not should_retrain or self.retrain_callback is None:
            return None

        logger.info(
            "Auto-retrain triggered (reason=%s, drift=%.3f, r2=%s)",
            reason,
            self.detector.drift_score(),
            r2,
        )

        self._last_retrain_time = now
        self._retrain_count += 1

        try:
            result = self.retrain_callback()
            result["trigger_reason"] = reason
            result["retrain_number"] = self._retrain_count
            return result
        except Exception as exc:
            logger.error("Auto-retrain failed: %s", exc)
            return {"error": str(exc), "trigger_reason": reason}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise trigger state."""
        return {
            "retrain_count": self._retrain_count,
            "last_retrain_time": self._last_retrain_time,
            "cooldown_seconds": self.cooldown_seconds,
            "accuracy_floor": self.accuracy_floor,
            "detector": self.detector.to_dict(),
        }
