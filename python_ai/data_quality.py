"""
Data Quality Monitor for NEO Hybrid AI.

Validates OHLCV data for completeness, consistency, and
anomalies before it reaches the model or backtesting engine.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Container for a data quality check result.

    Attributes:
        is_valid: True if all checks pass.
        checks: Individual check results.
        warnings: Non-fatal issues.
        errors: Fatal issues.
    """

    is_valid: bool = True
    checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report."""
        return {
            "is_valid": self.is_valid,
            "checks": self.checks,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class DataQualityMonitor:
    """Validate OHLCV data for trading consumption.

    Checks performed:
    - Required columns present
    - No NaN / Inf values
    - OHLCV ordering (low <= open/close <= high)
    - Volume non-negative
    - Timestamp gaps
    - Price anomaly detection (z-score)
    """

    def __init__(
        self,
        max_gap_bars: int = 5,
        anomaly_z_threshold: float = 5.0,
    ) -> None:
        """Initialise the monitor.

        Args:
            max_gap_bars: Max consecutive missing bars before error.
            anomaly_z_threshold: Z-score above which a price is
                flagged as anomalous.
        """
        self.max_gap_bars = max_gap_bars
        self.anomaly_z_threshold = anomaly_z_threshold

    def validate(
        self,
        ohlcv: Dict[str, List[float]],
        timestamps: Optional[List[float]] = None,
    ) -> QualityReport:
        """Run all quality checks on OHLCV data.

        Args:
            ohlcv: Dict with keys ``open``, ``high``, ``low``,
                   ``close``, ``volume``.
            timestamps: Optional list of Unix timestamps for gap
                       detection.

        Returns:
            QualityReport with check results.
        """
        report = QualityReport()

        self._check_required_columns(ohlcv, report)
        if report.errors:
            report.is_valid = False
            return report

        self._check_lengths(ohlcv, report)
        self._check_nan_inf(ohlcv, report)
        self._check_ohlc_ordering(ohlcv, report)
        self._check_volume(ohlcv, report)
        self._check_anomalies(ohlcv, report)

        if timestamps:
            self._check_gaps(timestamps, report)

        report.is_valid = len(report.errors) == 0
        return report

    # ── Individual checks ─────────────────────────────────────

    def _check_required_columns(
        self,
        ohlcv: Dict[str, List[float]],
        report: QualityReport,
    ) -> None:
        """Verify that all required OHLCV columns are present."""
        required = {"open", "high", "low", "close", "volume"}
        present = set(ohlcv.keys())
        missing = required - present
        if missing:
            report.errors.append(f"Missing columns: {sorted(missing)}")
            report.checks["required_columns"] = False
        else:
            report.checks["required_columns"] = True

    def _check_lengths(
        self,
        ohlcv: Dict[str, List[float]],
        report: QualityReport,
    ) -> None:
        """Check that all columns have equal length."""
        lengths = {k: len(v) for k, v in ohlcv.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            report.errors.append(f"Column lengths mismatch: {lengths}")
            report.checks["equal_lengths"] = False
        else:
            report.checks["equal_lengths"] = True

        total = lengths.get("close", 0)
        if total < 2:
            report.errors.append(
                f"Insufficient data: {total} bars (need >= 2)"
            )
            report.checks["min_bars"] = False
        else:
            report.checks["min_bars"] = True

    def _check_nan_inf(
        self,
        ohlcv: Dict[str, List[float]],
        report: QualityReport,
    ) -> None:
        """Detect NaN or Inf values in the data."""
        ok = True
        for col in ("open", "high", "low", "close", "volume"):
            arr = np.array(ohlcv.get(col, []), dtype=float)
            if np.any(np.isnan(arr)):
                report.errors.append(f"NaN values in '{col}'")
                ok = False
            if np.any(np.isinf(arr)):
                report.errors.append(f"Inf values in '{col}'")
                ok = False
        report.checks["no_nan_inf"] = ok

    def _check_ohlc_ordering(
        self,
        ohlcv: Dict[str, List[float]],
        report: QualityReport,
    ) -> None:
        """Validate OHLC ordering (low <= open/close <= high)."""
        o = np.array(ohlcv.get("open", []), dtype=float)
        h = np.array(ohlcv.get("high", []), dtype=float)
        lo = np.array(ohlcv.get("low", []), dtype=float)
        c = np.array(ohlcv.get("close", []), dtype=float)

        violations = int(
            np.sum((lo > np.minimum(o, c)) | (h < np.maximum(o, c)))
        )
        if violations > 0:
            report.warnings.append(
                f"OHLC ordering violations: {violations} bars"
            )
            report.checks["ohlc_ordering"] = False
        else:
            report.checks["ohlc_ordering"] = True

    def _check_volume(
        self,
        ohlcv: Dict[str, List[float]],
        report: QualityReport,
    ) -> None:
        """Ensure volume values are non-negative."""
        vol = np.array(ohlcv.get("volume", []), dtype=float)
        neg = int(np.sum(vol < 0))
        if neg > 0:
            report.errors.append(f"Negative volume in {neg} bars")
            report.checks["volume_positive"] = False
        else:
            report.checks["volume_positive"] = True

        zero_pct = float(np.mean(vol == 0)) * 100 if len(vol) > 0 else 0
        if zero_pct > 50:
            report.warnings.append(f"{zero_pct:.0f}% of bars have zero volume")

    def _check_anomalies(
        self,
        ohlcv: Dict[str, List[float]],
        report: QualityReport,
    ) -> None:
        """Flag close-price anomalies using z-score analysis."""
        close = np.array(ohlcv.get("close", []), dtype=float)
        if len(close) < 10:
            report.checks["no_anomalies"] = True
            return

        returns = np.diff(close) / (close[:-1] + 1e-10)
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns))
        if std_r == 0:
            report.checks["no_anomalies"] = True
            return

        z_scores = np.abs((returns - mean_r) / std_r)
        anomalies = int(np.sum(z_scores > self.anomaly_z_threshold))
        if anomalies > 0:
            report.warnings.append(
                f"{anomalies} price anomalies "
                f"(z > {self.anomaly_z_threshold})"
            )
            report.checks["no_anomalies"] = False
        else:
            report.checks["no_anomalies"] = True

    def _check_gaps(
        self,
        timestamps: List[float],
        report: QualityReport,
    ) -> None:
        """Detect timestamp gaps larger than 1.5x the median interval."""
        if len(timestamps) < 2:
            report.checks["no_large_gaps"] = True
            return

        diffs = np.diff(timestamps)
        median_diff = float(np.median(diffs))
        if median_diff <= 0:
            report.checks["no_large_gaps"] = True
            return

        large_gaps = int(np.sum(diffs > median_diff * self.max_gap_bars))
        if large_gaps > 0:
            report.warnings.append(
                f"{large_gaps} gaps > {self.max_gap_bars}x "
                f"median bar interval"
            )
            report.checks["no_large_gaps"] = False
        else:
            report.checks["no_large_gaps"] = True
