"""
Statistical Arbitrage / Pairs Trading for NEO Hybrid AI.

Implements cointegration testing (Engle-Granger style)
and spread-based signal generation for pairs of assets.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CointegrationResult:
    """Result of a cointegration test.

    Attributes:
        statistic: Test statistic (ADF on residuals).
        critical_values: Dict mapping significance levels
            to critical values.
        cointegrated: ``True`` if the pair appears
            cointegrated at the 5 % level.
        hedge_ratio: OLS hedge ratio (beta).
        mean_spread: Mean of the spread series.
        std_spread: Standard deviation of the spread.
    """

    statistic: float
    critical_values: Dict[str, float]
    cointegrated: bool
    hedge_ratio: float
    mean_spread: float
    std_spread: float


@dataclass
class PairsSignal:
    """A trading signal from the pairs strategy.

    Attributes:
        z_score: Current spread z-score.
        signal: ``"long_a_short_b"``,
            ``"short_a_long_b"``, or ``"neutral"``.
        spread: Current raw spread value.
    """

    z_score: float
    signal: str
    spread: float


def _ols_hedge_ratio(
    a: np.ndarray,  # type: ignore[type-arg]
    b: np.ndarray,  # type: ignore[type-arg]
) -> float:
    """Ordinary least-squares hedge ratio (b regressed on a).

    Args:
        a: Dependent series.
        b: Independent series.

    Returns:
        Slope coefficient.
    """
    b_mean = np.mean(b)
    a_mean = np.mean(a)
    cov = np.sum((b - b_mean) * (a - a_mean))
    var = np.sum((b - b_mean) ** 2)
    if var == 0:
        return 0.0
    return float(cov / var)


def _adf_statistic(
    series: np.ndarray,  # type: ignore[type-arg]
) -> float:
    """Augmented Dickey-Fuller test statistic (no lags).

    Fits delta_y = alpha * y_{t-1} + epsilon and
    returns t-statistic on alpha.

    Args:
        series: 1-D time series.

    Returns:
        t-statistic for the unit-root coefficient.
    """
    y = series[:-1]
    dy = np.diff(series)
    n = len(y)
    if n < 3:
        return 0.0
    y_mean = np.mean(y)
    alpha = float(
        np.sum(dy * (y - y_mean)) / (np.sum((y - y_mean) ** 2) + 1e-12)
    )
    residuals = dy - alpha * (y - y_mean)
    se = float(
        np.sqrt(
            np.sum(residuals**2)
            / (n - 1)
            / (np.sum((y - y_mean) ** 2) + 1e-12)
        )
    )
    if se == 0:
        return 0.0
    return alpha / se


# Approximate ADF critical values for sample size >= 100.
_ADF_CRITICAL: Dict[str, float] = {
    "1%": -3.43,
    "5%": -2.86,
    "10%": -2.57,
}


class PairsTrader:
    """Pairs trading strategy using cointegration.

    Workflow:
        1. ``test_cointegration(a, b)`` — test whether
           two price series are cointegrated.
        2. ``generate_signals(a, b)`` — produce z-score
           based entry/exit signals.

    Args:
        entry_z: Z-score threshold to enter a trade.
        exit_z: Z-score threshold to close a trade.
        lookback: Rolling window for spread statistics.
            When ``None``, use the full history.
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lookback: Optional[int] = None,
    ) -> None:
        """Initialise the trader."""
        self._entry_z = entry_z
        self._exit_z = exit_z
        self._lookback = lookback
        self._last_result: Optional[CointegrationResult] = None

    # ------------------------------------------------------------------
    # Cointegration testing
    # ------------------------------------------------------------------

    def test_cointegration(
        self,
        prices_a: Sequence[float],
        prices_b: Sequence[float],
    ) -> CointegrationResult:
        """Test for cointegration between two series.

        Uses an Engle-Granger style approach:
        1. Estimate OLS hedge ratio.
        2. Compute spread residuals.
        3. Run ADF test on residuals.

        Args:
            prices_a: First price series.
            prices_b: Second price series.

        Returns:
            :class:`CointegrationResult`.

        Raises:
            ValueError: If series lengths do not match
                or are too short.
        """
        a = np.asarray(prices_a, dtype=float)
        b = np.asarray(prices_b, dtype=float)
        if len(a) != len(b):
            raise ValueError("Series must have equal length")
        if len(a) < 20:
            raise ValueError("Need >= 20 observations")

        beta = _ols_hedge_ratio(a, b)
        spread = a - beta * b
        stat = _adf_statistic(spread)
        cointegrated = stat < _ADF_CRITICAL["5%"]

        self._last_result = CointegrationResult(
            statistic=round(stat, 4),
            critical_values=dict(_ADF_CRITICAL),
            cointegrated=cointegrated,
            hedge_ratio=round(beta, 6),
            mean_spread=round(float(np.mean(spread)), 6),
            std_spread=round(float(np.std(spread)) + 1e-12, 6),
        )
        logger.info(
            "Cointegration test: stat=%.4f, "
            "cointegrated=%s, hedge_ratio=%.4f",
            stat,
            cointegrated,
            beta,
        )
        return self._last_result

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        prices_a: Sequence[float],
        prices_b: Sequence[float],
    ) -> List[PairsSignal]:
        """Generate trading signals from the spread.

        If ``test_cointegration`` has not been called,
        it is called automatically.

        Args:
            prices_a: First series.
            prices_b: Second series.

        Returns:
            List of :class:`PairsSignal`, one per
            observation.
        """
        a = np.asarray(prices_a, dtype=float)
        b = np.asarray(prices_b, dtype=float)

        if self._last_result is None:
            self.test_cointegration(prices_a, prices_b)

        result = self._last_result
        assert result is not None
        spread = a - result.hedge_ratio * b

        signals: List[PairsSignal] = []
        for i in range(len(spread)):
            if self._lookback and i >= self._lookback:
                window = spread[i - self._lookback + 1 : i + 1]
            else:
                window = spread[: i + 1]

            mu = float(np.mean(window))
            sigma = float(np.std(window)) + 1e-12
            z = (spread[i] - mu) / sigma

            if z > self._entry_z:
                sig = "short_a_long_b"
            elif z < -self._entry_z:
                sig = "long_a_short_b"
            elif abs(z) < self._exit_z:
                sig = "neutral"
            else:
                sig = "neutral"

            signals.append(
                PairsSignal(
                    z_score=round(z, 4),
                    signal=sig,
                    spread=round(float(spread[i]), 6),
                )
            )
        return signals

    def current_signal(
        self,
        prices_a: Sequence[float],
        prices_b: Sequence[float],
    ) -> PairsSignal:
        """Return the latest signal only.

        Convenience method that returns the last element
        of :meth:`generate_signals`.

        Args:
            prices_a: First series.
            prices_b: Second series.

        Returns:
            Most-recent :class:`PairsSignal`.
        """
        sigs = self.generate_signals(prices_a, prices_b)
        return sigs[-1]

    def summary(self) -> Dict[str, Any]:
        """Return strategy summary.

        Returns:
            Dict with parameters and last test result.
        """
        return {
            "entry_z": self._entry_z,
            "exit_z": self._exit_z,
            "lookback": self._lookback,
            "last_cointegration": (
                {
                    "statistic": self._last_result.statistic,
                    "cointegrated": (self._last_result.cointegrated),
                    "hedge_ratio": (self._last_result.hedge_ratio),
                }
                if self._last_result
                else None
            ),
        }
