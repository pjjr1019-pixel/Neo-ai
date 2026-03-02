"""
Monte Carlo Simulation for NEO Hybrid AI.

Generates synthetic price paths from fitted return
distributions and produces probabilistic metrics
(confidence intervals, drawdown distributions, etc.).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Generate and analyse synthetic price paths.

    Fits a normal distribution to historical returns
    then draws thousands of paths forward.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialise the simulator."""
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._paths: Optional[np.ndarray] = None

    # ── simulation ────────────────────────────────────

    def simulate(
        self,
        returns: List[float],
        n_paths: int = 10_000,
        horizon: int = 30,
        initial_value: float = 1.0,
    ) -> np.ndarray:
        """Run Monte Carlo simulation.

        Args:
            returns: Historical return series used to
                estimate μ and σ.
            n_paths: Number of simulated paths.
            horizon: Forward time-steps to simulate.
            initial_value: Starting portfolio / price level.

        Returns:
            Array of shape ``(n_paths, horizon + 1)``
            with cumulative price paths.
        """
        arr = np.array(returns, dtype=np.float64)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))

        sims = self._rng.normal(mu, sigma, (n_paths, horizon))
        paths = np.zeros((n_paths, horizon + 1))
        paths[:, 0] = initial_value
        for t in range(horizon):
            paths[:, t + 1] = paths[:, t] * (1.0 + sims[:, t])
        self._paths = paths
        logger.info(
            "Simulated %d paths over %d periods",
            n_paths,
            horizon,
        )
        return paths

    # ── analytics ─────────────────────────────────────

    def confidence_interval(
        self,
        confidence: float = 0.95,
    ) -> Dict[str, List[float]]:
        """Extract confidence band around the median.

        Args:
            confidence: Fraction of paths within band.

        Returns:
            Dict with ``"lower"``, ``"median"``, ``"upper"``
            series aligned to the horizon.
        """
        if self._paths is None:
            return {
                "lower": [],
                "median": [],
                "upper": [],
            }
        lower_q = (1.0 - confidence) / 2.0 * 100
        upper_q = (1.0 + confidence) / 2.0 * 100
        lower = np.percentile(self._paths, lower_q, axis=0).tolist()
        median = np.median(self._paths, axis=0).tolist()
        upper = np.percentile(self._paths, upper_q, axis=0).tolist()
        return {
            "lower": lower,
            "median": median,
            "upper": upper,
        }

    def terminal_stats(self) -> Dict[str, float]:
        """Compute statistics of terminal values.

        Returns:
            Dict with ``mean``, ``std``, ``min``,
            ``max``, ``median``, ``p5``, ``p95``.
        """
        if self._paths is None:
            return {}
        finals = self._paths[:, -1]
        return {
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals, ddof=1)),
            "min": float(np.min(finals)),
            "max": float(np.max(finals)),
            "median": float(np.median(finals)),
            "p5": float(np.percentile(finals, 5)),
            "p95": float(np.percentile(finals, 95)),
        }

    def max_drawdown_distribution(
        self,
    ) -> Dict[str, float]:
        """Compute max-drawdown stats across all paths.

        Returns:
            Dict with ``mean``, ``median``, ``p95``,
            ``worst`` drawdowns (positive values).
        """
        if self._paths is None:
            return {}
        mdd_list: List[float] = []
        for i in range(self._paths.shape[0]):
            path = self._paths[i]
            peak = np.maximum.accumulate(path)
            dd = (peak - path) / np.where(peak > 0, peak, 1.0)
            mdd_list.append(float(np.max(dd)))
        arr = np.array(mdd_list)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "worst": float(np.max(arr)),
        }

    def probability_of_loss(self) -> float:
        """Fraction of paths ending below starting value.

        Returns:
            Proportion in ``[0, 1]``.
        """
        if self._paths is None:
            return 0.0
        start = self._paths[:, 0]
        end = self._paths[:, -1]
        return float(np.mean(end < start))

    def summary(self) -> Dict[str, Any]:
        """Full summary combining all analytics."""
        return {
            "terminal": self.terminal_stats(),
            "confidence_95": self.confidence_interval(0.95),
            "max_drawdown": self.max_drawdown_distribution(),
            "prob_loss": self.probability_of_loss(),
        }
