"""
A/B Testing Framework for Trading Strategies in NEO.

Allows running two strategy variants simultaneously
with traffic splitting, then compares statistical
significance of performance differences.
"""

import logging
import math
import random
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StrategyVariant:
    """Container for one variant in an A/B test.

    Attributes:
        name: Variant label (e.g. ``"control"``).
        weight: Traffic share (0-1).
        results: Collected metric values.
    """

    def __init__(self, name: str, weight: float = 0.5) -> None:
        """Initialise a variant."""
        self.name = name
        self.weight = weight
        self.results: List[float] = []

    def record(self, metric: float) -> None:
        """Record a single metric observation."""
        self.results.append(metric)

    @property
    def mean(self) -> float:
        """Mean of recorded results."""
        if not self.results:
            return 0.0
        return sum(self.results) / len(self.results)

    @property
    def std(self) -> float:
        """Standard deviation of results."""
        if len(self.results) < 2:
            return 0.0
        m = self.mean
        var = sum((x - m) ** 2 for x in self.results) / (len(self.results) - 1)
        return float(var**0.5)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict."""
        return {
            "name": self.name,
            "weight": self.weight,
            "n": len(self.results),
            "mean": self.mean,
            "std": self.std,
        }


class ABTest:
    """A/B test harness for two strategy variants.

    Randomly assigns incoming signals to variant A or B
    based on configured weights, collects results, and
    computes a Welch's t-test for significance.

    Args:
        name: Experiment name.
        variant_a_name: Label for the control variant.
        variant_b_name: Label for the treatment variant.
        split: Fraction of traffic routed to variant B
            (0.0–1.0, default 0.5).
    """

    def __init__(
        self,
        name: str = "strategy_ab",
        variant_a_name: str = "control",
        variant_b_name: str = "treatment",
        split: float = 0.5,
    ) -> None:
        """Initialise the A/B test."""
        self.name = name
        self.variant_a = StrategyVariant(variant_a_name, 1.0 - split)
        self.variant_b = StrategyVariant(variant_b_name, split)
        self._created = time.time()

    # ── assignment ────────────────────────────────────

    def assign(self) -> str:
        """Randomly assign to a variant.

        Returns:
            The ``name`` of the assigned variant.
        """
        if random.random() < self.variant_b.weight:
            return self.variant_b.name
        return self.variant_a.name

    def record(self, variant_name: str, metric: float) -> None:
        """Record a result for a variant.

        Args:
            variant_name: Which variant produced this.
            metric: The performance metric value.
        """
        if variant_name == self.variant_a.name:
            self.variant_a.record(metric)
        elif variant_name == self.variant_b.name:
            self.variant_b.record(metric)
        else:
            logger.warning("Unknown variant '%s'", variant_name)

    # ── statistics ────────────────────────────────────

    def welch_t_test(
        self,
    ) -> Dict[str, Optional[float]]:
        """Compute Welch's unequal-variance t-test.

        Returns:
            Dict with ``t_stat``, ``df``, ``p_value``,
            ``significant`` (at α=0.05), and ``winner``.
        """
        na = len(self.variant_a.results)
        nb = len(self.variant_b.results)
        if na < 2 or nb < 2:
            return {
                "t_stat": None,
                "df": None,
                "p_value": None,
                "significant": False,
                "winner": None,
            }

        ma = self.variant_a.mean
        mb = self.variant_b.mean
        sa = self.variant_a.std
        sb = self.variant_b.std

        se = ((sa**2 / na) + (sb**2 / nb)) ** 0.5
        if se < 1e-12:
            return {
                "t_stat": 0.0,
                "df": na + nb - 2,
                "p_value": 1.0,
                "significant": False,
                "winner": None,
            }

        t_stat = (ma - mb) / se

        # Welch-Satterthwaite degrees of freedom
        num = (sa**2 / na + sb**2 / nb) ** 2
        denom = (sa**2 / na) ** 2 / (na - 1) + (sb**2 / nb) ** 2 / (nb - 1)
        df = num / denom if denom > 0 else 1.0

        # Approximate p-value using normal for large df
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / (2**0.5))))

        significant = p_value < 0.05
        winner = None
        if significant:
            winner = self.variant_a.name if ma > mb else self.variant_b.name

        return {
            "t_stat": t_stat,
            "df": df,
            "p_value": p_value,
            "significant": significant,
            "winner": winner,
        }

    def summary(self) -> Dict[str, Any]:
        """Full experiment summary."""
        return {
            "name": self.name,
            "variant_a": self.variant_a.to_dict(),
            "variant_b": self.variant_b.to_dict(),
            "test": self.welch_t_test(),
        }
