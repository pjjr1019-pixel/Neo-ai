"""
Market Regime Detection for NEO Hybrid AI.

Uses Gaussian Mixture Models to classify market
conditions into discrete regimes (e.g. bull, bear,
sideways).  Regime labels and transition matrices
can be used by downstream strategy selectors to
adapt trading behaviour.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Friendly names assigned to regimes sorted by mean return.
_DEFAULT_LABELS: Dict[int, str] = {
    0: "bear",
    1: "sideways",
    2: "bull",
}


@dataclass
class RegimeResult:
    """Container for regime detection output.

    Attributes:
        labels: Integer label per observation.
        probabilities: Per-observation probability matrix
            of shape ``(n, n_regimes)``.
        means: Mean return for each regime.
        variances: Variance for each regime.
        transition_matrix: Empirical transition counts
            normalised to probabilities.
        label_names: Human-readable regime names.
    """

    labels: np.ndarray  # type: ignore[type-arg]
    probabilities: np.ndarray  # type: ignore[type-arg]
    means: np.ndarray  # type: ignore[type-arg]
    variances: np.ndarray  # type: ignore[type-arg]
    transition_matrix: np.ndarray  # type: ignore[type-arg]
    label_names: Dict[int, str] = field(
        default_factory=lambda: dict(_DEFAULT_LABELS)
    )

    def current_regime(self) -> str:
        """Return the name of the most-recent regime."""
        idx = int(self.labels[-1])
        return self.label_names.get(idx, f"regime_{idx}")


def _compute_transition_matrix(
    labels: np.ndarray,  # type: ignore[type-arg]
    n_regimes: int,
) -> np.ndarray:  # type: ignore[type-arg]
    """Build an empirical transition probability matrix.

    Args:
        labels: Sequence of integer regime labels.
        n_regimes: Number of distinct regimes.

    Returns:
        Square matrix of shape ``(n_regimes, n_regimes)``
        where entry ``[i, j]`` is ``P(regime_j | regime_i)``.
    """
    tm = np.zeros((n_regimes, n_regimes), dtype=float)
    for prev, curr in zip(labels[:-1], labels[1:]):
        tm[prev, curr] += 1.0
    # Normalise rows (avoid div-by-zero).
    row_sums = tm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return tm / row_sums  # type: ignore[no-any-return]


class RegimeDetector:
    """Gaussian-Mixture-based market regime detector.

    Fits a GMM on log-returns to discover *n_regimes*
    clusters.  Regimes are sorted by mean return so that
    the lowest-mean cluster is labelled ``0`` (bear) and
    the highest ``n_regimes - 1`` (bull).

    Args:
        n_regimes: Number of regimes to detect.
        lookback: If set, only the last *lookback*
            returns are used for fitting.
        label_names: Optional mapping of regime index
            to human-readable name.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        lookback: Optional[int] = None,
        label_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initialise the detector."""
        self._n = n_regimes
        self._lookback = lookback
        self._label_names = (
            label_names if label_names is not None else dict(_DEFAULT_LABELS)
        )
        self._fitted = False

    @property
    def n_regimes(self) -> int:
        """Number of regimes."""
        return self._n

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def detect(
        self,
        prices: Sequence[float],
    ) -> RegimeResult:
        """Detect regimes from a price series.

        Internally computes log-returns, then fits a GMM
        via the EM algorithm.

        Args:
            prices: Chronological price observations
                (at least ``n_regimes + 2`` values).

        Returns:
            A :class:`RegimeResult` with labels,
            probabilities, means, variances, and
            transition matrix.

        Raises:
            ValueError: If the series is too short.
        """
        arr = np.asarray(prices, dtype=float)
        if len(arr) < self._n + 2:
            raise ValueError(
                f"Need >= {self._n + 2} prices, " f"got {len(arr)}"
            )
        # Log-returns.
        returns = np.diff(np.log(arr))
        if self._lookback and len(returns) > self._lookback:
            returns = returns[-self._lookback :]

        labels, probs, means, variances = self._fit_gmm(returns)
        tm = _compute_transition_matrix(labels, self._n)
        self._fitted = True
        logger.info(
            "Regime detection complete: %d observations, " "%d regimes",
            len(returns),
            self._n,
        )
        return RegimeResult(
            labels=labels,
            probabilities=probs,
            means=means,
            variances=variances,
            transition_matrix=tm,
            label_names=dict(self._label_names),
        )

    # ------------------------------------------------------------------
    # Simplified EM-based GMM (no sklearn dependency)
    # ------------------------------------------------------------------

    def _fit_gmm(
        self,
        data: np.ndarray,  # type: ignore[type-arg]
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple:  # type: ignore[type-arg]
        """Fit a 1-D Gaussian Mixture Model via EM.

        Args:
            data: 1-D array of observations.
            max_iter: Maximum EM iterations.
            tol: Convergence tolerance on log-likelihood.

        Returns:
            Tuple of (labels, responsibilities, means,
            variances) sorted by ascending mean.
        """
        n = len(data)
        k = self._n

        # Initialise means via quantiles, equal weights.
        quantiles = np.linspace(0, 1, k + 2)[1:-1]
        means = np.quantile(data, quantiles)
        variances = np.full(k, np.var(data) + 1e-8)
        weights = np.ones(k) / k

        prev_ll = -np.inf
        resp = np.zeros((n, k))

        for _ in range(max_iter):
            # E-step
            for j in range(k):
                diff = data - means[j]
                resp[:, j] = (
                    weights[j]
                    * np.exp(-0.5 * diff**2 / variances[j])
                    / np.sqrt(2 * np.pi * variances[j])
                )
            row_sum = resp.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1e-12
            resp /= row_sum

            # Log-likelihood
            ll_terms = np.zeros(n)
            for j in range(k):
                diff = data - means[j]
                ll_terms += (
                    weights[j]
                    * np.exp(-0.5 * diff**2 / variances[j])
                    / np.sqrt(2 * np.pi * variances[j])
                )
            ll_terms[ll_terms <= 0] = 1e-300
            ll = float(np.sum(np.log(ll_terms)))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # M-step
            nk = resp.sum(axis=0)
            nk[nk == 0] = 1e-12
            weights = nk / n
            for j in range(k):
                means[j] = np.dot(resp[:, j], data) / nk[j]
                diff = data - means[j]
                variances[j] = np.dot(resp[:, j], diff**2) / nk[j] + 1e-8

        # Sort regimes by ascending mean.
        order = np.argsort(means)
        means = means[order]  # type: ignore[assignment]
        variances = variances[order]  # type: ignore[assignment]
        resp = resp[:, order]  # type: ignore[assignment]
        labels = np.argmax(resp, axis=1)

        return labels, resp, means, variances
