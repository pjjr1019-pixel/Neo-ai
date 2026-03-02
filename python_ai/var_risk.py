"""
Portfolio Value-at-Risk (VaR) for NEO Hybrid AI.

Implements three VaR methodologies:
  * Historical simulation
  * Variance–covariance (parametric / Gaussian)
  * Monte Carlo simulation

All functions work with numpy arrays and return results
denominated in the same units as the input returns.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Historical VaR ────────────────────────────────────


def historical_var(
    returns: List[float],
    confidence: float = 0.95,
) -> float:
    """Compute VaR using historical simulation.

    Sorts past returns and picks the quantile at
    ``(1 - confidence)``.

    Args:
        returns: Series of periodic returns.
        confidence: Confidence level (e.g. 0.95 for 95%).

    Returns:
        VaR as a positive loss amount.
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    percentile = (1.0 - confidence) * 100.0
    var = float(-np.percentile(arr, percentile))
    return max(var, 0.0)


# ── Parametric (Variance-Covariance) VaR ──────────────


def parametric_var(
    returns: List[float],
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """Compute VaR using the variance-covariance method.

    Assumes returns are normally distributed.

    Args:
        returns: Series of periodic returns.
        confidence: Confidence level.
        portfolio_value: Current portfolio value
            (multiplied into the result).

    Returns:
        VaR dollar amount (positive).
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    # z-score lookup for common confidence levels.
    z_map = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_map.get(confidence, 1.645)
    var_pct = -(mu - z * sigma)
    return max(var_pct * portfolio_value, 0.0)


# ── Monte Carlo VaR ───────────────────────────────────


def monte_carlo_var(
    returns: List[float],
    confidence: float = 0.95,
    simulations: int = 10_000,
    horizon: int = 1,
    seed: Optional[int] = None,
) -> float:
    """Compute VaR via Monte Carlo simulation.

    Draws random returns from a normal distribution
    fitted to the historical data and simulates
    portfolio paths over *horizon* periods.

    Args:
        returns: Historical return series.
        confidence: Confidence level.
        simulations: Number of Monte Carlo paths.
        horizon: Lookahead periods.
        seed: Random seed for reproducibility.

    Returns:
        VaR as a positive loss amount.
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    rng = np.random.default_rng(seed)
    sims = rng.normal(mu, sigma, (simulations, horizon))
    cum = np.sum(sims, axis=1)
    percentile = (1.0 - confidence) * 100.0
    var = float(-np.percentile(cum, percentile))
    return max(var, 0.0)


# ── Conditional VaR (Expected Shortfall) ──────────────


def conditional_var(
    returns: List[float],
    confidence: float = 0.95,
) -> float:
    """Compute CVaR (Expected Shortfall).

    The average loss *beyond* the VaR threshold —
    captures tail risk better than plain VaR.

    Args:
        returns: Return series.
        confidence: Confidence level.

    Returns:
        CVaR as a positive loss amount.
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    percentile = (1.0 - confidence) * 100.0
    cutoff = np.percentile(arr, percentile)
    tail = arr[arr <= cutoff]
    if len(tail) == 0:
        return 0.0
    return float(-np.mean(tail))


# ── Portfolio-level aggregation ───────────────────────


def portfolio_var(
    asset_returns: Dict[str, List[float]],
    weights: Dict[str, float],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Compute VaR for a weighted portfolio.

    Args:
        asset_returns: Per-asset return series.
        weights: Allocation weight per asset
            (should sum to ~1.0).
        confidence: Confidence level.
        method: ``"historical"`` or ``"parametric"``.

    Returns:
        Portfolio VaR as a positive value.
    """
    common = sorted(set(asset_returns.keys()) & set(weights.keys()))
    if not common:
        return 0.0

    min_len = min(len(asset_returns[s]) for s in common)
    if min_len < 2:
        return 0.0

    port_returns: List[float] = []
    for i in range(min_len):
        r = sum(weights[s] * asset_returns[s][i] for s in common)
        port_returns.append(r)

    if method == "parametric":
        return parametric_var(port_returns, confidence)
    return historical_var(port_returns, confidence)
