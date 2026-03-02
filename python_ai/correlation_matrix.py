"""
Cross-Asset Correlation Matrix for NEO Hybrid AI.

Computes correlation and covariance matrices from
multi-asset return series.  Supports rolling windows
and exponentially weighted (EWMA) correlations.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_correlation(
    asset_returns: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise Pearson correlation matrix.

    Args:
        asset_returns: Mapping of asset symbol to
            return series (all same length).

    Returns:
        Nested dict ``{asset_a: {asset_b: corr}}``.
    """
    symbols = sorted(asset_returns.keys())
    if len(symbols) < 2:
        return {s: {s: 1.0} for s in symbols}

    min_len = min(len(asset_returns[s]) for s in symbols)
    if min_len < 2:
        return {s: {s: 1.0} for s in symbols}

    matrix = np.column_stack(
        [
            np.array(asset_returns[s][:min_len], dtype=np.float64)
            for s in symbols
        ]
    )
    corr_matrix = np.corrcoef(matrix, rowvar=False)

    result: Dict[str, Dict[str, float]] = {}
    for i, sa in enumerate(symbols):
        result[sa] = {}
        for j, sb in enumerate(symbols):
            result[sa][sb] = float(corr_matrix[i, j])  # type: ignore[index]
    return result


def compute_covariance(
    asset_returns: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise covariance matrix.

    Args:
        asset_returns: Asset → return series mapping.

    Returns:
        Nested dict ``{asset_a: {asset_b: cov}}``.
    """
    symbols = sorted(asset_returns.keys())
    if len(symbols) < 2:
        return {s: {s: 0.0} for s in symbols}

    min_len = min(len(asset_returns[s]) for s in symbols)
    if min_len < 2:
        return {s: {s: 0.0} for s in symbols}

    matrix = np.column_stack(
        [
            np.array(asset_returns[s][:min_len], dtype=np.float64)
            for s in symbols
        ]
    )
    cov = np.cov(matrix, rowvar=False)

    result: Dict[str, Dict[str, float]] = {}
    for i, sa in enumerate(symbols):
        result[sa] = {}
        for j, sb in enumerate(symbols):
            result[sa][sb] = float(cov[i, j])
    return result


def rolling_correlation(
    returns_a: List[float],
    returns_b: List[float],
    window: int = 30,
) -> List[Optional[float]]:
    """Compute rolling Pearson correlation.

    Args:
        returns_a: First return series.
        returns_b: Second return series.
        window: Rolling window size.

    Returns:
        List of correlation values (``None`` for the
        initial warm-up period).
    """
    min_len = min(len(returns_a), len(returns_b))
    a = np.array(returns_a[:min_len], dtype=np.float64)
    b = np.array(returns_b[:min_len], dtype=np.float64)

    result: List[Optional[float]] = []
    for i in range(min_len):
        if i < window - 1:
            result.append(None)
        else:
            wa = a[i - window + 1 : i + 1]
            wb = b[i - window + 1 : i + 1]
            corr = np.corrcoef(wa, wb)[0, 1]
            result.append(float(corr))
    return result


def ewma_correlation(
    returns_a: List[float],
    returns_b: List[float],
    span: int = 30,
) -> List[Optional[float]]:
    """Compute exponentially weighted correlation.

    Uses EWMA variance / covariance from RiskMetrics.

    Args:
        returns_a: First return series.
        returns_b: Second return series.
        span: EWMA span (larger = slower decay).

    Returns:
        List of EWMA correlations (``None`` for first).
    """
    alpha = 2.0 / (span + 1)
    min_len = min(len(returns_a), len(returns_b))
    a = np.array(returns_a[:min_len], dtype=np.float64)
    b = np.array(returns_b[:min_len], dtype=np.float64)

    if min_len < 2:
        return [None] * min_len

    var_a = float((a[0]) ** 2)
    var_b = float((b[0]) ** 2)
    cov_ab = float(a[0] * b[0])

    result: List[Optional[float]] = [None]
    for i in range(1, min_len):
        var_a = (1 - alpha) * var_a + alpha * (a[i] ** 2)
        var_b = (1 - alpha) * var_b + alpha * (b[i] ** 2)
        cov_ab = (1 - alpha) * cov_ab + alpha * (a[i] * b[i])
        denom = (var_a * var_b) ** 0.5
        if denom > 1e-12:
            result.append(float(cov_ab / denom))
        else:
            result.append(None)
    return result


def diversification_ratio(
    asset_returns: Dict[str, List[float]],
    weights: Dict[str, float],
) -> float:
    """Compute the diversification ratio.

    DR = weighted average volatility / portfolio volatility.
    A DR > 1 indicates diversification benefit.

    Args:
        asset_returns: Asset → return series.
        weights: Portfolio weights.

    Returns:
        Diversification ratio (≥ 1.0 when diversified).
    """
    common = sorted(set(asset_returns.keys()) & set(weights.keys()))
    if len(common) < 2:
        return 1.0
    min_len = min(len(asset_returns[s]) for s in common)
    if min_len < 2:
        return 1.0

    w = np.array([weights[s] for s in common])
    stds = np.array(
        [float(np.std(asset_returns[s][:min_len], ddof=1)) for s in common]
    )
    matrix = np.column_stack(
        [
            np.array(asset_returns[s][:min_len], dtype=np.float64)
            for s in common
        ]
    )
    cov = np.cov(matrix, rowvar=False)
    port_var = float(w @ cov @ w)
    port_std = port_var**0.5 if port_var > 0 else 1e-12
    weighted_std = float(np.dot(np.abs(w), stds))
    return weighted_std / port_std if port_std > 0 else 1.0
