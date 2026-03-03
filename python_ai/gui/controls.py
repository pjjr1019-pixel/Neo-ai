"""GUI controls and input validation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class ControlValidationResult:
    """Validation result for one UI control submission."""

    valid: bool
    errors: List[str]


def validate_backtest_controls(
    *,
    symbol: str,
    lookback_days: int,
    risk_limit: float,
    mode: str,
    allowed_modes: Iterable[str] = ("paper", "live"),
) -> ControlValidationResult:
    """Validate backtest and trading control panel inputs."""
    errors: List[str] = []
    if not symbol:
        errors.append("symbol is required")
    if lookback_days <= 0:
        errors.append("lookback_days must be > 0")
    if not (0.0 < risk_limit <= 1.0):
        errors.append("risk_limit must be in (0, 1]")
    if mode not in set(allowed_modes):
        errors.append("mode is invalid")
    return ControlValidationResult(valid=not errors, errors=errors)
