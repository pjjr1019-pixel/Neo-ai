"""Model validation and promotion helpers for CI/CD workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ValidationThresholds:
    """Thresholds required for model promotion."""

    min_r2: float = 0.10
    max_mse: float = 10_000.0


def validate_model_metrics(
    metrics: Dict[str, float],
    thresholds: ValidationThresholds | None = None,
) -> bool:
    """Validate model metrics against promotion thresholds."""
    limits = thresholds or ValidationThresholds()
    r2 = float(metrics.get("r2_ensemble", -1.0))
    mse = float(metrics.get("mse_ensemble", float("inf")))
    return r2 >= limits.min_r2 and mse <= limits.max_mse


def validate_and_promote(
    candidate_metrics: Dict[str, float],
    current_metrics: Dict[str, float] | None = None,
    thresholds: ValidationThresholds | None = None,
) -> Dict[str, object]:
    """Decide whether a candidate model should be promoted.

    Promotion requires:
    - Candidate passes absolute thresholds.
    - Candidate is not worse than current model on key metrics.
    """
    limits = thresholds or ValidationThresholds()
    passed = validate_model_metrics(candidate_metrics, limits)
    if not passed:
        return {"promote": False, "reason": "thresholds_failed"}

    if current_metrics:
        current_r2 = float(current_metrics.get("r2_ensemble", -1.0))
        current_mse = float(current_metrics.get("mse_ensemble", float("inf")))
        cand_r2 = float(candidate_metrics.get("r2_ensemble", -1.0))
        cand_mse = float(candidate_metrics.get("mse_ensemble", float("inf")))

        if cand_r2 < current_r2 or cand_mse > current_mse:
            return {"promote": False, "reason": "candidate_regression"}

    return {"promote": True, "reason": "validated"}
