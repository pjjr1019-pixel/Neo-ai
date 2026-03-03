"""Deployment strategies for rollback/canary promotion decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CanaryConfig:
    """Canary rollout policy."""

    error_rate_threshold: float = 0.02
    latency_p95_threshold_ms: float = 250.0


def evaluate_canary(
    candidate_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    config: CanaryConfig | None = None,
) -> Dict[str, object]:
    """Evaluate canary metrics and decide promote/rollback."""
    policy = config or CanaryConfig()
    cand_err = float(candidate_metrics.get("error_rate", 1.0))
    base_err = float(baseline_metrics.get("error_rate", 1.0))
    cand_p95 = float(candidate_metrics.get("latency_p95_ms", float("inf")))

    if cand_err > policy.error_rate_threshold or cand_err > base_err:
        return {"action": "rollback", "reason": "error_rate"}
    if cand_p95 > policy.latency_p95_threshold_ms:
        return {"action": "rollback", "reason": "latency"}
    return {"action": "promote", "reason": "canary_healthy"}


def rollback_plan() -> Dict[str, str]:
    """Return deterministic rollback steps for automation scripts."""
    return {
        "step_1": "Route 100% traffic to previous stable version",
        "step_2": "Invalidate candidate deployment cache",
        "step_3": "Notify on-call and attach canary metrics",
    }
