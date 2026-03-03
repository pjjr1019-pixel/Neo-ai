"""Tests for CI model validation and deployment strategy decisions."""

from ci.deployment_strategies import evaluate_canary, rollback_plan
from ci.model_validation import validate_and_promote, validate_model_metrics


def test_validate_model_metrics_threshold_pass() -> None:
    metrics = {"r2_ensemble": 0.35, "mse_ensemble": 120.0}
    assert validate_model_metrics(metrics)


def test_validate_and_promote_blocks_regression() -> None:
    candidate = {"r2_ensemble": 0.3, "mse_ensemble": 100.0}
    current = {"r2_ensemble": 0.5, "mse_ensemble": 80.0}
    decision = validate_and_promote(candidate, current)
    assert decision["promote"] is False


def test_evaluate_canary_promote_when_healthy() -> None:
    candidate = {"error_rate": 0.005, "latency_p95_ms": 120.0}
    baseline = {"error_rate": 0.01, "latency_p95_ms": 150.0}
    decision = evaluate_canary(candidate, baseline)
    assert decision["action"] == "promote"


def test_evaluate_canary_rolls_back_on_errors() -> None:
    candidate = {"error_rate": 0.03, "latency_p95_ms": 120.0}
    baseline = {"error_rate": 0.01, "latency_p95_ms": 150.0}
    decision = evaluate_canary(candidate, baseline)
    assert decision["action"] == "rollback"
    assert decision["reason"] == "error_rate"


def test_rollback_plan_contains_steps() -> None:
    plan = rollback_plan()
    assert "step_1" in plan
    assert "step_2" in plan
    assert "step_3" in plan
