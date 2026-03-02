"""
FastAPI app for NEO Hybrid AI service.

Exposes root, predict, learn, metrics, health, and explain endpoints.
Uses real ML model for predictions with confidence calibration.
"""

import logging
import platform
import time
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, Response
from pydantic import BaseModel

from python_ai.data_pipeline import get_pipeline
from python_ai.ml_model import get_model
from python_ai.rate_limiter import RateLimitMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(title="NEO Hybrid AI", version="0.2.0")
app.add_middleware(
    RateLimitMiddleware,  # type: ignore[arg-type]
    requests_per_minute=120,
)

# ── Simple in-process counters ────────────────────────────────
_request_counts: Dict[str, int] = {
    "predict": 0,
    "learn": 0,
    "compute_features": 0,
    "explain": 0,
}
_start_time = time.time()

# ── Sample buffer for incremental learning ────────────────────
_learn_buffer: List[Dict[str, Any]] = []
_RETRAIN_THRESHOLD = 50  # retrain after this many samples


class ComputeFeaturesInput(BaseModel):
    """Input schema for /compute-features endpoint."""

    symbol: str
    ohlcv_data: Dict[str, List[float]]


class PredictInput(BaseModel):
    """Input schema for /predict endpoint."""

    features: Dict[str, float]


@app.get("/")
def root():
    """Root endpoint returns service status message."""
    return {"message": "NEO Hybrid AI Service is running."}


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check with model, DB, and system status.

    Returns:
        Dict with status, model info, uptime, and system details.
    """
    model = get_model()
    uptime = time.time() - _start_time
    status = "healthy"
    checks: Dict[str, Any] = {}

    # Model health
    checks["model"] = {
        "status": "ok" if model.is_trained else "degraded",
        "trained": model.is_trained,
        "train_count": model.train_count,
    }
    if not model.is_trained:
        status = "degraded"

    # System info
    checks["system"] = {
        "python": platform.python_version(),
        "platform": platform.system(),
    }

    return {
        "status": status,
        "uptime_seconds": round(uptime, 1),
        "version": "0.2.0",
        "checks": checks,
    }


@app.post("/compute-features")
def compute_features(
    payload: ComputeFeaturesInput,
) -> Dict[str, float]:
    """Compute features from raw OHLCV price data.

    Args:
        payload: ComputeFeaturesInput with symbol and OHLCV data.

    Returns:
        Dict with f0-f9 feature keys and normalized float values.
    """
    _request_counts["compute_features"] += 1
    pipeline = get_pipeline()
    pipeline.update_price_data(payload.symbol, payload.ohlcv_data)
    features = pipeline.compute_features(payload.symbol)
    return features


@app.post("/predict")
def predict(payload: PredictInput) -> Dict[str, Any]:
    """Predict endpoint for real model inference.

    Args:
        payload: PredictInput with features dict.

    Returns:
        Dict with prediction, confidence, and signal.
    """
    _request_counts["predict"] += 1
    model = get_model()
    pred, confidence, signal = model.predict(payload.features)
    return {
        "prediction": float(pred),
        "confidence": float(confidence),
        "signal": signal,
    }


def get_learning_logic():
    """Get learning logic dependency for /learn endpoint."""
    return learning_logic


def learning_logic(data: Dict[str, Any]) -> Dict[str, Any]:
    """Buffer samples and retrain the model when threshold is met.

    Each call appends ``(features, target)`` to the in-process
    buffer.  When the buffer reaches ``_RETRAIN_THRESHOLD`` the
    model is retrained on the full buffer and then cleared.

    Args:
        data: Dict with ``features`` (list of floats) and
              ``target`` (numeric).

    Returns:
        Status dict with buffer size and optional training metrics.
    """
    if not isinstance(data, dict):
        return {"status": "error", "detail": "payload must be dict"}

    features = data.get("features")
    target = data.get("target")
    if not isinstance(features, list) or target is None:
        return {
            "status": "error",
            "detail": "missing features or target",
        }

    _learn_buffer.append(
        {"features": features, "target": float(target)},
    )

    buffer_size = len(_learn_buffer)

    if buffer_size >= _RETRAIN_THRESHOLD:
        X = np.array([s["features"] for s in _learn_buffer])
        y = np.array([s["target"] for s in _learn_buffer])

        model = get_model()
        metrics = model.train(X, y)
        _learn_buffer.clear()

        logger.info(
            "Retrained model with %d buffered samples",
            buffer_size,
        )
        return {
            "status": "retrained",
            "samples": buffer_size,
            "metrics": metrics,
        }

    return {
        "status": "buffered",
        "buffer_size": buffer_size,
        "retrain_at": _RETRAIN_THRESHOLD,
    }


class LearnInput(BaseModel):
    """Input schema for /learn endpoint."""

    features: list
    target: float


@app.post("/learn")
def learn(payload: LearnInput) -> Dict[str, Any]:
    """Learn endpoint for incremental model training."""
    _request_counts["learn"] += 1
    logic = get_learning_logic()
    result: Dict[str, Any] = logic(payload.model_dump())
    return result


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Return real request counts and model training metrics."""
    model = get_model()
    uptime = time.time() - _start_time
    return {
        "request_counts": dict(_request_counts),
        "total_requests": sum(_request_counts.values()),
        "uptime_seconds": round(uptime, 1),
        "model_trained": model.is_trained,
        "train_count": model.train_count,
        "training_metrics": model.training_metrics,
    }


@app.get("/metrics/prometheus")
def metrics_prometheus() -> Response:
    """Expose metrics in Prometheus text exposition format.

    Returns:
        Plain-text response with Prometheus-compatible metrics.
    """
    model = get_model()
    uptime = time.time() - _start_time
    lines: List[str] = [
        "# HELP neo_requests_total Total requests by endpoint.",
        "# TYPE neo_requests_total counter",
    ]
    for endpoint, count in _request_counts.items():
        lines.append(f'neo_requests_total{{endpoint="{endpoint}"}} {count}')
    lines += [
        "# HELP neo_uptime_seconds Service uptime in seconds.",
        "# TYPE neo_uptime_seconds gauge",
        f"neo_uptime_seconds {uptime:.1f}",
        "# HELP neo_model_trained Whether the model is trained.",
        "# TYPE neo_model_trained gauge",
        f"neo_model_trained {1 if model.is_trained else 0}",
        "# HELP neo_train_count Number of training runs.",
        "# TYPE neo_train_count counter",
        f"neo_train_count {model.train_count}",
    ]

    if model.training_metrics:
        r2 = model.training_metrics.get("r2_ensemble", 0)
        mse = model.training_metrics.get("mse_ensemble", 0)
        lines += [
            "# HELP neo_model_r2 R-squared of ensemble.",
            "# TYPE neo_model_r2 gauge",
            f"neo_model_r2 {r2:.6f}",
            "# HELP neo_model_mse Mean squared error of ensemble.",
            "# TYPE neo_model_mse gauge",
            f"neo_model_mse {mse:.6f}",
        ]

    body = "\n".join(lines) + "\n"
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/explain")
def explain() -> Dict[str, Any]:
    """Return real feature importance from model.

    Returns:
        Dict with feature importance from Random Forest and Gradient  Boosting.
    """
    _request_counts["explain"] += 1

    model = get_model()
    rf_importance: np.ndarray = (
        model.rf_model.feature_importances_
        if model.rf_model is not None
        else np.array([])
    )
    gb_importance: np.ndarray = (
        model.gb_model.feature_importances_
        if model.gb_model is not None
        else np.array([])
    )

    # Average importance across models
    avg_importance: np.ndarray = (
        (rf_importance + gb_importance) / 2.0
        if len(rf_importance) > 0
        else np.array([])
    )

    return {
        "feature_importance": {
            f"feature_{i}": float(imp) for i, imp in enumerate(avg_importance)
        },
        "explanation": "Feature importance from ensemble (RF + GB).",
        "model_type": "RandomForest + GradientBoosting Ensemble",
    }
