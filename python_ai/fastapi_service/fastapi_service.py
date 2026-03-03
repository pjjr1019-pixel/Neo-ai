"""
FastAPI app for NEO Hybrid AI service.

Exposes root, predict, learn, metrics, health, and explain endpoints.
Uses real ML model for predictions with confidence calibration.
Auth-protected routes use JWT / API-key dependency injection.
"""

import asyncio
import logging
import platform
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, List, Type

import numpy as np
from fastapi import Depends, FastAPI, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from python_ai.auth.dependencies import get_current_user
from python_ai.auth.jwt_handler import decode_token
from python_ai.auth.models import User
from python_ai.config.settings import get_settings
from python_ai.data_lineage import get_lineage_store
from python_ai.data_pipeline import DataPipeline, get_pipeline
from python_ai.explainability import get_global_importance
from python_ai.logging import LogConfig, LogFormat, LogLevel, setup_logging
from python_ai.middleware import (
    CorrelationIDMiddleware,
    RequestLoggingMiddleware,
    register_exception_handlers,
)
from python_ai.ml_model import MLModel, get_model
from python_ai.rate_limiter import RateLimitMiddleware
from python_ai.ws_signal_stream import websocket_signal_handler

logger = logging.getLogger(__name__)

_settings = get_settings()
_VERSION = "0.6.0"
_default_response_class: Type[Response]

try:  # pragma: no cover - optional dependency path
    import orjson  # noqa: F401
    from fastapi.responses import ORJSONResponse

    _default_response_class = ORJSONResponse
except Exception:  # pragma: no cover - optional dependency path
    _default_response_class = JSONResponse


# ── Lifespan (startup / shutdown) ─────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    Startup:
      - Log that the service is starting.
      - Pre-load the ML model so the first request is fast.

    Shutdown:
      - Flush any buffered learn samples to log.
      - Log a clean shutdown message.
    """
    logger.info("NEO Hybrid AI v%s starting up", _VERSION)

    # Configure structured logging from settings
    log_settings = _settings.logging
    log_config = LogConfig(
        level=LogLevel(log_settings.level),
        format=LogFormat(log_settings.format.lower()),
        app_name=_settings.app_name,
        enable_file_logging=log_settings.enable_file_logging,
        max_file_size_mb=log_settings.max_file_size_mb,
        backup_count=log_settings.backup_count,
    )
    setup_logging(log_config)

    # Pre-load model on startup
    get_model()
    logger.info("Model loaded, service ready")
    yield
    # ── Shutdown ──────────────────────────────────────────────
    if _learn_buffer:
        logger.warning(
            "Shutting down with %d un-trained samples in buffer",
            len(_learn_buffer),
        )
    logger.info("NEO Hybrid AI shutting down gracefully")


app = FastAPI(
    title="NEO Hybrid AI",
    version=_VERSION,
    lifespan=lifespan,
    default_response_class=_default_response_class,
)

# ── Exception handlers (must be registered before middleware) ──
register_exception_handlers(app)

# ── Middleware (order matters: last added = first executed) ────
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-API-Key",
        "X-Correlation-ID",
    ],
)

app.add_middleware(
    RateLimitMiddleware,  # type: ignore[arg-type]
    requests_per_minute=_settings.api.rate_limit_per_minute,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(CorrelationIDMiddleware)

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
_learn_lock = asyncio.Lock()
_RETRAIN_THRESHOLD: int = _settings.model.retrain_threshold


# ── Dependency-injection helpers ──────────────────────────────


def get_ml_model() -> MLModel:
    """Dependency: return the ML model singleton."""
    return get_model()


def get_data_pipeline() -> DataPipeline:
    """Dependency: return the DataPipeline singleton."""
    return get_pipeline()


class ComputeFeaturesInput(BaseModel):
    """Input schema for /compute-features endpoint."""

    symbol: str
    ohlcv_data: Dict[str, List[float]]


class PredictInput(BaseModel):
    """Input schema for /predict endpoint.

    Attributes:
        features: Dict mapping feature names to float values.
    """

    features: Dict[str, float] = Field(
        ...,
        min_length=1,
        description="Feature dict with at least one entry",
    )

    @field_validator("features")
    @classmethod
    def validate_feature_values(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Reject NaN, Inf, and extreme values."""
        import math

        for key, val in v.items():
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Feature '{key}' contains NaN or Inf")
            if abs(val) > 1e12:
                raise ValueError(
                    f"Feature '{key}' value {val} exceeds "
                    f"allowed range (±1e12)"
                )
        return v


@app.get("/")
def root() -> Dict[str, str]:
    """Root endpoint returns service status message."""
    return {"message": "NEO Hybrid AI Service is running."}


@app.get("/health")
def health() -> Dict[str, str]:
    """Public liveness probe — returns only a status string.

    Does **not** expose system details; use ``/health/details``
    (auth-protected) for diagnostics.

    Returns:
        Dict with ``status`` key.
    """
    return {"status": "ok"}


@app.get("/health/details")
def health_details(
    model: MLModel = Depends(get_ml_model),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Auth-protected health check with model, DB, and system status.

    Requires authentication (JWT or API key).

    Returns:
        Dict with status, model info, uptime, and system details.
    """
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
        "version": _VERSION,
        "checks": checks,
    }


@app.post("/compute-features")
def compute_features(
    payload: ComputeFeaturesInput,
    pipeline: DataPipeline = Depends(get_data_pipeline),
    _user: User = Depends(get_current_user),
) -> Dict[str, float]:
    """Compute features from raw OHLCV price data.

    Requires authentication (JWT or API key).

    Args:
        payload: ComputeFeaturesInput with symbol and OHLCV data.
        pipeline: Injected data pipeline.
        _user: Authenticated user.

    Returns:
        Dict with descriptive feature keys and normalized float values.
    """
    _request_counts["compute_features"] += 1
    pipeline.update_price_data(payload.symbol, payload.ohlcv_data)
    features = pipeline.compute_features(payload.symbol)
    return features


@app.post("/predict")
def predict(
    payload: PredictInput,
    model: MLModel = Depends(get_ml_model),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Predict endpoint for real model inference.

    Requires authentication (JWT or API key).

    Args:
        payload: PredictInput with features dict.
        model: Injected ML model.
        _user: Authenticated user.

    Returns:
        Dict with prediction, confidence, and signal.
    """
    _request_counts["predict"] += 1
    pred, confidence, signal = model.predict(payload.features)
    return {
        "prediction": float(pred),
        "confidence": float(confidence),
        "signal": signal,
    }


def get_learning_logic() -> Callable[..., Dict[str, Any]]:
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
    """Input schema for /learn endpoint.

    Attributes:
        features: List of numeric feature values.
        target: Target value for supervised learning.
    """

    features: List[float] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Feature vector (list of floats)",
    )
    target: float = Field(
        ...,
        ge=-1e6,
        le=1e6,
        description="Target value for training",
    )

    @field_validator("features")
    @classmethod
    def validate_features_finite(cls, v: List[float]) -> List[float]:
        """Reject NaN and Inf in feature vector."""
        import math

        for i, val in enumerate(v):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Feature at index {i} contains NaN or Inf")
        return v


@app.post("/learn")
async def learn(
    payload: LearnInput,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Learn endpoint for incremental model training.

    Requires authentication (JWT or API key).
    Uses an asyncio lock to prevent concurrent buffer mutations.

    Args:
        payload: LearnInput with features list and target float.
        _user: Authenticated user.

    Returns:
        Dict with buffered/retrained status.
    """
    _request_counts["learn"] += 1
    async with _learn_lock:
        logic = get_learning_logic()
        result: Dict[str, Any] = logic(payload.model_dump())
    return result


@app.get("/metrics")
def metrics(
    model: MLModel = Depends(get_ml_model),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return real request counts and model training metrics.

    Requires authentication (JWT or API key).

    Args:
        model: Injected ML model.
        _user: Authenticated user.

    Returns:
        Dict with request counts, uptime, and model info.
    """
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
def metrics_prometheus(
    model: MLModel = Depends(get_ml_model),
    _user: User = Depends(get_current_user),
) -> Response:
    """Expose metrics in Prometheus text exposition format.

    Requires authentication (JWT or API key).

    Args:
        model: Injected ML model.
        _user: Authenticated user.

    Returns:
        Plain-text response with Prometheus-compatible metrics.
    """
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
def explain(
    model: MLModel = Depends(get_ml_model),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return feature importance from model.

    Uses SHAP TreeExplainer when the ``shap`` library is
    installed, otherwise falls back to sklearn gini importances.

    Requires authentication (JWT or API key).

    Args:
        model: Injected ML model.
        _user: Authenticated user.

    Returns:
        Dict with feature importance and explanation method.
    """
    _request_counts["explain"] += 1
    result = get_global_importance(model)
    result["model_type"] = "RandomForest + GradientBoosting Ensemble"
    return result


@app.get("/lineage")
def lineage_summary(
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return a summary of the data/model lineage store.

    Requires authentication (JWT or API key).

    Args:
        _user: Authenticated user.

    Returns:
        Dict with lineage record counts by type.
    """
    store = get_lineage_store()
    return store.summary()


# ── WebSocket endpoint ────────────────────────────────────────


@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket) -> None:
    """Stream live trading signals over WebSocket.

    Validates a JWT token from the ``token`` query parameter
    before accepting the connection.  Returns close code 4001
    if authentication fails.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001)
        return
    token_data = decode_token(token)
    if token_data is None or not token_data.username:
        await websocket.close(code=4001)
        return
    await websocket.accept()
    await websocket_signal_handler(websocket)
