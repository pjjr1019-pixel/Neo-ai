"""
FastAPI app for NEO Hybrid AI service.

Exposes root, predict, learn, metrics, and explain endpoints.
Uses real ML model for predictions with confidence calibration.
"""

from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from python_ai.data_pipeline import get_pipeline
from python_ai.ml_model import get_model

app = FastAPI()


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


def learning_logic(data):
    """Dummy learning logic for demonstration."""
    if not isinstance(data, dict):
        return {"status": "error"}
    features = data.get("features")
    target = data.get("target")
    if not isinstance(features, list) or target is None:
        return {"status": "error"}
    return {"status": "learning triggered"}


class LearnInput(BaseModel):
    """Input schema for /learn endpoint."""

    features: list
    target: int


@app.post("/learn")
def learn(payload: LearnInput):
    """Learn endpoint for model training."""
    logic = get_learning_logic()
    result = logic(payload.model_dump())
    return result


@app.get("/metrics")
def metrics():
    """Metrics endpoint returns request count."""
    return {"request_count": 0}


@app.get("/explain")
def explain() -> Dict[str, Any]:
    """Return real feature importance from model.

    Returns:
        Dict with feature importance from Random Forest and Gradient  Boosting.
    """
    import numpy as np

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
