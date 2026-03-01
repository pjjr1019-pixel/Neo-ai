"""
Minimal FastAPI app for NEO Hybrid AI service.
Exposes root, predict, learn, metrics, and explain endpoints.
Flake8-compliant and best practices.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictInput(BaseModel):
    """Input schema for /predict endpoint."""

    input: str


@app.get("/")
def root():
    """Root endpoint returns service status message."""
    return {"message": "NEO Hybrid AI Service is running."}


@app.post("/predict")
def predict(payload: PredictInput):
    """Predict endpoint for model inference."""
    return {"output": f"Predicted value for '{payload.input}'"}


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
def explain():
    """Return dummy feature importance for explainability."""
    # In a real system, this would return model-specific feature importances
    return {
        "feature_importance": {
            "feature1": 0.5,
            "feature2": 0.3,
            "feature3": 0.2,
        },
        "explanation": "Feature importance is illustrative.",
    }
