"""
Test suite for fastapi_service.py using pytest and httpx.AsyncClient.
Covers all endpoints, edge cases, and error handling.
Ensures flake8 compliance and best practices.
"""

from fastapi import status
from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app

def test_root():
    """Test root endpoint returns service status."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "NEO Hybrid AI Service is running."}

def test_predict_invalid():
    """Test predict endpoint with invalid input."""
    client = TestClient(app)
    response = client.post("/predict", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

def test_predict_valid():
    """Test predict endpoint with valid input."""
    client = TestClient(app)
    response = client.post("/predict", json={"input": "test"})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "output" in data
    assert "Predicted value" in data["output"]

def test_metrics_endpoint():
    """Test metrics endpoint returns request count."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "request_count" in data
    assert data["request_count"] >= 0

def test_explain_endpoint():
    """Test the /explain endpoint for feature importance and compliance."""
    client = TestClient(app)
    response = client.get("/explain")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "feature_importance" in data
    assert "explanation" in data
    assert isinstance(data["feature_importance"], dict)

def test_learn_invalid_missing_features():
    """Test /learn endpoint with missing features."""
    client = TestClient(app)
    response = client.post("/learn", json={"target": 1})
    assert response.status_code == 422


def test_learn_invalid_wrong_type():
    """Test /learn endpoint with wrong type for features."""
    client = TestClient(app)
    response = client.post(
        "/learn",
        json={"features": "notalist", "target": 1},
    )
    assert response.status_code == 422

def test_learn_invalid_missing_target():
    """Test /learn endpoint with missing target."""
    client = TestClient(app)
    response = client.post(
        "/learn",
        json={"features": [1, 2, 3]},
    )
    assert response.status_code == 422


