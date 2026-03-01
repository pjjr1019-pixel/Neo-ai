"""
Test suite for fastapi_service.py using pytest and FastAPI's TestClient.
Ensures endpoint coverage, error handling, and flake8 compliance.
Follows full coding best practices: clear docstrings, proper imports,
blank lines, and error handling.
"""

from fastapi import status
from fastapi.testclient import TestClient

from python_ai.fastapi_service.fastapi_service import app


def test_root():
    """
    Test root endpoint for FastAPI service.
    Ensures the root endpoint returns the expected status and message.
    """
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    expected = {"message": "NEO Hybrid AI Service is running."}
    assert response.json() == expected


def test_predict_invalid():
    """
    Test predict endpoint with invalid input.
    Ensures the endpoint returns a 422 validation error for bad schema.
    """
    client = TestClient(app)
    response = client.post("/predict", json={"input": "test"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_metrics_endpoint():
    """
    Test metrics endpoint for FastAPI service.
    Ensures the endpoint returns the expected status and metrics data.
    """
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "request_count" in data
    assert isinstance(data["request_count"], int)


def test_explain_endpoint():
    """
    Test explain endpoint for FastAPI service.
    Ensures the endpoint returns the expected status and
    explanation data.
    """
    client = TestClient(app)
    response = client.get("/explain")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "feature_importance" in data
    assert "explanation" in data
    assert isinstance(data["feature_importance"], dict)


def test_learn_invalid_missing_features():
    """
    Test learn endpoint with missing features.
    Ensures the endpoint returns the expected error status for
    missing features.
    """
    client = TestClient(app)
    response = client.post("/learn", json={"target": 1})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_learn_invalid_wrong_type():
    """
    Test learn endpoint with wrong feature type.
    Ensures the endpoint returns the expected error status for
    wrong feature type.
    """
    client = TestClient(app)
    response = client.post(
        "/learn",
        json={"features": "notalist", "target": 1},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_learn_invalid_missing_target():
    """
    Test learn endpoint with missing target.
    Ensures the endpoint returns the expected error status for
    missing target.
    """
    client = TestClient(app)
    response = client.post("/learn", json={"input": "test"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
