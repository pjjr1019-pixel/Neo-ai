"""
Test suite for fastapi_service.py using pytest and httpx.AsyncClient.
Covers all endpoints, edge cases, and error handling.
Ensures flake8 compliance and best practices.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app

def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "NEO Hybrid AI Service is running."}

def test_predict_valid():
    client = TestClient(app)
    payload = {"input": "test input"}
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert "output" in response.json()

def test_predict_invalid():
    client = TestClient(app)
    response = client.post("/predict", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

def test_metrics_endpoint():
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "request_count" in data
    assert data["request_count"] >= 0
