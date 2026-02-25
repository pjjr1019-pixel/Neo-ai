import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure the parent directory is in sys.path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi_service import app

client = TestClient(app)

def test_predict_buy():
    response = client.post("/predict", json={"price": 150, "volume": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "buy"
    assert data["confidence"] == 0.95
    assert data["risk"] == 0.1

def test_predict_hold():
    response = client.post("/predict", json={"price": 50, "volume": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "hold"
    assert data["confidence"] == 0.95
    assert data["risk"] == 0.1

def test_predict_exception(monkeypatch):
    def raise_exception(*args, **kwargs):
        raise Exception("Test error")
    # Patch logging.info to raise an exception, triggering the except block
    monkeypatch.setattr("fastapi_service.logging.info", raise_exception)
    response = client.post("/predict", json={"price": 150, "volume": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "error"
    assert data["confidence"] == 0.0
    assert data["risk"] is None

def test_learn_success():
    response = client.post("/learn", json={"foo": "bar"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "learning triggered"
    assert data["received"] == {"foo": "bar"}

def test_learn_exception(monkeypatch):
    async def raise_exception(*args, **kwargs):
        raise Exception("Test error")
    monkeypatch.setattr("fastapi_service.Request.json", raise_exception)
    response = client.post("/learn", json={"foo": "bar"})
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"] == "Test error"
