import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi.testclient import TestClient
from fastapi_service import app

client = TestClient(app)


def test_predict_buy():
    """Test /predict endpoint for 'buy' action."""
    response = client.post("/predict", json={"price": 150, "volume": 10})
    data = response.json()
    assert response.status_code == 200
    assert data["action"] == "buy"
    assert data["confidence"] == 0.95
    assert data["risk"] == 0.1


def test_predict_hold():
    """Test /predict endpoint for 'hold' action."""
    response = client.post("/predict", json={"price": 50, "volume": 10})
    data = response.json()
    assert response.status_code == 200
    assert data["action"] == "hold"
    assert data["confidence"] == 0.95
    assert data["risk"] == 0.1


def test_predict_exception(monkeypatch):
    """Test /predict endpoint exception handling."""
    def raise_exception(*args, **kwargs):
        raise Exception("Test error")
    monkeypatch.setattr("fastapi_service.logging.info", raise_exception)
    response = client.post("/predict", json={"price": 150, "volume": 10})
    data = response.json()
    assert response.status_code == 200
    assert data["action"] == "error"
    assert data["confidence"] == 0.0
    assert data["risk"] is None


def test_learn_success():
    """Test /learn endpoint for successful learning."""
    response = client.post("/learn", json={"foo": "bar"})
    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "learning triggered"
    assert data["received"] == {"foo": "bar"}


def test_learn_exception(monkeypatch):
    """Test /learn endpoint exception handling."""
    async def raise_exception(*args, **kwargs):
        raise Exception("Test error")
    monkeypatch.setattr("fastapi_service.Request.json", raise_exception)
    response = client.post("/learn", json={"foo": "bar"})
    data = response.json()
    assert response.status_code == 200
    assert "error" in data
    assert data["error"] == "Test error"
