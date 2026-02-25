from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app, learning_logic

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


def test_predict_exception(monkeypatch) -> None:
    """
    Test /predict endpoint exception handling.
    Monkeypatch logging.info to raise an exception and verify error response.
    """
    import python_ai.fastapi_service.fastapi_service as fas
    def raise_exception(*args, **kwargs):
        raise Exception("Test error")
    monkeypatch.setattr(fas.logging, "info", raise_exception)
    response = client.post("/predict", json={"price": 150, "volume": 10})
    data = response.json()
    assert response.status_code == 200
    assert data["action"] == "error"
    assert data["confidence"] == 0.0
    assert data["risk"] is None


from python_ai.fastapi_service.fastapi_service import get_learning_logic, learning_logic

def test_learn_success():
    """Test /learn endpoint for successful learning."""
    payload = {"features": [1, 2, 3], "target": 1.0}
    app.dependency_overrides[get_learning_logic] = lambda: learning_logic
    client = TestClient(app)
    try:
        response = client.post("/learn", json=payload)
        data = response.json()
        assert response.status_code == 200
        assert data["status"] == "learning triggered"
        assert data["received"] == payload
    finally:
        app.dependency_overrides.pop(get_learning_logic, None)


from python_ai.fastapi_service.fastapi_service import get_learning_logic

from python_ai.fastapi_service.fastapi_service import app, get_learning_logic

def test_learn_exception() -> None:
    """
    Test /learn endpoint exception handling using dependency override.
    Should return status 'error' and received data.
    """
    def error_logic(data):
        raise Exception("Test error")
    app.dependency_overrides[get_learning_logic] = lambda: error_logic
    client = TestClient(app)
    try:
        response = client.post(
            "/learn",
            json={"features": [1, 2, 3], "target": 1.0}
        )
        data = response.json()
        assert response.status_code == 200
        assert data["status"] == "error"
        assert data["received"] == {"features": [1, 2, 3], "target": 1.0}
    finally:
        app.dependency_overrides.pop(get_learning_logic, None)
