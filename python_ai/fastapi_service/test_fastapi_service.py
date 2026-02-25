import pytest
from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app, get_learning_logic, learning_logic

client = TestClient(app)

@pytest.mark.parametrize(
    "payload,expected_action,expected_status",
    [
        ( {"price": 150, "volume": 10}, "buy", 200 ),
        ( {"price": 50, "volume": 10}, "hold", 200 ),
        ( {"price": 150, "volume": 0}, "buy", 200 ),  # edge: zero volume
        ( {"price": 0, "volume": 10}, "hold", 200 ),  # edge: zero price
        ( {"price": "bad", "volume": 10}, None, 422 ),  # wrong type
        ( {}, None, 422 ),  # missing all fields
    ]
)
def test_predict_cases(payload, expected_action, expected_status):
    """Test /predict endpoint with valid, edge, and error cases."""
    response = client.post("/predict", json=payload)
    assert response.status_code == expected_status
    if expected_action:
        assert response.json()["action"] == expected_action

def test_predict_exception(monkeypatch):
    """Test /predict endpoint exception handling."""
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

@pytest.mark.parametrize(
    "payload,expected_status,expected_key",
    [
        ( {"features": [1, 2, 3], "target": 1.0}, 200, "status" ),
        ( {"features": [], "target": 0}, 200, "status" ),  # edge: empty features
        ( {"features": [1, 2, 3]}, 422, None ),  # missing target
        ( {}, 422, None ),  # missing all fields
        ( {"features": [1, 2, 3], "target": "badtype"}, 422, None ),  # wrong type
    ]
)
def test_learn_cases(payload, expected_status, expected_key):
    """Test /learn endpoint with valid, edge, and error cases."""
    app.dependency_overrides[get_learning_logic] = lambda: learning_logic
    client = TestClient(app)
    try:
        response = client.post("/learn", json=payload)
        assert response.status_code == expected_status
        if expected_key:
            assert expected_key in response.json()
    finally:
        app.dependency_overrides.pop(get_learning_logic, None)

def test_learn_exception():
    """Test /learn endpoint exception handling using dependency override."""
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
