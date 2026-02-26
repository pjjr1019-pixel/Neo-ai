import pytest
from python_ai.fastapi_service.fastapi_service import (
    app,
    get_learning_logic,
    learning_logic,
)
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "payload,expected_status,expected_key",
    [
        (
            {"features": [1, 2, 3], "target": 1},
            200,
            "status",
        ),
        (
            {"features": [], "target": 0},
            200,
            "status",
        ),  # edge: empty features
        (
            {"features": [1, 2, 3]},
            422,
            None,
        ),  # missing target
        (
            {},
            422,
            None,
        ),  # missing all fields
        (
            {"features": [1, 2, 3], "target": "badtype"},
            422,
            None,
        ),  # wrong type
    ],
)
def test_learn(payload, expected_status, expected_key):
    """Comprehensive /learn endpoint test: valid, edge, and error cases."""
    app.dependency_overrides[get_learning_logic] = lambda: learning_logic
    client = TestClient(app)
    try:
        response = client.post("/learn", json=payload)
        assert response.status_code == expected_status
        if expected_key:
            assert expected_key in response.json()
    finally:
        app.dependency_overrides.pop(get_learning_logic, None)


def test_learn_dependency_override():
    """Test dependency override for /learn endpoint."""
    app.dependency_overrides[get_learning_logic] = lambda: learning_logic
    client = TestClient(app)
    response = client.post("/learn", json={"features": [1, 2, 3], "target": 1})
    assert response.status_code == 200
    app.dependency_overrides.pop(get_learning_logic, None)


def test_learn_error_handling():
    """Test error handling for /learn endpoint with invalid input."""
    app.dependency_overrides[get_learning_logic] = lambda: learning_logic
    client = TestClient(app)
    payload = {
        "features": "notalist",
        "target": None
    }
    response = client.post("/learn", json=payload)
    assert response.status_code == 422
    app.dependency_overrides.pop(get_learning_logic, None)


def test_learn_invalid_type():
    """Test /learn endpoint with completely invalid types."""
    app.dependency_overrides[get_learning_logic] = lambda: learning_logic
    client = TestClient(app)
    response = client.post("/learn", json={"features": 123, "target": "abc"})
    assert response.status_code == 422
    app.dependency_overrides.pop(get_learning_logic, None)


def test_learn_non_dict_payload():
    """Test learning_logic with non-dict payload (should return error)."""
    app.dependency_overrides[get_learning_logic] = lambda: learning_logic
    client = TestClient(app)
    response = client.post("/learn", json=None)
    assert response.status_code == 422
    app.dependency_overrides.pop(get_learning_logic, None)


def test_learn_missing_features_and_target():
    """Test /learn endpoint with missing features and target."""
    app.dependency_overrides[get_learning_logic] = lambda: learning_logic
    client = TestClient(app)
    response = client.post("/learn", json={})
    assert response.status_code == 422
    app.dependency_overrides.pop(get_learning_logic, None)
