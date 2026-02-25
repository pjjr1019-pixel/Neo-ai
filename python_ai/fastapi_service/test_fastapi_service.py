import pytest
from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import (
    app,
    get_learning_logic,
    learning_logic
)



@pytest.mark.parametrize(
    "payload,expected_status,expected_key",
    [
        ({"features": [1, 2, 3], "target": 1.0}, 200, "status"),
        ({"features": [], "target": 0}, 200, "status"),  # edge: empty features
        ({"features": [1, 2, 3]}, 422, None),  # missing target
        ({}, 422, None),  # missing all fields
        ( {"features": [1, 2, 3], "target": "badtype"}, 422, None),  # wrong type
    ]
)
def test_learn_cases(payload, expected_status, expected_key):
    """Test /learn endpoint with valid, edge, and error cases."""
    app.dependency_overrides[get_learning_logic] = lambda: learning_logic
    client = TestClient(app)
    try:
        response = client.post(
            "/learn",
            json=payload
        )
        assert response.status_code == expected_status
        if expected_key:
            assert expected_key in response.json()
    finally:
        app.dependency_overrides.pop(get_learning_logic, None)



def test_metrics_endpoint():
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "avg_latency" in data
    assert "throughput" in data
    assert "memory_mb" in data
    assert "cpu_percent" in data



def test_predict_headers():
    client = TestClient(app)
    payload = {"price": 123.45, "volume": 1000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "X-Latency" in response.headers
    assert "X-Throughput" in response.headers
    assert "X-Memory-MB" in response.headers
    assert "X-CPU-Percent" in response.headers
