from fastapi.testclient import TestClient

from python_ai.fastapi_service.fastapi_service import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns the service status message."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"].startswith("NEO Hybrid AI Service")


def test_predict_endpoint_valid():
    """Test the predict endpoint with valid input returns a prediction."""
    payload = {"features": {"f0": 1.0, "f1": 2.0}}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    resp_data = resp.json()
    assert "prediction" in resp_data
    assert "confidence" in resp_data
    assert "signal" in resp_data


def test_predict_endpoint_invalid():
    """Test the predict endpoint with invalid input returns 422 error."""
    resp = client.post("/predict", json={"wrong": "field"})
    assert resp.status_code == 422


def test_learn_endpoint_valid():
    """Test the learn endpoint with valid input buffers sample."""
    payload = {"features": [1, 2, 3], "target": 1.0}
    resp = client.post("/learn", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "buffered"


def test_learn_endpoint_invalid():
    """Test the learn endpoint with invalid input returns 422 error."""
    # Missing features
    resp = client.post("/learn", json={"target": 1})
    assert resp.status_code == 422
    # Wrong types
    resp = client.post("/learn", json={"features": "notalist", "target": 1})
    assert resp.status_code == 422


def test_metrics_endpoint():
    """Test the metrics endpoint returns request counts."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "request_counts" in resp.json()


def test_explain_endpoint():
    """Test the explain endpoint returns feature importance and explanation."""
    resp = client.get("/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert "feature_importance" in data
    assert "explanation" in data
    assert isinstance(data["feature_importance"], dict)
    assert isinstance(data["explanation"], str)
