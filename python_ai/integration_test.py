from fastapi.testclient import TestClient

from python_ai.fastapi_service.fastapi_service import app

client = TestClient(app)


def test_predict_and_learn_integration_v2() -> None:
    """
    Integration test for /predict and /learn endpoints using FastAPI's
    TestClient. Verifies end-to-end behavior and response structure.
    """
    # Test /predict endpoint with real features
    predict_payload = {"features": {"f0": 1.0, "f1": 2.0, "f2": 3.0}}
    predict_resp = client.post("/predict", json=predict_payload)
    assert predict_resp.status_code == 200
    resp_data = predict_resp.json()
    assert "prediction" in resp_data
    assert "confidence" in resp_data
    assert "signal" in resp_data

    # Test /learn endpoint
    learn_payload = {"features": [1, 2, 3], "target": 1}
    learn_resp = client.post("/learn", json=learn_payload)
    assert learn_resp.status_code == 200
    assert learn_resp.json()["status"] == "learning triggered"

    # Test /metrics endpoint
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_data = metrics_resp.json()
    assert "request_count" in metrics_data


def test_root_endpoint() -> None:
    """Test the root endpoint returns service status message."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"].startswith("NEO Hybrid AI Service")


def test_learn_endpoint_valid() -> None:
    """Test /learn endpoint with valid payload returns learning triggered."""
    resp = client.post("/learn", json={"features": [1, 2], "target": 1})
    assert resp.status_code == 200
    assert resp.json()["status"] == "learning triggered"


def test_learn_endpoint_invalid():
    """Test /learn endpoint with invalid payload returns 422."""
    # Missing features
    resp = client.post("/learn", json={"target": 1})
    assert resp.status_code == 422
    # Features not a list
    resp = client.post("/learn", json={"features": 123, "target": 1})
    assert resp.status_code == 422


def test_metrics_endpoint():
    """Test /metrics endpoint returns request count."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "request_count" in resp.json()


def test_explain_endpoint():
    """Test /explain endpoint returns explanation and feature importance."""
    resp = client.get("/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert "feature_importance" in data
    assert "explanation" in data
