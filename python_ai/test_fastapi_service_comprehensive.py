from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app, get_learning_logic


def test_root():
    """Test the root endpoint returns service status."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"].startswith("NEO Hybrid AI Service")


def test_predict():
    """Test the predict endpoint returns a prediction."""
    client = TestClient(app)
    resp = client.post("/predict", json={"input": "test"})
    assert resp.status_code == 200
    assert "Predicted value" in resp.json()["output"]


def test_learn_success():
    """Test the learn endpoint with valid input triggers learning."""
    client = TestClient(app)
    payload = {"features": [1, 2, 3], "target": 1}
    resp = client.post("/learn", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "learning triggered"


def test_learn_error_cases():
    """Test the learn endpoint with invalid input returns error status."""
    logic = get_learning_logic()
    assert logic(None)["status"] == "error"
    assert logic({"features": "notalist", "target": 1})["status"] == "error"
    assert logic({"features": [1, 2, 3]})["status"] == "error"


def test_metrics():
    """Test the metrics endpoint returns request count."""
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "request_count" in resp.json()


def test_explain():
    """Test the explain endpoint returns feature importance and explanation."""
    client = TestClient(app)
    resp = client.get("/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert "feature_importance" in data
    assert "explanation" in data
