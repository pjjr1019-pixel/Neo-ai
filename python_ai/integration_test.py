from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app

client = TestClient(app)


def test_predict_and_learn_integration_v2() -> None:
    """
    Integration test for /predict and /learn endpoints using FastAPI's
    TestClient. Verifies end-to-end behavior and response structure.
    """
    # Test /predict endpoint
    predict_payload = {"input": "test input"}
    predict_resp = client.post("/predict", json=predict_payload)
    assert predict_resp.status_code == 200
    assert "output" in predict_resp.json()

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
