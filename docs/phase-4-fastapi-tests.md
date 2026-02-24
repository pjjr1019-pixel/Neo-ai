# NEO Hybrid AI — FastAPI Unit & Integration Tests

## Overview
- Automated tests for /predict and /learn endpoints
- Validate input/output schemas and edge cases

## Example (Python, pytest)
from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app

client = TestClient(app)

def test_predict_valid():
    response = client.post("/predict", json={"price": 120.0, "volume": 210.0})
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert "confidence" in data
    assert "risk" in data

def test_predict_invalid():
    response = client.post("/predict", json={"price": "bad", "volume": 210.0})
    assert response.status_code == 422  # Unprocessable Entity

def test_learn():
    response = client.post("/learn", json={"price": 120.0, "volume": 210.0, "label": "buy"})
    assert response.status_code == 200
    assert "status" in response.json()

---
## Logging
- Log test results and failures
- Update this file as tests evolve