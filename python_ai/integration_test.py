



import pytest
from fastapi import status
from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app

def test_predict_and_learn_integration() -> None:
	"""
	Integration test for /predict and /learn endpoints using FastAPI's TestClient.
	Verifies end-to-end behavior and response structure.
	"""
	client = TestClient(app)
	# Test /predict endpoint
	predict_payload = {"price": 123.45, "volume": 1000}
	predict_resp = client.post("/predict", json=predict_payload)
	assert predict_resp.status_code == status.HTTP_200_OK
	predict_data = predict_resp.json()
	assert "action" in predict_data
	assert "confidence" in predict_data
	assert "risk" in predict_data

	# Test /learn endpoint
	learn_payload = {"features": [1, 2, 3], "target": 1.0}
	learn_resp = client.post("/learn", json=learn_payload)
	assert learn_resp.status_code == status.HTTP_200_OK
	learn_data = learn_resp.json()
	assert "status" in learn_data
	assert "received" in learn_data
