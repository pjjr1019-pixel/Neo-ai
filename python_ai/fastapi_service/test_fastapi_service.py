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
		({"features": [], "target": 0}, 200, "status"),
		({"features": [1, 2, 3]}, 422, None),
		({}, 422, None),
		({"features": [1, 2, 3], "target": "badtype"}, 422, None),
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
	"""Test /metrics endpoint returns all expected keys."""
	client = TestClient(app)
	response = client.get("/metrics")
	assert response.status_code == 200
	data = response.json()
	assert "avg_latency" in data
	assert "throughput" in data
	assert "memory_mb" in data
	assert "cpu_percent" in data
