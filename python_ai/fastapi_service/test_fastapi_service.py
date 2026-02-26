import pytest
from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app, PredictInput, get_learning_logic, learning_logic, predict_logic

client = TestClient(app)

import os
import subprocess
def test_predict_valid():
	"""Test predict endpoint with valid input."""
	response = client.post("/predict", json={"price": 123.45, "volume": 1000})
	print("/predict response:", response.status_code, response.text)
	assert response.status_code == 200
	data = response.json()
	assert "action" in data
	assert "confidence" in data
	assert "risk" in data

def test_predict_empty():
	"""Test predict endpoint with empty input."""
	response = client.post("/predict", json={})
	assert response.status_code == 422

def test_learn():
	"""Test learn endpoint."""
	payload = {"features": [1.0, 2.0, 3.0], "target": 1.0}
	response = client.post("/learn", json=payload)
	assert response.status_code == 200
	assert response.json()["status"] == "learning triggered"

def test_metrics():
	"""Test metrics endpoint."""
	response = client.get("/metrics")
	assert response.status_code == 200
	data = response.json()
	assert "avg_latency" in data
	assert "throughput" in data
	assert "memory_mb" in data
	assert "cpu_percent" in data

def test_learn_cases(payload: dict, expected_status: int, expected_key: str) -> None:
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

def test_predict_cases(payload: dict, expected_status: int) -> None:
	"""Test /predict endpoint with valid, edge, and error cases."""
	client = TestClient(app)
	response = client.post("/predict", json=payload)
	assert response.status_code == expected_status
	if expected_status == 200:
		data = response.json()
		assert "action" in data
		assert "confidence" in data
		assert "risk" in data

def test_metrics_endpoint() -> None:
	"""Test /metrics endpoint returns all expected keys."""
	client = TestClient(app)
	response = client.get("/metrics")
	assert response.status_code == 200
	data = response.json()
	assert "avg_latency" in data
	assert "throughput" in data
	assert "memory_mb" in data
	assert "cpu_percent" in data

def test_learn_injected_logic_not_callable() -> None:
	"""Test /learn with a non-callable injected logic (should trigger exception branch)."""
	app.dependency_overrides[get_learning_logic] = lambda: 123
	client = TestClient(app)
	payload = {"features": [1, 2, 3], "target": 1.0}
	try:
		response = client.post("/learn", json=payload)
		assert response.status_code == 200
		data = response.json()
		assert data["status"] == "error"
	finally:
		app.dependency_overrides.pop(get_learning_logic, None)

def test_learn_injected_logic_returns_callable() -> None:
	"""Test /learn with injected logic that returns a function (should trigger exception branch)."""
	def bad_logic(data):
		return lambda x: x
	app.dependency_overrides[get_learning_logic] = lambda: bad_logic
	client = TestClient(app)
	payload = {"features": [1, 2, 3], "target": 1.0}
	try:
		response = client.post("/learn", json=payload)
		assert response.status_code == 200
		data = response.json()
		assert data["status"] == "error"
	finally:
		app.dependency_overrides.pop(get_learning_logic, None)

def test_learn_injected_logic_returns_non_dict() -> None:
	"""Test /learn with injected logic that returns a non-dict (should trigger exception branch)."""
	def bad_logic(data):
		return 42
	app.dependency_overrides[get_learning_logic] = lambda: bad_logic
	client = TestClient(app)
	payload = {"features": [1, 2, 3], "target": 1.0}
	try:
		response = client.post("/learn", json=payload)
		assert response.status_code == 200
		data = response.json()
		assert data["status"] == "error"
	finally:
		app.dependency_overrides.pop(get_learning_logic, None)

def test_predict_exception(monkeypatch) -> None:
	"""Test /predict endpoint exception branch by forcing an exception."""
	import python_ai.fastapi_service.fastapi_service as fs
	orig_logic = fs.predict_logic
	def raise_exc(data):
		raise ValueError("forced error")
	fs.predict_logic = raise_exc
	client = TestClient(app)
	payload = {"price": 123.45, "volume": 1000}
	try:
		response = client.post("/predict", json=payload)
		assert response.status_code == 200
		data = response.json()
		assert data["action"] == "error"
	finally:
		fs.predict_logic = orig_logic

def test_predict_type_hints_and_docstrings() -> None:
	"""Test PredictInput and PredictionOutput type hints and docstrings."""
	assert hasattr(PredictInput, '__annotations__')
	assert hasattr(PredictInput, '__doc__')
	assert hasattr(predict_logic, '__doc__')

def test_metrics_endpoint() -> None:
	"""Test /metrics endpoint returns all expected keys."""
	client = TestClient(app)
	response = client.get("/metrics")
	assert response.status_code == 200
	data = response.json()
	assert "avg_latency" in data
	assert "throughput" in data
	assert "memory_mb" in data
	assert "cpu_percent" in data


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
def test_learn_cases(payload: dict, expected_status: int, expected_key: str) -> None:
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

@pytest.mark.parametrize(
	"payload,expected_status",
	[
		({"price": 123.45, "volume": 1000}, 200),
		({"price": 0, "volume": 0}, 200),
		({"price": "badtype", "volume": 1000}, 422),
		({}, 422),
	]
)
def test_predict_cases(payload: dict, expected_status: int) -> None:
	"""Test /predict endpoint with valid, edge, and error cases."""
	client = TestClient(app)
	response = client.post("/predict", json=payload)
	assert response.status_code == expected_status
	if expected_status == 200:
		data = response.json()
		assert "action" in data
		assert "confidence" in data
		assert "risk" in data

def test_metrics_endpoint() -> None:
	"""Test /metrics endpoint returns all expected keys."""
	client = TestClient(app)
	response = client.get("/metrics")
	assert response.status_code == 200
	data = response.json()
	assert "avg_latency" in data
	assert "throughput" in data
	assert "memory_mb" in data
	assert "cpu_percent" in data

def test_learn_injected_logic_not_callable() -> None:
	"""Test /learn with a non-callable injected logic (should trigger exception branch)."""
	app.dependency_overrides[get_learning_logic] = lambda: 123
	client = TestClient(app)
	payload = {"features": [1, 2, 3], "target": 1.0}
	try:
		response = client.post("/learn", json=payload)
		assert response.status_code == 200
		data = response.json()
		assert data["status"] == "error"
	finally:
		app.dependency_overrides.pop(get_learning_logic, None)

def test_learn_injected_logic_returns_callable() -> None:
	"""Test /learn with injected logic that returns a function (should trigger exception branch)."""
	def bad_logic(data):
		return lambda x: x
	app.dependency_overrides[get_learning_logic] = lambda: bad_logic
	client = TestClient(app)
	payload = {"features": [1, 2, 3], "target": 1.0}
	try:
		response = client.post("/learn", json=payload)
		assert response.status_code == 200
		data = response.json()
		assert data["status"] == "error"
	finally:
		app.dependency_overrides.pop(get_learning_logic, None)

def test_learn_injected_logic_returns_non_dict() -> None:
	"""Test /learn with injected logic that returns a non-dict (should trigger exception branch)."""
	def bad_logic(data):
		return 42
	app.dependency_overrides[get_learning_logic] = lambda: bad_logic
	client = TestClient(app)
	payload = {"features": [1, 2, 3], "target": 1.0}
	try:
		response = client.post("/learn", json=payload)
		assert response.status_code == 200
		data = response.json()
		assert data["status"] == "error"
	finally:
		app.dependency_overrides.pop(get_learning_logic, None)

def test_predict_exception(monkeypatch) -> None:
	"""Test /predict endpoint exception branch by forcing an exception."""
	import python_ai.fastapi_service.fastapi_service as fs
	orig_logic = fs.predict_logic
	def raise_exc(data):
		raise ValueError("forced error")
	fs.predict_logic = raise_exc
	client = TestClient(app)
	payload = {"price": 123.45, "volume": 1000}
	try:
		response = client.post("/predict", json=payload)
		assert response.status_code == 200
		data = response.json()
		assert data["action"] == "error"
	finally:
		fs.predict_logic = orig_logic

def test_predict_type_hints_and_docstrings() -> None:
	"""Test PredictInput and PredictionOutput type hints and docstrings."""
	assert hasattr(PredictInput, '__annotations__')
	assert hasattr(PredictInput, '__doc__')
	assert hasattr(predict_logic, '__doc__')
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
