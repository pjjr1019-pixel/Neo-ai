from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app, learning_logic

client = TestClient(app)


def test_predict_buy():
	"""Test /predict endpoint for 'buy' action."""
	response = client.post("/predict", json={"price": 150, "volume": 10})
	data = response.json()
	assert response.status_code == 200
	assert data["action"] == "buy"
	assert data["confidence"] == 0.95
	assert data["risk"] == 0.1


def test_predict_hold():
	"""Test /predict endpoint for 'hold' action."""
	response = client.post("/predict", json={"price": 50, "volume": 10})
	data = response.json()
	assert response.status_code == 200
	assert data["action"] == "hold"
	assert data["confidence"] == 0.95
	assert data["risk"] == 0.1


def test_predict_exception(monkeypatch):
	"""Test /predict endpoint exception handling."""
	def raise_exception(*args, **kwargs):
		raise Exception("Test error")
	monkeypatch.setattr("fastapi_service.logging.info", raise_exception)
	response = client.post("/predict", json={"price": 150, "volume": 10})
	data = response.json()
	assert response.status_code == 200
	assert data["action"] == "error"
	assert data["confidence"] == 0.0
	assert data["risk"] is None


def test_learn_success():
	"""Test /learn endpoint for successful learning."""
	payload = {"features": [1, 2, 3], "target": 1.0}
	response = client.post("/learn", json=payload)
	data = response.json()
	assert response.status_code == 200
	assert data["status"] == "learning triggered"
	assert data["received"] == payload


def test_learn_exception():
	"""Test /learn endpoint exception handling using dependency override."""
	def error_logic(data):
		raise Exception("Test error")
	app.dependency_overrides[learning_logic] = error_logic
	response = client.post("/learn", json={"features": [1, 2, 3], "target": 1.0})
	assert response.status_code in (422, 500)
	app.dependency_overrides.pop(learning_logic, None)
