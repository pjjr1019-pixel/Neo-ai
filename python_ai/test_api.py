
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app

client = TestClient(app)

def test_predict_valid():
    """Test valid /predict endpoint."""
    # ...existing code...
    response = client.post("/predict", json={"price": 120.0, "volume": 210.0})
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert "confidence" in data
    assert "risk" in data

    

def test_predict_invalid():
    """Test invalid /predict endpoint."""
    # ...existing code...
    response = client.post("/predict", json={"price": "bad", "volume": 210.0})
    assert response.status_code == 422  # Unprocessable Entity

    

def test_learn():
    """Test /learn endpoint."""
    # ...existing code...
    response = client.post("/learn", json={"features": [1, 2, 3], "target": 1})
    assert response.status_code == 200
    assert "status" in response.json()
