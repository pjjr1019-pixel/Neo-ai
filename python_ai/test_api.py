from python_ai.fastapi_service.fastapi_service import app, get_learning_logic, learning_logic
from fastapi.testclient import TestClient

def test_learn():
    """Test /learn endpoint."""
    app.dependency_overrides[get_learning_logic] = lambda: learning_logic
    client = TestClient(app)
    try:
        response = client.post("/learn", json={"features": [1, 2, 3], "target": 1})
        print("/learn response:", response.status_code, response.json())
        assert response.status_code == 200
        assert "status" in response.json()
    finally:
        app.dependency_overrides.pop(get_learning_logic, None)