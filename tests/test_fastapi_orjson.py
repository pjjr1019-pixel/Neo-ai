"""Tests for FastAPI serialization configuration."""

from fastapi.testclient import TestClient

from python_ai.auth.dependencies import get_current_user
from python_ai.auth.models import User, UserRole
from python_ai.fastapi_service.fastapi_service import app


def _admin_user() -> User:
    return User(
        username="orjson_tester",
        email="orjson@neo.ai",
        full_name="ORJSON Tester",
        roles=[UserRole.ADMIN],
        disabled=False,
    )


def test_predict_response_json_serialization() -> None:
    app.dependency_overrides[get_current_user] = _admin_user
    client = TestClient(app)
    try:
        response = client.post(
            "/predict",
            json={"features": {"sma_14": 1.0, "rsi_14": 50.0}},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        payload = response.json()
        assert "prediction" in payload
        assert "confidence" in payload
        assert "signal" in payload
    finally:
        app.dependency_overrides.pop(get_current_user, None)
