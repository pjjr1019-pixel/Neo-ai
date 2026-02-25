"""
NEO Hybrid AI Full Integration Test

Tests end-to-end connectivity and workflow between Java client, Python AI,
PostgreSQL, and Redis. Logs results and updates documentation.
"""
import subprocess
import psycopg2
import redis
from fastapi.testclient import TestClient
from python_ai.fastapi_service.fastapi_service import app
from pathlib import Path
import pytest
import os

RESULTS = []
    # ...existing code...
def log_result(name, success, details=""):
    """Log the result of a test step.
    Args:
        name (str): Name of the test step.
        success (bool): Whether the test passed.
        details (str): Additional details.
    """
    result = (
        f"{name}: {'PASS' if success else 'FAIL'} "
        f"{details[:75]}"
        f"{'...' if len(details) > 75 else ''}"
    )
    print(result)
    RESULTS.append(result)
    # ...existing code...
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get('RUN_FULL_INTEGRATION') != '1',
    reason=(
        "Full integration test skipped: "
        "RUN_FULL_INTEGRATION env var not set."
    )
)
def test_full_integration():
    """Run the full integration workflow and assert all steps pass."""
    # 1. Test PostgreSQL connection
    test_postgres_connection()
    # ...existing code...
    # 2. Test Redis connection
    test_redis_connection()
    # ...existing code...
    # 3. Test FastAPI /predict endpoint
    test_fastapi_predict()


    # 4. Test FastAPI /learn endpoint
    test_fastapi_learn()


    # 5. Test Java client (simulate call)
    try:
        result = subprocess.run(
            [
                "java",
                "-cp",
                "java_core",
                "data_ingestion.RealTimeDataFetcher"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        log_result(
            "Java client execution",
            result.returncode == 0,
            (result.stdout + result.stderr)[:75]
            + (
                '...'
                if len(result.stdout + result.stderr) > 75
                else ''
            )
        )
        assert result.returncode == 0, (
            f"Java client failed: {result.stderr}"
        )
    except Exception as e:
        log_result("Java client execution", False, str(e))
        assert False, f"Java client execution failed: {e}"

    # 6. Log results to docs
    results_path = (
        Path(__file__).parent.parent /
        'docs' /
        'phase-5.5-integration-test-results.md'
    )
    with open(results_path, "w") as f:
        f.write("# NEO Hybrid AI - Phase 5.5 Integration Test Results\n\n")
        for line in RESULTS:
            f.write(line + "\n")

    print(
        "Integration test complete. Results logged in "
        "docs/phase-5.5-integration-test-results.md."
    )


def test_postgres_connection():
    """Test PostgreSQL connection with valid credentials."""
    conn = psycopg2.connect(
        dbname='neoai_db', user='neoai', password='neoai123', host='localhost',
        port=5432
    )
    conn.close()


def test_postgres_connection_fail():
    """Test PostgreSQL connection with invalid credentials (should fail)."""
    with pytest.raises(Exception):
        psycopg2.connect(
            dbname='neoai_db', user='wrong', password='wrong', host='localhost',
            port=5432
        )


def test_redis_connection():
    """Test Redis connection."""
    r = redis.Redis(host='localhost', port=6379)
    assert r.ping() is True
    r.close()


def test_redis_connection_fail():
    """Test Redis connection with wrong port (should fail)."""
    with pytest.raises(Exception):
        r = redis.Redis(host='localhost', port=9999)
        r.ping()


def test_fastapi_predict():
    """Test FastAPI /predict endpoint with valid input."""
    client = TestClient(app)
    response = client.post("/predict", json={"price": 123.45, "volume": 1000})
    assert response.status_code == 200
    data = response.json()
    assert "action" in data and "confidence" in data


def test_fastapi_predict_invalid():
    """Test FastAPI /predict endpoint with invalid input (should fail)."""
    client = TestClient(app)
    response = client.post("/predict", json={"foo": "bar"})
    assert response.status_code != 200


def test_fastapi_learn():
    """Test FastAPI /learn endpoint with valid input."""
    client = TestClient(app)
    response = client.post("/learn", json={"features": [1, 2, 3], "target": 1})
    assert response.status_code == 200


def test_fastapi_learn_invalid():
    """Test FastAPI /learn endpoint with invalid input (should fail)."""
    client = TestClient(app)
    response = client.post("/learn", json={"foo": "bar"})
    assert response.status_code != 200
