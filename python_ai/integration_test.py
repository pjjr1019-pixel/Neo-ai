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


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('RUN_FULL_INTEGRATION') != '1', reason="Full integration test skipped: RUN_FULL_INTEGRATION env var not set.")
def test_full_integration():
    """Run the full integration workflow and assert all steps pass."""
    # 1. Test PostgreSQL connection
    try:
        conn = psycopg2.connect(
            dbname='neoai_db', user='neoai', password='neoai123',
            host='localhost', port=5432)
        log_result("PostgreSQL connection", True)
        conn.close()
    except Exception as e:
        log_result("PostgreSQL connection", False, str(e))
        assert False, f"PostgreSQL connection failed: {e}"

    # 2. Test Redis connection
    try:
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        log_result("Redis connection", True)
        r.close()
    except Exception as e:
        log_result("Redis connection", False, str(e))
        assert False, f"Redis connection failed: {e}"

    # 3. Test FastAPI /predict endpoint
    try:
        client = TestClient(app)
        response = client.post(
            "/predict", json={"price": 123.45, "volume": 1000}
        )
        log_result(
            "FastAPI /predict endpoint",
            response.status_code == 200,
            str(response.json())[:75] + (
                '...' if len(str(response.json())) > 75 else ''
            )
        )
        assert response.status_code == 200, (
            f"/predict failed: {response.text}"
        )
    except Exception as e:
        log_result("FastAPI /predict endpoint", False, str(e))
        assert False, f"FastAPI /predict endpoint failed: {e}"

    # 4. Test FastAPI /learn endpoint
    try:
        response = client.post(
            "/learn", json={"features": [1, 2, 3], "target": 1}
        )
        log_result(
            "FastAPI /learn endpoint",
            response.status_code == 200,
            str(response.json())[:75] + (
                '...' if len(str(response.json())) > 75 else ''
            )
        )
        assert response.status_code == 200, f"/learn failed: {response.text}"
    except Exception as e:
        log_result("FastAPI /learn endpoint", False, str(e))
        assert False, f"FastAPI /learn endpoint failed: {e}"

    # 5. Test Java client (simulate call)
    try:
        result = subprocess.run(
            ["java", "-cp", "java_core", "data_ingestion.RealTimeDataFetcher"],
            capture_output=True,
            text=True,
            timeout=10
        )
        log_result(
            "Java client execution",
            result.returncode == 0,
            (result.stdout + result.stderr)[:75]
            + (
                '...' if len(result.stdout + result.stderr) > 75 else ''
            )
        )
        assert result.returncode == 0, f"Java client failed: {result.stderr}"
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
