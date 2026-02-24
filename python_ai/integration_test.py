
"""
NEO Hybrid AI Full Integration Test

Tests end-to-end connectivity and workflow between Java client, Python AI, PostgreSQL, and Redis.
Logs results and updates documentation.
"""

import subprocess
import psycopg2
import redis
from fastapi.testclient import TestClient
from fastapi_service.fastapi_service import app
from pathlib import Path

RESULTS = []


def log_result(name, success, details=""):
    """Log the result of a test step.
    Args:
        name (str): Name of the test step.
        success (bool): Whether the test passed.
        details (str): Additional details.
    """
    print(result)
    RESULTS.append(result)
    result = (
        f"{name}: {'PASS' if success else 'FAIL'} "
        f"{details[:75]}{'...' if len(details) > 75 else ''}"
    )
    print(result)
    RESULTS.append(result)


# 1. Test PostgreSQL connection
try:
    conn = psycopg2.connect(
        dbname='neoai_db', user='neoai', password='neoai123',
        host='localhost', port=5432)
    log_result("PostgreSQL connection", True)
    conn.close()
except Exception as e:
    log_result("PostgreSQL connection", False, str(e))

# 2. Test Redis connection
try:
    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    log_result("Redis connection", True)
    r.close()
except Exception as e:
    log_result("Redis connection", False, str(e))

# 3. Test FastAPI endpoints
    response = client.post("/predict", json={"price": 123.45, "volume": 1000})
    response = client.post("/predict", json={"price": 123.45, "volume": 1000})
    log_result(
        "FastAPI /predict endpoint",
        response.status_code == 200,
        str(response.json())
    )
    log_result(
        "FastAPI /predict endpoint",
        response.status_code == 200,
        str(response.json())
    )
    response = client.post("/predict", json={"price": 123.45, "volume": 1000})
    log_result(
        "FastAPI /predict endpoint",
        response.status_code == 200,
        str(response.json())
try:
    response = client.post("/predict", json={"price": 123.45, "volume": 1000})
    )
        str(response.json())
    )
try:
    response = client.post("/predict", json={"price": 123.45, "volume": 1000})
        response.status_code == 200,
        str(response.json())
    )
try:
    response = client.post("/predict", json={"price": 123.45, "volume": 1000})
    log_result(
        "FastAPI /predict endpoint",
        response.status_code == 200,
        str(response.json())
    )
    )
client = TestClient(app)
    log_result(
        "FastAPI /predict endpoint",
        response.status_code == 200,
        str(response.json())
    )
client = TestClient(app)
        "FastAPI /predict endpoint",
        response.status_code == 200,
        str(response.json())
    )
client = TestClient(app)
        str(response.json())
    )
client = TestClient(app)
try:
    response = client.post("/predict", json={"price": 123.45, "volume": 1000})
    log_result(
        "FastAPI /predict endpoint",
        response.status_code == 200,
        str(response.json())
    )
    log_result(
        "FastAPI /predict endpoint",
        response.status_code == 200,
        str(response.json())[:75] + ('...' if len(str(response.json())) > 75 else '')
    )
except Exception as e:
    log_result("FastAPI /predict endpoint", False, str(e))

try:
    response = client.post("/learn", json={"features": [1, 2, 3], "target": 1})
        "FastAPI /learn endpoint",
        response.status_code == 200,
        str(response.json())
    )
    log_result(
        "FastAPI /learn endpoint",
        response.status_code == 200,
        str(response.json())[:75] + ('...' if len(str(response.json())) > 75 else '')
    )
except Exception as e:
    log_result("FastAPI /learn endpoint", False, str(e))

# 4. Test Java client (simulate call)
try:
    result = subprocess.run([
        "java", "-cp", "java_core", "data_ingestion.RealTimeDataFetcher"
    ], capture_output=True, text=True, timeout=10)
        "Java client execution",
        result.returncode == 0,
        (result.stdout + result.stderr)
    )
    log_result(
        "Java client execution",
        result.returncode == 0,
        (result.stdout + result.stderr)[:75] + ('...' if len(result.stdout + result.stderr) > 75 else '')
    )
except Exception as e:
    log_result("Java client execution", False, str(e))

# 5. Log results to docs

results_path = (
    Path(__file__).parent.parent / 'docs' / 'phase-5.5-integration-test-results.md'
)
with open(results_path, "w") as f:
    f.write("# NEO Hybrid AI - Phase 5.5 Integration Test Results\n\n")
    for line in RESULTS:
        f.write(line + "\n")

print(
    "Integration test complete. Results logged in docs/phase-5.5-integration-test-results.md."
)
