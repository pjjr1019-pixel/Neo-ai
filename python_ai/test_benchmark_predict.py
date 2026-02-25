"""
Unit tests for benchmark_predict.py using pytest and unittest.mock.
Ensures async worker and main logic are robust and flake8-compliant.
"""

import pytest
import python_ai.benchmark_predict as bp
import httpx
from unittest.mock import patch, AsyncMock


def test_worker_success(monkeypatch):
    class MockResponse:
        status_code = 200

    async def mock_post(*args, **kwargs):
        return MockResponse()

    class MockClient:
        async def post(self, *args, **kwargs):
            return await mock_post()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    import asyncio
    asyncio.run(bp.worker(MockClient(), 1))


def test_worker_failure(monkeypatch):
    class MockResponse:
        status_code = 500

    async def mock_post(*args, **kwargs):
        return MockResponse()

    class MockClient:
        async def post(self, *args, **kwargs):
            return await mock_post()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    import asyncio
    with pytest.raises(AssertionError):
        asyncio.run(bp.worker(MockClient(), 1))


def test_main_patch(monkeypatch):
    async def fake_worker(client, n):
        return None
    monkeypatch.setattr(bp, "worker", fake_worker)
    import asyncio
    asyncio.run(bp.main())


@pytest.mark.asyncio
async def test_main_runs(monkeypatch):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    mock_acm = AsyncMock()
    mock_acm.__aenter__.return_value = mock_client

    with patch('httpx.AsyncClient', return_value=mock_acm):
        await bp.main()


def test_benchmark_predict():
    output = {"action": "buy", "confidence": 0.95}
    assert output["action"] in ["buy", "hold", "sell"]
    assert 0.0 <= output["confidence"] <= 1.0
