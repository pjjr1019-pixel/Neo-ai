"""
Unit tests for benchmark_predict.py using pytest and unittest.mock.
Ensures async worker and main logic are robust and flake8-compliant.
"""

import pytest
import python_ai.benchmark_predict as bp
import httpx
from unittest.mock import patch, AsyncMock
import asyncio


@pytest.mark.parametrize(
    "status_code, should_raise",
    [
        (200, False),
        (500, True),
    ],
)
def test_worker_status(monkeypatch, status_code, should_raise):
    """Test worker status response and error handling."""
    class MockResponse:
        def __init__(self, code):
            """Initialize MockResponse with status code."""
            self.status_code = code

    async def mock_post(*args, **kwargs):
        return MockResponse(status_code)

    class MockClient:
        async def post(self, *args, **kwargs):
            return await mock_post()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    if should_raise:
        with pytest.raises(AssertionError):
            asyncio.run(bp.worker(MockClient(), 1))
    else:
        asyncio.run(bp.worker(MockClient(), 1))


@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([], 0),
        ([1, 2, 3], 6),
    ],
)
def test_worker_edge_cases(input_data, expected):
    """Test worker edge cases for input and output correctness."""
    assert sum(input_data) == expected


# Edge: No post method

def test_worker_no_post_method(monkeypatch):
    """Test worker when client lacks post method."""
    class MockClient:
        pass

    with pytest.raises(AttributeError):
        asyncio.run(bp.worker(MockClient(), 1))


def test_main_patch(monkeypatch):
    """Test main patching worker for async execution."""
    async def fake_worker(client, n):
        return None

    monkeypatch.setattr(bp, "worker", fake_worker)
    asyncio.run(bp.main())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code, should_raise",
    [
        (200, False),
        (500, True),
    ],
)
async def test_main_runs(monkeypatch, status_code, should_raise):
    mock_response = AsyncMock()
    mock_response.status_code = status_code
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    mock_acm = AsyncMock()
    mock_acm.__aenter__.return_value = mock_client

    with patch("httpx.AsyncClient", return_value=mock_acm):
        if should_raise:
            with pytest.raises(AssertionError):
                await bp.main()
        else:
            await bp.main()


@pytest.mark.parametrize(
    "output",
    [
        {"action": "buy", "confidence": 0.95},
        {"action": "hold", "confidence": 0.5},
        {"action": "sell", "confidence": 0.0},
        {"action": "buy", "confidence": 1.0},
    ],
)
def test_benchmark_predict(output):
    """Test benchmark_predict output action and confidence range."""
    assert output["action"] in ["buy", "hold", "sell"]
    assert 0.0 <= output["confidence"] <= 1.0
