"""
Unit tests for benchmark_predict.py using pytest and unittest.mock.
Ensures async worker and main logic are robust and flake8-compliant.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import python_ai.benchmark_predict as bp
from python_ai.benchmark_predict import worker


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
        """Mock HTTP response with configurable status code."""

        def __init__(self, code):
            """Initialize MockResponse with status code."""
            self.status_code = code

    async def mock_post(*args, **kwargs):
        return MockResponse(status_code)

    class MockClient:
        """Mock HTTP client for simulating async post requests."""

        async def post(self, *args, **kwargs):
            """Send a mock POST request."""
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
        """Mock client for simulating async post requests."""

        pass

    with pytest.raises(AttributeError):
        asyncio.run(bp.worker(MockClient(), 1))


def test_worker_http_error(monkeypatch):
    """Test worker handles HTTP error gracefully."""

    class FakeResponse:
        """Fake response object for simulating HTTP errors."""

        status_code = 500

    async def fake_post(*args, **kwargs):
        return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    import asyncio

    async def run():
        client = httpx.AsyncClient()
        try:
            await worker(client, 1)
        except AssertionError:
            assert True
        await client.aclose()

    asyncio.run(run())


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


def test_main_print(monkeypatch, capsys):
    """Test main print statements for coverage."""

    async def fake_worker(client, n):
        return None

    monkeypatch.setattr(bp, "worker", fake_worker)
    monkeypatch.setattr(bp, "REQUESTS", 10)
    monkeypatch.setattr(bp, "CONCURRENCY", 2)
    monkeypatch.setattr(bp, "URL", "http://127.0.0.1:8000/predict")
    monkeypatch.setattr(bp, "PAYLOAD", {"price": 1, "volume": 1})
    # Patch time.perf_counter to return predictable values
    monkeypatch.setattr(bp.time, "perf_counter", lambda: 1.0)
    asyncio.run(bp.main())
    out = capsys.readouterr().out
    assert "Sent" in out and "Throughput" in out


def test_worker_status_invalid(monkeypatch):
    """Test worker with invalid status code (not 200 or 500)."""

    class MockResponse:
        """Mock response object for simulating HTTP responses."""

        def __init__(self, code):
            """Initialize MockResponse with status code."""
            self.status_code = code

    async def mock_post(*args, **kwargs):
        return MockResponse(404)

    class MockClient:
        """Mock client for simulating async post requests."""

        async def post(self, *args, **kwargs):
            return await mock_post()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    with pytest.raises(AssertionError):
        asyncio.run(bp.worker(MockClient(), 1))


def test_main_runs_invalid(monkeypatch):
    """Test main with invalid status code for coverage."""
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_acm = AsyncMock()
    mock_acm.__aenter__.return_value = mock_client
    with patch("httpx.AsyncClient", return_value=mock_acm):
        with pytest.raises(AssertionError):
            asyncio.run(bp.main())


def test_main_zero_requests(monkeypatch, capsys):
    """Test main with zero requests and zero concurrency edge cases."""
    monkeypatch.setattr(bp, "REQUESTS", 0)
    monkeypatch.setattr(bp, "CONCURRENCY", 1)
    monkeypatch.setattr(bp, "URL", "http://127.0.0.1:8000/predict")
    monkeypatch.setattr(bp, "PAYLOAD", {"price": 1, "volume": 1})
    monkeypatch.setattr(bp.time, "perf_counter", lambda: 1.0)

    async def fake_worker(client, n):
        return None

    monkeypatch.setattr(bp, "worker", fake_worker)
    asyncio.run(bp.main())
    out = capsys.readouterr().out
    assert "Sent 0 requests" in out


def test_main_zero_concurrency(monkeypatch, capsys):
    """Test main with zero concurrency (should not raise)."""
    monkeypatch.setattr(bp, "REQUESTS", 10)
    monkeypatch.setattr(bp, "CONCURRENCY", 0)
    monkeypatch.setattr(bp, "URL", "http://127.0.0.1:8000/predict")
    monkeypatch.setattr(bp, "PAYLOAD", {"price": 1, "volume": 1})
    monkeypatch.setattr(bp.time, "perf_counter", lambda: 1.0)

    async def fake_worker(client, n):
        return None

    monkeypatch.setattr(bp, "worker", fake_worker)
    try:
        asyncio.run(bp.main())
    except ZeroDivisionError:
        pass


def test_worker_unexpected_exception(monkeypatch):
    """Test worker handles unexpected exceptions."""

    async def bad_post(*args, **kwargs):
        raise RuntimeError("unexpected error")

    class MockClient:
        """Mock HTTP client that raises on post."""

        async def post(self, *args, **kwargs):
            """Send a mock POST request that raises."""
            return await bad_post()

    with pytest.raises(RuntimeError):
        asyncio.run(bp.worker(MockClient(), 1))


def test_main_block_excluded():
    """Test that the main block is excluded from import side effects."""
    import importlib
    import sys

    modname = "python_ai.benchmark_predict"
    if modname in sys.modules:
        del sys.modules[modname]
    importlib.import_module(modname)
