import asyncio
import time
import python_ai.benchmark_predict as bp
import pytest


class DummyResponse:
    """Dummy response object for simulating HTTP responses."""

    def __init__(self, status_code=200):
        """Initialize DummyResponse with a status code."""
        self.status_code = status_code


class DummyClient:
    """Dummy client for simulating async post requests."""

    def __init__(self):
        """Initialize DummyClient with an empty post_calls list."""
        self.post_calls = []

    async def post(self, url, json):
        """Simulate an async POST request and record the call."""
        self.post_calls.append((url, json))
        return DummyResponse()


@pytest.mark.asyncio
async def test_worker_success():
    client = DummyClient()
    client.post_calls = []  # Ensure isolation
    import python_ai.benchmark_predict as bp

    await bp.worker(client, 3)
    assert len(client.post_calls) == 3
    for url, payload in client.post_calls:
        assert url == bp.URL
        assert payload == bp.PAYLOAD


@pytest.mark.asyncio
async def test_main_runs(monkeypatch):
    class DummyAsyncClient:
        """Dummy async context manager for simulating httpx.AsyncClient."""

        def __init__(self, *a, **kw):
            """Initialize DummyAsyncClient (dummy for context manager)."""
            pass

        async def __aenter__(self):
            return DummyClient()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(bp.httpx, "AsyncClient", DummyAsyncClient)
    # Patch asyncio.gather to a dummy implementation
    # that just runs the coroutines sequentially

    async def dummy_gather(*tasks, **kwargs):
        results = []
        for t in tasks:
            results.append(await t)
        return results

    monkeypatch.setattr(asyncio, "gather", dummy_gather)
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    output = []
    import builtins

    monkeypatch.setattr(builtins, "print", output.append)
    await bp.main()
    assert any("Throughput" in str(line) for line in output)


def test_main_block_excluded(monkeypatch):
    """Test that importing the module does not run main()."""
    import importlib
    import sys

    name = "python_ai.benchmark_predict"
    if name in sys.modules:
        del sys.modules[name]
    importlib.import_module(name)
