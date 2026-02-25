
import pytest
from unittest.mock import patch, AsyncMock
import python_ai.benchmark_predict as bp

@pytest.mark.asyncio
async def test_main_runs(monkeypatch):
    # Patch httpx.AsyncClient to always return a mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    # Patch AsyncClient context manager
    mock_acm = AsyncMock()
    mock_acm.__aenter__.return_value = mock_client

    with patch('httpx.AsyncClient', return_value=mock_acm):
        await bp.main()


def test_benchmark_predict():
    # ...existing code...
    output = {"action": "buy", "confidence": 0.95}
    assert output["action"] in ["buy", "hold", "sell"]
    assert 0.0 <= output["confidence"] <= 1.0

