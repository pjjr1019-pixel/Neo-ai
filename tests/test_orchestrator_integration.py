"""
Tests for Python-Java Orchestrator Integration.

Covers feature computation, autonomous cycle execution, backtesting
integration, and execution history tracking.
"""

import pytest

from python_ai.orchestrator_integration import (
    OrchestratorIntegration,
    get_orchestrator_integration,
)


class TestOrchestratorIntegration:
    """Test orchestrator integration layer."""

    def test_integration_initialization(self) -> None:
        """Test orchestrator integration initializes correctly."""
        integration = OrchestratorIntegration()
        assert integration.pipeline is not None
        assert integration.model is not None
        assert integration.backtest_engine is not None
        assert integration.execution_history == []

    def test_compute_features_from_prices(self) -> None:
        """Test feature computation from raw OHLCV data."""
        integration = OrchestratorIntegration()
        ohlcv_data = {
            "open": [100.0, 105.0, 110.0],
            "high": [101.0, 106.0, 111.0],
            "low": [99.0, 104.0, 109.0],
            "close": [100.5, 105.5, 110.5],
            "volume": [1000.0, 1100.0, 1200.0],
        }
        features = integration.compute_features_from_prices(
            "BTC/USD",
            ohlcv_data,
        )
        assert isinstance(features, dict)
        assert len(features) == 10
        assert all(
            key in features
            for key in [
                "f0",
                "f1",
                "f2",
                "f3",
                "f4",
                "f5",
                "f6",
                "f7",
                "f8",
                "f9",
            ]
        )

    def test_compute_features_invalid_data(self) -> None:
        """Test feature computation rejects invalid data."""
        integration = OrchestratorIntegration()
        with pytest.raises(ValueError):
            integration.compute_features_from_prices("", {})
        with pytest.raises(ValueError):
            integration.compute_features_from_prices(
                "BTC/USD",
                {"close": []},
            )

    def test_predict_with_model(self) -> None:
        """Test model prediction on computed features."""
        integration = OrchestratorIntegration()
        features = {
            "f0": 0.5,
            "f1": 0.6,
            "f2": 0.7,
            "f3": 0.8,
            "f4": 0.4,
            "f5": 0.5,
            "f6": 0.6,
            "f7": 0.7,
            "f8": 0.3,
            "f9": 0.4,
        }
        result = integration.predict_with_model(features)
        assert "prediction" in result
        assert "confidence" in result
        assert "signal" in result
        assert result["signal"] in ["BUY", "SELL"]
        assert 0.5 <= result["confidence"] <= 0.95

    def test_predict_invalid_features(self) -> None:
        """Test prediction rejects invalid features."""
        integration = OrchestratorIntegration()
        with pytest.raises(ValueError):
            integration.predict_with_model([1, 2, 3])  # type: ignore

    def test_execute_autonomous_cycle(self) -> None:
        """Test full autonomous trading cycle."""
        integration = OrchestratorIntegration()
        ohlcv_data = {
            "open": [100.0, 105.0, 110.0],
            "high": [101.0, 106.0, 111.0],
            "low": [99.0, 104.0, 109.0],
            "close": [100.5, 105.5, 110.5],
            "volume": [1000.0, 1100.0, 1200.0],
        }
        result = integration.execute_autonomous_cycle(
            "BTC/USD",
            ohlcv_data,
            0.02,
        )
        assert "symbol" in result
        assert "features" in result
        assert "prediction" in result
        assert "confidence" in result
        assert "signal" in result
        assert "volatility" in result
        assert result["symbol"] == "BTC/USD"
        assert result["volatility"] == 0.02
        assert len(integration.execution_history) == 1

    def test_execute_multiple_cycles(self) -> None:
        """Test multiple autonomous cycles track execution history."""
        integration = OrchestratorIntegration()
        ohlcv_data = {
            "open": [100.0 + i for i in range(5)],
            "high": [101.0 + i for i in range(5)],
            "low": [99.0 + i for i in range(5)],
            "close": [100.5 + i for i in range(5)],
            "volume": [1000.0] * 5,
        }
        for i in range(3):
            integration.execute_autonomous_cycle(
                f"SYMBOL_{i}",
                ohlcv_data,
                0.01 + i * 0.01,
            )
        assert len(integration.execution_history) == 3

    def test_backtest_signal_series(self) -> None:
        """Test backtesting signal series."""
        integration = OrchestratorIntegration()
        ohlcv_data = {
            "open": [100.0, 105.0, 110.0, 115.0, 120.0],
            "high": [101.0, 106.0, 111.0, 116.0, 121.0],
            "low": [99.0, 104.0, 109.0, 114.0, 119.0],
            "close": [100.5, 105.5, 110.5, 115.5, 120.5],
            "volume": [1000.0] * 5,
        }
        signals = ["BUY", "HOLD", "HOLD", "HOLD", "SELL"]
        metrics = integration.backtest_signal_series(
            "BTC/USD",
            ohlcv_data,
            signals,
        )
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "num_trades" in metrics
        assert "fitness_score" in metrics

    def test_backtest_invalid_data(self) -> None:
        """Test backtest rejects invalid data."""
        integration = OrchestratorIntegration()
        with pytest.raises(ValueError):
            integration.backtest_signal_series("", {}, [])

    def test_get_execution_history(self) -> None:
        """Test retrieving execution history."""
        integration = OrchestratorIntegration()
        ohlcv_data = {
            "open": [100.0, 105.0, 110.0],
            "high": [101.0, 106.0, 111.0],
            "low": [99.0, 104.0, 109.0],
            "close": [100.5, 105.5, 110.5],
            "volume": [1000.0, 1100.0, 1200.0],
        }
        integration.execute_autonomous_cycle(
            "BTC/USD",
            ohlcv_data,
            0.02,
        )
        history = integration.get_execution_history()
        assert len(history) == 1
        assert history[0]["symbol"] == "BTC/USD"
        assert history[0]["volatility"] == 0.02

    def test_clear_execution_history(self) -> None:
        """Test clearing execution history."""
        integration = OrchestratorIntegration()
        ohlcv_data = {
            "open": [100.0, 105.0, 110.0],
            "high": [101.0, 106.0, 111.0],
            "low": [99.0, 104.0, 109.0],
            "close": [100.5, 105.5, 110.5],
            "volume": [1000.0, 1100.0, 1200.0],
        }
        integration.execute_autonomous_cycle(
            "BTC/USD",
            ohlcv_data,
            0.02,
        )
        assert len(integration.execution_history) == 1
        integration.clear_execution_history()
        assert len(integration.execution_history) == 0


class TestGlobalIntegrationSingleton:
    """Test global orchestrator integration singleton."""

    def test_get_orchestrator_integration(self) -> None:
        """Test get_orchestrator_integration returns singleton."""
        integration1 = get_orchestrator_integration()
        integration2 = get_orchestrator_integration()
        assert integration1 is integration2
        assert isinstance(
            integration1,
            OrchestratorIntegration,
        )


class TestIntegrationWithEvolutionEngine:
    """Test orchestrator integration with evolution engine."""

    def test_integration_provides_data_for_evolution(self) -> None:
        """Test that orchestrator integration can feed data to evolution."""
        integration = OrchestratorIntegration()
        ohlcv_data = {
            "open": [100.0 + i for i in range(10)],
            "high": [101.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "volume": [1000.0] * 10,
        }
        signals = ["BUY", "HOLD"] * 5

        metrics = integration.backtest_signal_series(
            "BTC/USD",
            ohlcv_data,
            signals,
        )

        assert metrics["fitness_score"] >= 0.0
        assert metrics["fitness_score"] <= 1.0
        assert metrics["num_trades"] >= 0
