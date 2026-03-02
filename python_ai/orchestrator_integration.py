"""
Python-Java Orchestrator Integration for NEO Hybrid AI.

Integrates data pipeline, ML model, and backtesting for autonomous
trading loop execution via Java orchestrator. Provides REST API bridge
for feature computation and prediction with real model inference.
"""

from typing import Any, Dict, List, Optional

from python_ai.backtesting_engine import get_backtesting_engine
from python_ai.data_pipeline import get_pipeline
from python_ai.ml_model import get_model


class OrchestratorIntegration:
    """Integration layer for Java orchestrator with Python ML system."""

    def __init__(self) -> None:
        """Initialize orchestrator integration with pipeline and model."""
        self.pipeline = get_pipeline()
        self.model = get_model()
        self.backtest_engine = get_backtesting_engine()
        self.execution_history: List[Dict[str, Any]] = []

    def compute_features_from_prices(
        self,
        symbol: str,
        ohlcv_data: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """Compute features from raw OHLCV price data.

        This method integrates the data pipeline to convert raw prices
        into feature vectors for model inference.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USD').
            ohlcv_data: Dict with 'open', 'high', 'low', 'close',
                       'volume' lists.

        Returns:
            10-element feature dict (f0-f9) normalized and ready for
            prediction.

        Raises:
            ValueError: If price data is invalid.
        """
        if not symbol or not ohlcv_data:
            raise ValueError("Symbol and OHLCV data required")

        close_prices = ohlcv_data.get("close", [])
        if not close_prices or len(close_prices) < 2:
            raise ValueError("OHLCV data must have at least 2 price points")

        try:
            self.pipeline.update_price_data(symbol, ohlcv_data)
            features = self.pipeline.compute_features(symbol)
            return features
        except Exception as e:
            raise ValueError(
                f"Feature computation failed for {symbol}: {str(e)}"
            )

    def predict_with_model(
        self,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """Run ML model inference on computed features.

        Performs prediction with confidence calibration and signal
        generation (BUY/SELL).

        Args:
            features: Feature dict from compute_features_from_prices().

        Returns:
            Dict with 'prediction', 'confidence', 'signal'.

        Raises:
            ValueError: If features are invalid.
        """
        if not isinstance(features, dict):
            raise ValueError("Features must be a dictionary")

        try:
            pred, confidence, signal = self.model.predict(features)
            return {
                "prediction": float(pred),
                "confidence": float(confidence),
                "signal": signal,
            }
        except Exception as e:
            raise ValueError(f"Model prediction failed: {str(e)}")

    def execute_autonomous_cycle(
        self,
        symbol: str,
        ohlcv_data: Dict[str, List[float]],
        current_volatility: float,
    ) -> Dict[str, Any]:
        """Execute full autonomous trading cycle.

        Combines feature computation, model prediction, and execution
        tracking for autonomous loop operation.

        Args:
            symbol: Trading symbol.
            ohlcv_data: Raw OHLCV price data.
            current_volatility: Current market volatility (0-1).

        Returns:
            Execution result with signal, confidence, and timestamp.
        """
        try:
            features = self.compute_features_from_prices(
                symbol,
                ohlcv_data,
            )
            prediction_result = self.predict_with_model(features)

            execution_record = {
                "symbol": symbol,
                "features": features,
                "prediction": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "signal": prediction_result["signal"],
                "volatility": current_volatility,
                "timestamp": None,
            }
            self.execution_history.append(execution_record)

            return execution_record
        except Exception as e:
            raise ValueError(f"Autonomous cycle failed: {str(e)}")

    def backtest_signal_series(
        self,
        symbol: str,
        ohlcv_data: Dict[str, List[float]],
        signals: List[str],
    ) -> Dict[str, Any]:
        """Backtest a series of trading signals.

        Evaluates signal performance using historical data for
        evolution engine feedback.

        Args:
            symbol: Trading symbol.
            ohlcv_data: Historical price data.
            signals: List of trading signals ('BUY', 'SELL', 'HOLD').

        Returns:
            Backtest metrics dict with fitness score.
        """
        if not symbol or not ohlcv_data or not signals:
            raise ValueError("Symbol, OHLCV data, and signals required")

        try:
            metrics = self.backtest_engine.run_backtest(
                ohlcv_data,
                signals,
            )
            return metrics.to_dict()
        except Exception as e:
            raise ValueError(f"Backtest failed: {str(e)}")

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get record of all executed autonomous cycles.

        Returns:
            List of execution records with signals and results.
        """
        return self.execution_history.copy()

    def clear_execution_history(self) -> None:
        """Clear execution history for fresh runs."""
        self.execution_history.clear()


def get_orchestrator_integration() -> OrchestratorIntegration:
    """Get global orchestrator integration singleton.

    Returns:
        OrchestratorIntegration instance.
    """
    global _orchestrator_integration
    if _orchestrator_integration is None:
        _orchestrator_integration = OrchestratorIntegration()
    return _orchestrator_integration


_orchestrator_integration: Optional[OrchestratorIntegration] = None
