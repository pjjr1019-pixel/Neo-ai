"""
Training Data Builder for NEO Hybrid AI.

Bridges the gap between stored historical candles and ML model training.
Loads candles from HistoricalDataStore, converts to columnar OHLCV format,
computes features via DataPipeline, and generates target labels.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from python_ai.data_ingestion_api import (
    DataIngestionAPI,
    get_data_ingestion_api,
)
from python_ai.data_pipeline import DataPipeline, get_pipeline

logger = logging.getLogger(__name__)


def candles_to_ohlcv(
    candles: List[Dict[str, float]],
) -> Dict[str, List[float]]:
    """Convert row-oriented candle dicts to columnar OHLCV format.

    Args:
        candles: List of dicts with open, high, low, close, volume keys.

    Returns:
        Dict with 'open', 'high', 'low', 'close', 'volume' lists.

    Raises:
        ValueError: If candles list is empty.
    """
    if not candles:
        raise ValueError("Cannot convert empty candles list")

    return {
        "open": [float(c["open"]) for c in candles],
        "high": [float(c["high"]) for c in candles],
        "low": [float(c["low"]) for c in candles],
        "close": [float(c["close"]) for c in candles],
        "volume": [float(c["volume"]) for c in candles],
    }


class TrainingDataBuilder:
    """Build training datasets from historical candle data.

    Loads stored candles, computes feature vectors over sliding windows,
    and generates forward-return targets for supervised learning.
    """

    def __init__(
        self,
        pipeline: Optional[DataPipeline] = None,
        ingestion_api: Optional[DataIngestionAPI] = None,
        window_size: int = 30,
        target_horizon: int = 1,
    ) -> None:
        """Initialize training data builder.

        Args:
            pipeline: DataPipeline instance (default: global singleton).
            ingestion_api: DataIngestionAPI instance (default: singleton).
            window_size: Minimum bars needed for indicator calculation.
            target_horizon: Number of bars ahead for target return.
        """
        self.pipeline = pipeline or get_pipeline()
        self.ingestion_api = ingestion_api or get_data_ingestion_api()
        self.window_size = window_size
        self.target_horizon = target_horizon

    def build_from_candles(
        self,
        symbol: str,
        candles: List[Dict[str, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build training data from a list of candles.

        Computes features at each timestep using a sliding window
        and pairs them with forward returns as targets.

        Args:
            symbol: Trading symbol (used as pipeline key).
            candles: List of OHLCV candle dicts, chronologically ordered.

        Returns:
            Tuple of (X, y) where X is (n_samples, 10) features
            and y is (n_samples,) forward returns.

        Raises:
            ValueError: If not enough candles for window + target.
        """
        min_required = self.window_size + self.target_horizon
        if len(candles) < min_required:
            raise ValueError(
                f"Need at least {min_required} candles, got {len(candles)}"
            )

        ohlcv = candles_to_ohlcv(candles)
        closes = ohlcv["close"]

        features_list: List[List[float]] = []
        targets: List[float] = []

        end_idx = len(candles) - self.target_horizon
        for i in range(self.window_size, end_idx):
            window_candles = candles[: i + 1]
            window_ohlcv = candles_to_ohlcv(window_candles)

            self.pipeline.update_price_data(symbol, window_ohlcv)
            feature_dict = self.pipeline.compute_features(symbol)

            feature_values = [feature_dict[f"f{j}"] for j in range(10)]
            features_list.append(feature_values)

            current_close = closes[i]
            future_close = closes[i + self.target_horizon]
            if current_close != 0:
                forward_return = (future_close - current_close) / current_close
            else:
                forward_return = 0.0
            targets.append(forward_return)

        X = np.array(features_list, dtype=np.float64)
        y = np.array(targets, dtype=np.float64)

        logger.info(
            "Built training data: %d samples, %d features from %s",
            X.shape[0],
            X.shape[1],
            symbol,
        )

        return X, y

    def build_from_store(
        self,
        symbol: str,
        limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build training data from stored historical candles.

        Loads candles from HistoricalDataStore and delegates to
        build_from_candles.

        Args:
            symbol: Trading symbol to load.
            limit: Max candles to load (None = all).

        Returns:
            Tuple of (X, y) numpy arrays.

        Raises:
            ValueError: If no candles found for symbol.
        """
        candles = self.ingestion_api.get_historical_data(symbol, limit=limit)
        if not candles:
            raise ValueError(f"No historical data found for {symbol}")

        return self.build_from_candles(symbol, candles)


_builder: Optional[TrainingDataBuilder] = None


def get_training_data_builder() -> TrainingDataBuilder:
    """Get global TrainingDataBuilder singleton.

    Returns:
        TrainingDataBuilder instance.
    """
    global _builder
    if _builder is None:
        _builder = TrainingDataBuilder()
    return _builder
