"""
Data Ingestion API for NEO Hybrid AI.

Handles real OHLCV data ingestion from market sources, validation,
and historical data storage for backtesting and model training.
"""

from typing import Any, Dict, List, Optional
import csv
from datetime import datetime
from pathlib import Path

import numpy as np


class DataValidator:
    """Validate OHLCV data for quality and integrity."""

    @staticmethod
    def validate_candle(candle: Dict[str, float]) -> bool:
        """Validate a single OHLCV candle.

        Args:
            candle: OHLCV dict with open, high, low, close, volume.

        Returns:
            True if candle passes validation.
        """
        required_keys = {"open", "high", "low", "close", "volume"}
        if not all(key in candle for key in required_keys):
            return False

        high = candle.get("high", 0)
        low = candle.get("low", 0)
        open_price = candle.get("open", 0)
        close_price = candle.get("close", 0)

        if high < low:
            return False

        if high < open_price or high < close_price:
            return False

        if low > open_price or low > close_price:
            return False

        volume = candle.get("volume", 0)
        if volume < 0:
            return False

        return True

    @staticmethod
    def validate_price_series(
        prices: List[float],
        max_price_jump: float = 0.1,
    ) -> bool:
        """Validate price series for anomalies.

        Args:
            prices: List of prices.
            max_price_jump: Max allowed price change (0.1 = 10%).

        Returns:
            True if prices pass validation.
        """
        if len(prices) < 2:
            return True

        for i in range(1, len(prices)):
            if prices[i - 1] == 0:
                continue

            price_change = abs((prices[i] - prices[i - 1]) / prices[i - 1])
            if price_change > max_price_jump:
                return False

        return True


class HistoricalDataStore:
    """Store and retrieve historical OHLCV data."""

    def __init__(self, data_dir: str = "data/historical") -> None:
        """Initialize data store.

        Args:
            data_dir: Directory for storing historical data files.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.in_memory_cache: Dict[str, List[Dict]] = {}

    def save_candles(
        self,
        symbol: str,
        candles: List[Dict[str, float]],
        append: bool = True,
    ) -> bool:
        """Save candles to disk.

        Args:
            symbol: Trading symbol.
            candles: List of OHLCV candles.
            append: Append to existing file if True.

        Returns:
            True if save successful.
        """
        try:
            file_path = self.data_dir / f"{symbol}.csv"
            file_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append and file_path.exists() else "w"
            with open(file_path, mode, newline="") as f:
                if mode == "w":
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        ],
                    )
                    writer.writeheader()
                else:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        ],
                    )

                for candle in candles:
                    row = {
                        "timestamp": datetime.now().isoformat(),
                        "open": candle.get("open"),
                        "high": candle.get("high"),
                        "low": candle.get("low"),
                        "close": candle.get("close"),
                        "volume": candle.get("volume"),
                    }
                    writer.writerow(row)

            self.in_memory_cache[symbol] = candles
            return True
        except Exception as e:
            print(f"Failed to save candles for {symbol}: {e}")
            return False

    def load_candles(
        self,
        symbol: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Load candles from disk.

        Args:
            symbol: Trading symbol.
            limit: Maximum number of candles to load.

        Returns:
            List of OHLCV candles.
        """
        file_path = self.data_dir / f"{symbol}.csv"
        candles = []

        if not file_path.exists():
            return candles

        try:
            with open(file_path, "r") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if limit and i >= limit:
                        break

                    candle = {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                    candles.append(candle)

            self.in_memory_cache[symbol] = candles
            return candles
        except Exception as e:
            print(f"Failed to load candles for {symbol}: {e}")
            return []

    def get_latest_candle(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get the latest candle for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            Latest OHLCV candle or None.
        """
        candles = self.load_candles(symbol, limit=1)
        return candles[0] if candles else None


class DataIngestionAPI:
    """Main API for data ingestion and management."""

    def __init__(
        self,
        data_dir: str = "data/historical",
    ) -> None:
        """Initialize data ingestion API.

        Args:
            data_dir: Directory for historical data storage.
        """
        self.store = HistoricalDataStore(data_dir)
        self.validator = DataValidator()
        self.ingestion_stats: Dict[str, int] = {}

    def ingest_candles(
        self,
        symbol: str,
        candles: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Ingest and validate OHLCV candles.

        Args:
            symbol: Trading symbol.
            candles: List of OHLCV candles.

        Returns:
            Ingestion result with counts and validation.
        """
        valid_candles = []
        invalid_count = 0

        for candle in candles:
            if self.validator.validate_candle(candle):
                valid_candles.append(candle)
            else:
                invalid_count += 1

        if valid_candles:
            close_prices = [c["close"] for c in valid_candles]
            if not self.validator.validate_price_series(close_prices):
                return {
                    "success": False,
                    "message": "Price series validation failed",
                    "valid": 0,
                    "invalid": len(candles),
                }

            success = self.store.save_candles(
                symbol,
                valid_candles,
                append=True,
            )

            self.ingestion_stats[symbol] = self.ingestion_stats.get(
                symbol, 0
            ) + len(valid_candles)

            return {
                "success": success,
                "message": f"Ingested {len(valid_candles)} candles",
                "valid": len(valid_candles),
                "invalid": invalid_count,
                "symbol": symbol,
            }

        return {
            "success": False,
            "message": "No valid candles found",
            "valid": 0,
            "invalid": len(candles),
        }

    def get_historical_data(
        self,
        symbol: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Get historical OHLCV data.

        Args:
            symbol: Trading symbol.
            limit: Max candles to return.

        Returns:
            List of OHLCV candles.
        """
        return self.store.load_candles(symbol, limit=limit)

    def get_data_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistics for ingested data.

        Args:
            symbol: Trading symbol.

        Returns:
            Data statistics dict.
        """
        candles = self.store.load_candles(symbol)
        if not candles:
            return {
                "symbol": symbol,
                "total_candles": 0,
                "price_range": None,
            }

        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]

        return {
            "symbol": symbol,
            "total_candles": len(candles),
            "price_min": float(np.min(closes)),
            "price_max": float(np.max(closes)),
            "price_mean": float(np.mean(closes)),
            "price_std": float(np.std(closes)),
            "volume_min": float(np.min(volumes)),
            "volume_max": float(np.max(volumes)),
            "volume_mean": float(np.mean(volumes)),
        }

    def get_ingestion_summary(self) -> Dict[str, Any]:
        """Get summary of all ingestion activity.

        Returns:
            Summary of ingestion statistics.
        """
        total_candles = sum(self.ingestion_stats.values())
        return {
            "total_candles_ingested": total_candles,
            "symbols_processed": list(self.ingestion_stats.keys()),
            "ingestion_stats": self.ingestion_stats.copy(),
        }


def get_data_ingestion_api(
    data_dir: str = "data/historical",
) -> DataIngestionAPI:
    """Get global data ingestion API singleton.

    Args:
        data_dir: Directory for historical data.

    Returns:
        DataIngestionAPI instance.
    """
    global _data_ingestion_api
    if _data_ingestion_api is None:
        _data_ingestion_api = DataIngestionAPI(data_dir)
    return _data_ingestion_api


_data_ingestion_api: Optional[DataIngestionAPI] = None
