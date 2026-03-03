"""Storage abstraction for CSV/Parquet candle persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Literal, TypedDict

import pandas as pd


class Candle(TypedDict):
    """Candle row schema."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


StorageFormat = Literal["csv", "parquet"]


class CandleStorage:
    """Persist and load candle series in CSV or Parquet format."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, fmt: StorageFormat) -> Path:
        safe = symbol.replace("/", "_")
        return self.base_dir / f"{safe}.{fmt}"

    @staticmethod
    def _parquet_available() -> bool:
        try:
            import pyarrow  # noqa: F401

            return True
        except Exception:
            try:
                import fastparquet  # noqa: F401

                return True
            except Exception:
                return False

    def save(
        self,
        symbol: str,
        rows: Iterable[Candle],
        fmt: StorageFormat = "parquet",
    ) -> Path:
        frame = pd.DataFrame(list(rows))
        path = self._path(symbol, fmt)
        if fmt == "parquet" and self._parquet_available():
            frame.to_parquet(path, index=False)
        else:
            path = self._path(symbol, "csv")
            frame.to_csv(path, index=False)
        return path

    def load(
        self,
        symbol: str,
        fmt: StorageFormat = "parquet",
    ) -> List[Candle]:
        path = self._path(symbol, fmt)
        if fmt == "parquet" and not path.exists():
            path = self._path(symbol, "csv")
        if fmt == "parquet" and not self._parquet_available():
            path = self._path(symbol, "csv")
        if not path.exists():
            return []
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        return frame.to_dict(orient="records")
