"""Async batched candle writer for storage backends."""

from __future__ import annotations

import asyncio
from typing import List

from data.storage import Candle, CandleStorage, StorageFormat


class AsyncBatchCandleWriter:
    """Batch candles in memory and flush asynchronously to storage."""

    def __init__(
        self,
        storage: CandleStorage,
        symbol: str,
        *,
        batch_size: int = 50,
        fmt: StorageFormat = "parquet",
    ) -> None:
        self.storage = storage
        self.symbol = symbol
        self.batch_size = max(1, batch_size)
        self.fmt = fmt
        self._buffer: List[Candle] = []

    async def add(self, candle: Candle) -> None:
        """Add a candle and flush if threshold is reached."""
        self._buffer.append(candle)
        if len(self._buffer) >= self.batch_size:
            await self.flush()

    async def flush(self) -> None:
        """Flush current buffer to storage asynchronously."""
        if not self._buffer:
            return
        payload = list(self._buffer)
        self._buffer.clear()
        await asyncio.to_thread(
            self.storage.save,
            self.symbol,
            payload,
            self.fmt,
        )
