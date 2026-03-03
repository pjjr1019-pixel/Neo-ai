"""Per-symbol per-timestamp feature cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CacheStats:
    """Feature cache statistics."""

    hits: int = 0
    misses: int = 0


class FeatureCache:
    """In-memory cache for computed features by symbol and timestamp."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[int, Dict[str, float]]] = {}
        self.stats = CacheStats()

    def set(
        self,
        symbol: str,
        timestamp: int,
        features: Dict[str, float],
    ) -> None:
        self._store.setdefault(symbol, {})[timestamp] = dict(features)

    def get(self, symbol: str, timestamp: int) -> Optional[Dict[str, float]]:
        per_symbol = self._store.get(symbol, {})
        payload = per_symbol.get(timestamp)
        if payload is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return dict(payload)

    def hit_ratio(self) -> float:
        total = self.stats.hits + self.stats.misses
        if total == 0:
            return 0.0
        return self.stats.hits / total
