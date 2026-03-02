"""
Circular Buffer for Streaming Data in NEO Hybrid AI.

Fixed-size, O(1) append buffer backed by a pre-allocated
numpy array.  Ideal for maintaining a sliding window of
prices or indicator values without memory allocation
churn.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CircularBuffer:
    """Fixed-capacity ring buffer backed by numpy.

    Once full, new items overwrite the oldest entries.

    Args:
        capacity: Maximum number of elements.
        dtype: Numpy dtype for the backing array.
    """

    def __init__(
        self,
        capacity: int = 1000,
        dtype: Any = np.float64,
    ) -> None:
        """Initialise an empty buffer."""
        if capacity < 1:
            raise ValueError("Capacity must be >= 1")
        self._capacity = capacity
        self._data = np.zeros(capacity, dtype=dtype)
        self._head = 0
        self._count = 0

    # ── core operations ───────────────────────────────

    def append(self, value: float) -> None:
        """Add a value to the buffer.

        Overwrites the oldest entry when full.

        Args:
            value: Value to insert.
        """
        self._data[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def extend(self, values: List[float]) -> None:
        """Append multiple values.

        Args:
            values: List of values to append in order.
        """
        for v in values:
            self.append(v)

    # ── read access ───────────────────────────────────

    def to_array(self) -> np.ndarray:  # type: ignore[type-arg]
        """Return contents in chronological order.

        Returns:
            1-D array of length ``len(self)``.
        """
        if self._count < self._capacity:
            arr = self._data[: self._count].copy()
            return arr  # type: ignore[no-any-return]
        start = self._head
        return np.concatenate(  # type: ignore[no-any-return]
            [self._data[start:], self._data[:start]]
        )

    def to_list(self) -> List[float]:
        """Return contents as a Python list."""
        return list(self.to_array().tolist())

    def latest(self, n: int = 1) -> np.ndarray:
        """Return the *n* most recent values.

        Args:
            n: How many recent values.

        Returns:
            Array of length ``min(n, len(self))``.
        """
        arr = self.to_array()
        return arr[-n:]

    @property
    def last(self) -> Optional[float]:
        """The most recently appended value."""
        if self._count == 0:
            return None
        idx = (self._head - 1) % self._capacity
        return float(self._data[idx])

    @property
    def first(self) -> Optional[float]:
        """The oldest value in the buffer."""
        if self._count == 0:
            return None
        arr = self.to_array()
        return float(arr[0])

    # ── statistics ────────────────────────────────────

    def mean(self) -> float:
        """Mean of buffered values."""
        if self._count == 0:
            return 0.0
        return float(np.mean(self.to_array()))

    def std(self) -> float:
        """Standard deviation."""
        if self._count < 2:
            return 0.0
        return float(np.std(self.to_array(), ddof=1))

    def min_val(self) -> float:
        """Minimum value."""
        if self._count == 0:
            return 0.0
        return float(np.min(self.to_array()))

    def max_val(self) -> float:
        """Maximum value."""
        if self._count == 0:
            return 0.0
        return float(np.max(self.to_array()))

    # ── properties ────────────────────────────────────

    @property
    def capacity(self) -> int:
        """Maximum number of elements."""
        return self._capacity

    @property
    def is_full(self) -> bool:
        """Whether the buffer is at capacity."""
        return self._count == self._capacity

    def __len__(self) -> int:
        """Number of elements currently stored."""
        return self._count

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"CircularBuffer(count={self._count}, "
            f"capacity={self._capacity})"
        )

    def summary(self) -> Dict[str, Any]:
        """Buffer statistics summary."""
        return {
            "count": self._count,
            "capacity": self._capacity,
            "is_full": self.is_full,
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min_val(),
            "max": self.max_val(),
        }
