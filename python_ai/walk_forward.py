"""
Walk-Forward Optimizer for NEO Hybrid AI.

Implements anchored and rolling walk-forward analysis
to validate trading strategies on unseen data segments.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class WalkForwardResult:
    """Result of a single walk-forward fold.

    Attributes:
        fold: Fold index (0-based).
        train_range: ``(start_idx, end_idx)`` for training.
        test_range: ``(start_idx, end_idx)`` for testing.
        train_metric: Strategy metric on in-sample data.
        test_metric: Strategy metric on out-of-sample data.
        best_params: Optimised parameters for this fold.
    """

    def __init__(
        self,
        fold: int,
        train_range: tuple[int, int],
        test_range: tuple[int, int],
        train_metric: float,
        test_metric: float,
        best_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise a fold result."""
        self.fold = fold
        self.train_range = train_range
        self.test_range = test_range
        self.train_metric = train_metric
        self.test_metric = test_metric
        self.best_params = best_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict."""
        return {
            "fold": self.fold,
            "train_range": list(self.train_range),
            "test_range": list(self.test_range),
            "train_metric": self.train_metric,
            "test_metric": self.test_metric,
            "best_params": self.best_params,
        }


class WalkForwardOptimizer:
    """Rolling or anchored walk-forward validation.

    Splits a data series into sequential train/test
    windows and evaluates a strategy optimiser on each
    fold to measure out-of-sample robustness.

    Args:
        n_folds: Number of walk-forward folds.
        train_ratio: Fraction of each window used for
            training (remainder = test).
        anchored: If ``True``, training always starts at
            index 0; if ``False``, uses a rolling window.
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_ratio: float = 0.7,
        anchored: bool = False,
    ) -> None:
        """Initialise optimizer settings."""
        self._n_folds = max(n_folds, 1)
        self._train_ratio = min(max(train_ratio, 0.1), 0.95)
        self._anchored = anchored
        self._results: List[WalkForwardResult] = []

    # ── window generation ─────────────────────────────

    def _generate_windows(
        self, data_len: int
    ) -> List[tuple[tuple[int, int], tuple[int, int]]]:
        """Compute (train, test) index ranges per fold.

        Args:
            data_len: Total number of data points.

        Returns:
            List of ``((train_start, train_end),
            (test_start, test_end))`` tuples.
        """
        windows: List[tuple[tuple[int, int], tuple[int, int]]] = []
        fold_size = data_len // self._n_folds

        for fold in range(self._n_folds):
            if self._anchored:
                train_start = 0
            else:
                train_start = fold * fold_size
            window_end = min(
                (fold + 1) * fold_size + fold_size,
                data_len,
            )
            train_len = int((window_end - train_start) * self._train_ratio)
            train_end = train_start + train_len
            test_start = train_end
            test_end = window_end

            if test_start >= test_end:
                continue

            windows.append(
                (
                    (train_start, train_end),
                    (test_start, test_end),
                )
            )
        return windows

    # ── main API ──────────────────────────────────────

    def run(
        self,
        data: List[float],
        optimize_fn: Callable[[List[float]], Dict[str, Any]],
        evaluate_fn: Callable[[List[float], Dict[str, Any]], float],
    ) -> List[WalkForwardResult]:
        """Execute walk-forward analysis.

        For each fold:
          1. Call ``optimize_fn(train_data)`` to get
             optimal parameters.
          2. Call ``evaluate_fn(train_data, params)`` →
             in-sample metric.
          3. Call ``evaluate_fn(test_data, params)`` →
             out-of-sample metric.

        Args:
            data: Full dataset (e.g. list of returns or
                prices).
            optimize_fn: Receives training slice, returns
                a parameter dict.
            evaluate_fn: Receives data slice + params,
                returns a single metric value.

        Returns:
            List of ``WalkForwardResult`` per fold.
        """
        windows = self._generate_windows(len(data))
        self._results = []

        for idx, (train_r, test_r) in enumerate(windows):
            train_slice = data[train_r[0] : train_r[1]]
            test_slice = data[test_r[0] : test_r[1]]

            params = optimize_fn(train_slice)
            train_metric = evaluate_fn(train_slice, params)
            test_metric = evaluate_fn(test_slice, params)

            result = WalkForwardResult(
                fold=idx,
                train_range=train_r,
                test_range=test_r,
                train_metric=train_metric,
                test_metric=test_metric,
                best_params=params,
            )
            self._results.append(result)
            logger.info(
                "Fold %d: train=%.4f test=%.4f",
                idx,
                train_metric,
                test_metric,
            )

        return self._results

    # ── analytics ─────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Aggregate walk-forward results.

        Returns:
            Dict with ``avg_train``, ``avg_test``,
            ``std_test``, ``overfitting_ratio``,
            and per-fold detail.
        """
        if not self._results:
            return {}
        trains = [r.train_metric for r in self._results]
        tests = [r.test_metric for r in self._results]
        avg_train = sum(trains) / len(trains)
        avg_test = sum(tests) / len(tests)
        std_test = (
            sum((t - avg_test) ** 2 for t in tests) / len(tests)
        ) ** 0.5
        of_ratio = avg_test / avg_train if avg_train != 0 else 0
        return {
            "n_folds": len(self._results),
            "avg_train": avg_train,
            "avg_test": avg_test,
            "std_test": std_test,
            "overfitting_ratio": of_ratio,
            "folds": [r.to_dict() for r in self._results],
        }
