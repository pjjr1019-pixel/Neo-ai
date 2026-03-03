# Phase 3 Feature Engineering

## Rolling Features
- SMA, EMA, RSI implemented in `python_ai/feature_factory.py`.
- Vectorized indicator implementations live in
  `python_ai/vectorized_indicators.py`.

## Optimizations
- Memoization for repeated indicator calls.
- Optional numba JIT path for hot SMA computation.
- Feature cache by symbol/timestamp in `data/feature_cache.py`.

## Validation
- `tests/test_feature_factory.py`
- `tests/test_data_storage_cache_io.py`
