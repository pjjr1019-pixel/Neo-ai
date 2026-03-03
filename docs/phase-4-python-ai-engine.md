# Phase 4: Python AI Engine

## Endpoints
- `/predict`: model inference with confidence and signal output.
- `/learn`: async buffered online learning with retrain threshold.
- `/explain`: model feature importance (SHAP/gini fallback).
- `/metrics` and `/metrics/prometheus`: runtime and model metrics.

## Schemas
- Shared request/response schemas implemented in `python_ai/schemas.py`.
- Input validators reject NaN/Inf and malformed payloads.

## Optimizations
- RandomForest parallelized with `n_jobs=-1`.
- Prediction cache added in `MLModel.predict`.
- Async model persistence: `save_async`, `load_async`.
- Optional ONNX runtime inference hook with graceful fallback.
- Default FastAPI response class uses ORJSON when available.

## Data/Feature Enhancements
- `python_ai/feature_factory.py`:
  - vectorized indicator usage
  - memoized rolling feature calls
  - optional numba-accelerated SMA path
- `data/feature_cache.py`: per-symbol/per-timestamp feature cache.
- `data/storage.py`: CSV/Parquet storage abstraction with fallback.
- `data/io.py`: async batched candle writer.

## Validation
- Tests:
  - `tests/test_feature_factory.py`
  - `tests/test_data_storage_cache_io.py`
  - `tests/test_schemas.py`
  - `tests/test_fastapi_orjson.py`
  - `tests/test_ml_model_phase4_optimizations.py`
