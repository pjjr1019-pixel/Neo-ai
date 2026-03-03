# Phase 5 Automated Test Scripts

## Parallel Execution
- Added `pytest-xdist` to dev requirements.
- Added `make test-parallel` target for `pytest -n auto`.

## Synthetic Data
- Added synthetic generators in `python_ai/synthetic_data.py`.
- Supports deterministic price series and trade-like payloads.

## Validation
- `tests/test_synthetic_data.py`
- `tests/test_distributed_robustness.py`
