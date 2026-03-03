# Phase 5: Model Training and Selection

## Implemented
- Purged walk-forward CV utilities in `python_ai/model_selection.py`.
- Regime-aware scoring and significance testing utilities.
- Distributed-style hyperparameter search with Optuna parallel workers.

## Robustness
- Adversarial simulation helpers (FGSM/PGD) in `python_ai/robustness.py`.
- Adversarial training batch augmentation helper.

## Validation
- `tests/test_model_selection.py`
- `tests/test_distributed_robustness.py`
