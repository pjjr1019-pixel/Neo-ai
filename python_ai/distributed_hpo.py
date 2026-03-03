"""Distributed-style hyperparameter search helpers."""

from __future__ import annotations

from typing import Callable, Dict

import optuna


def distributed_hyperparameter_search(
    objective: Callable[[optuna.Trial], float],
    *,
    n_trials: int = 20,
    n_jobs: int = 2,
) -> Dict[str, object]:
    """Run Optuna search using parallel worker jobs."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=max(1, n_jobs))
    return {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "trials": len(study.trials),
    }
