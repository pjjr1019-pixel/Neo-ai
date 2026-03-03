"""Tests for distributed HPO and adversarial robustness helpers."""

import numpy as np

from python_ai.distributed_hpo import distributed_hyperparameter_search
from python_ai.robustness import adversarial_training_step, fgsm_attack, pgd_attack


def test_distributed_hyperparameter_search_runs() -> None:
    def objective(trial):
        x = trial.suggest_float("x", -1.0, 1.0)
        return -(x - 0.2) ** 2

    result = distributed_hyperparameter_search(objective, n_trials=8, n_jobs=2)
    assert "best_value" in result
    assert "best_params" in result
    assert result["trials"] == 8


def test_fgsm_and_pgd_attacks() -> None:
    x = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    grad = np.ones_like(x)
    adv_fgsm = fgsm_attack(x, grad, epsilon=0.01)
    assert adv_fgsm.shape == x.shape

    def grad_fn(inp):
        return np.ones_like(inp)

    adv_pgd = pgd_attack(x, grad_fn, epsilon=0.03, alpha=0.01, steps=3)
    assert adv_pgd.shape == x.shape
    assert np.all(np.abs(adv_pgd - x) <= 0.03 + 1e-12)


def test_adversarial_training_step() -> None:
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)

    def grad_fn(inp):
        return np.ones_like(inp)

    x_aug, y_aug = adversarial_training_step(x, y, grad_fn)
    assert x_aug.shape[0] == 4
    assert y_aug.shape[0] == 4
