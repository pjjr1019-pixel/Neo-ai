"""Adversarial simulation and simple adversarial training helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def fgsm_attack(
    x: NDArray[np.float64],
    grad: NDArray[np.float64],
    *,
    epsilon: float = 0.01,
) -> NDArray[np.float64]:
    """Generate FGSM adversarial samples."""
    out = x + epsilon * np.sign(grad)
    return np.asarray(out, dtype=np.float64)


def pgd_attack(
    x: NDArray[np.float64],
    grad_fn,
    *,
    epsilon: float = 0.03,
    alpha: float = 0.005,
    steps: int = 5,
) -> NDArray[np.float64]:
    """Generate PGD adversarial samples with projection."""
    adv = x.copy()
    for _ in range(max(1, steps)):
        grad = np.asarray(grad_fn(adv), dtype=np.float64)
        adv = adv + alpha * np.sign(grad)
        delta = np.clip(adv - x, -epsilon, epsilon)
        adv = x + delta
    return np.asarray(adv, dtype=np.float64)


def adversarial_training_step(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    grad_fn,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return augmented batch with FGSM adversarial examples."""
    grad = grad_fn(x)
    adv = fgsm_attack(x, grad, epsilon=0.01)
    x_aug = np.vstack([x, adv])
    y_aug = np.concatenate([y, y])
    return (
        np.asarray(x_aug, dtype=np.float64),
        np.asarray(y_aug, dtype=np.float64),
    )
