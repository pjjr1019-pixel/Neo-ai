"""Strategy evaluation utilities for Phase 7 evolution workflows.

Provides:
- Backtest-based fitness evaluation for a population.
- Novelty scoring from parameter-space distance.
- Simple speciation by parameter similarity.
- Pareto front extraction for multi-objective selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from python_ai.evolution_engine import Strategy


@dataclass(frozen=True)
class StrategyEvaluation:
    """Evaluation snapshot for a strategy."""

    strategy: Strategy
    fitness: float
    novelty: float
    drawdown: float = 0.0
    win_rate: float = 0.0


def _numeric_vector(params: Dict[str, Any]) -> List[float]:
    """Convert numeric params into a stable sorted vector."""
    keys = sorted(params.keys())
    vec: List[float] = []
    for key in keys:
        value = params[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            vec.append(float(value))
    return vec


def _vector_distance(va: List[float], vb: List[float]) -> float:
    """Compute Euclidean distance between two numeric vectors."""
    n = max(len(va), len(vb))
    if n == 0:
        return 0.0
    pa = va + [0.0] * (n - len(va))
    pb = vb + [0.0] * (n - len(vb))
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def parameter_distance(a: Strategy, b: Strategy) -> float:
    """Compute Euclidean distance between two strategy param vectors."""
    return _vector_distance(
        _numeric_vector(a.params),
        _numeric_vector(b.params),
    )


def novelty_scores(strategies: Sequence[Strategy], k: int = 3) -> List[float]:
    """Compute novelty as average distance to k nearest neighbors."""
    n = len(strategies)
    if n <= 1:
        return [0.0] * n

    vectors = [_numeric_vector(s.params) for s in strategies]
    neighbors: List[List[float]] = [[] for _ in range(n)]
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist = _vector_distance(vectors[i], vectors[j])
            neighbors[i].append(dist)
            neighbors[j].append(dist)

    k_eff = max(1, min(k, n - 1))
    scores: List[float] = [0.0] * n
    for i in range(n):
        dists = sorted(neighbors[i])
        scores[i] = sum(dists[:k_eff]) / k_eff
    return scores


def evaluate_strategies(
    strategies: Iterable[Strategy],
    data: Dict[str, Any] | None,
    *,
    novelty_k: int = 3,
) -> List[StrategyEvaluation]:
    """Evaluate fitness + novelty for a strategy population."""
    population = list(strategies)
    fitness_values = [s.evaluate(data) for s in population]
    novelty_values = novelty_scores(population, k=novelty_k)

    return [
        StrategyEvaluation(
            strategy=s,
            fitness=f,
            novelty=n,
            drawdown=0.0,
            win_rate=0.0,
        )
        for s, f, n in zip(population, fitness_values, novelty_values)
    ]


def _dominates(a: StrategyEvaluation, b: StrategyEvaluation) -> bool:
    """Return True if `a` dominates `b` in multi-objective space."""
    return (
        a.fitness >= b.fitness
        and a.novelty >= b.novelty
        and a.win_rate >= b.win_rate
        and a.drawdown <= b.drawdown
        and (
            a.fitness > b.fitness
            or a.novelty > b.novelty
            or a.win_rate > b.win_rate
            or a.drawdown < b.drawdown
        )
    )


def pareto_front(
    evaluations: Sequence[StrategyEvaluation],
) -> List[StrategyEvaluation]:
    """Extract non-dominated strategies."""
    front: List[StrategyEvaluation] = []
    for candidate in evaluations:
        dominated = False
        for other in evaluations:
            if other is candidate:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def speciate_strategies(
    strategies: Sequence[Strategy],
    *,
    distance_threshold: float = 0.25,
) -> List[List[Strategy]]:
    """Cluster strategies into species by parameter distance."""
    species: List[List[Strategy]] = []
    for strat in strategies:
        assigned = False
        for group in species:
            rep = group[0]
            if parameter_distance(strat, rep) <= distance_threshold:
                group.append(strat)
                assigned = True
                break
        if not assigned:
            species.append([strat])
    return species


def select_top_by_composite(
    evaluations: Sequence[StrategyEvaluation],
    *,
    top_n: int,
    novelty_weight: float = 0.2,
) -> List[Strategy]:
    """Select top-N by fitness + novelty-weighted score."""
    ranked = sorted(
        evaluations,
        key=lambda item: item.fitness + novelty_weight * item.novelty,
        reverse=True,
    )
    return [item.strategy for item in ranked[: max(0, top_n)]]
