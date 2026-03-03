"""Tests for strategy evaluation, speciation, and Pareto utilities."""

from python_ai.evolution_engine import Strategy
from python_ai.strategy_evaluation import (
    StrategyEvaluation,
    evaluate_strategies,
    novelty_scores,
    parameter_distance,
    pareto_front,
    select_top_by_composite,
    speciate_strategies,
)


def _sample_data():
    return {
        "ohlcv_data": {"close": [100.0, 101.0, 102.0, 103.0]},
        "signals": ["HOLD", "HOLD", "HOLD", "HOLD"],
    }


def test_parameter_distance_symmetric() -> None:
    a = Strategy({"threshold": 0.5, "stop_loss": 0.1})
    b = Strategy({"threshold": 0.7, "stop_loss": 0.2})
    d1 = parameter_distance(a, b)
    d2 = parameter_distance(b, a)
    assert d1 == d2
    assert d1 >= 0.0


def test_novelty_scores_identical_strategies() -> None:
    pop = [Strategy({"x": 1.0}) for _ in range(3)]
    scores = novelty_scores(pop, k=2)
    assert scores == [0.0, 0.0, 0.0]


def test_novelty_scores_single_strategy_zero() -> None:
    scores = novelty_scores([Strategy({"x": 1.0})], k=3)
    assert scores == [0.0]


def test_novelty_scores_k_larger_than_population() -> None:
    pop = [Strategy({"x": 0.0}), Strategy({"x": 1.0}), Strategy({"x": 3.0})]
    scores = novelty_scores(pop, k=99)
    assert len(scores) == 3
    assert all(score >= 0.0 for score in scores)


def test_parameter_distance_empty_vectors() -> None:
    a = Strategy({"name": "alpha", "flag": True})
    b = Strategy({"name": "beta", "flag": False})
    assert parameter_distance(a, b) == 0.0


def test_evaluate_strategies_returns_metrics() -> None:
    population = [Strategy({"threshold": 0.5 + i * 0.1}) for i in range(4)]
    evaluations = evaluate_strategies(population, _sample_data(), novelty_k=2)
    assert len(evaluations) == 4
    assert all(isinstance(item, StrategyEvaluation) for item in evaluations)
    assert all(isinstance(item.fitness, float) for item in evaluations)


def test_pareto_front_filters_dominated_points() -> None:
    s1 = Strategy({"x": 1.0})
    s2 = Strategy({"x": 2.0})
    s3 = Strategy({"x": 3.0})
    evals = [
        StrategyEvaluation(s1, fitness=0.9, novelty=0.9, drawdown=0.1),
        StrategyEvaluation(s2, fitness=0.8, novelty=0.7, drawdown=0.2),
        StrategyEvaluation(s3, fitness=0.9, novelty=0.5, drawdown=0.4),
    ]
    front = pareto_front(evals)
    assert evals[0] in front
    assert evals[1] not in front


def test_speciate_strategies_clusters_by_distance() -> None:
    population = [
        Strategy({"threshold": 0.10}),
        Strategy({"threshold": 0.11}),
        Strategy({"threshold": 2.00}),
    ]
    species = speciate_strategies(population, distance_threshold=0.05)
    assert len(species) == 2
    sizes = sorted(len(group) for group in species)
    assert sizes == [1, 2]


def test_select_top_by_composite_respects_top_n() -> None:
    s1 = Strategy({"x": 1.0})
    s2 = Strategy({"x": 2.0})
    s3 = Strategy({"x": 3.0})
    evals = [
        StrategyEvaluation(s1, fitness=0.5, novelty=0.1),
        StrategyEvaluation(s2, fitness=0.4, novelty=1.0),
        StrategyEvaluation(s3, fitness=0.2, novelty=0.0),
    ]
    selected = select_top_by_composite(evals, top_n=2, novelty_weight=0.3)
    assert len(selected) == 2
    assert all(isinstance(s, Strategy) for s in selected)
