"""Tests for strategy lifecycle tracking and warm-start logic."""

import pytest

from python_ai.evolution_engine import Strategy
from python_ai.strategy_lifecycle import StrategyLifecycleManager


def test_register_record_and_retire() -> None:
    manager = StrategyLifecycleManager()
    sid = manager.register(Strategy({"threshold": 0.5}))
    manager.record_fitness(sid, 0.75)
    assert sid in manager.active
    assert manager.active[sid].fitness_history[-1] == 0.75

    manager.retire(sid, reason="underperforming")
    assert sid not in manager.active
    assert sid in manager.archive
    assert manager.archive[sid].retired_reason == "underperforming"


def test_family_tree_tracks_parents() -> None:
    manager = StrategyLifecycleManager()
    parent = manager.register(Strategy({"x": 1.0}))
    child = manager.register(
        Strategy({"x": 1.2}),
        parent_ids=[parent],
        generation=1,
    )
    ancestors = manager.family_tree(child)
    assert parent in ancestors


def test_complexity_penalty_positive() -> None:
    manager = StrategyLifecycleManager()
    penalty = manager.complexity_penalty(
        Strategy({"a": 10.0, "b": [1, 2], "c": "name"})
    )
    assert penalty > 0.0


def test_age_adjusted_fitness_decays() -> None:
    manager = StrategyLifecycleManager()
    sid = manager.register(Strategy({"x": 1.0}))
    manager.record_fitness(sid, 1.0)
    manager.active[sid].created_at -= 10.0
    decayed = manager.age_adjusted_fitness(sid, decay=0.9)
    assert 0.0 < decayed < 1.0


def test_warm_start_population_returns_copies() -> None:
    manager = StrategyLifecycleManager()
    s1 = manager.register(Strategy({"x": 1.0}))
    s2 = manager.register(Strategy({"x": 2.0}))
    manager.record_fitness(s1, 0.8)
    manager.record_fitness(s2, 0.2)

    warm = manager.warm_start_population(top_n=1)
    assert len(warm) == 1
    assert warm[0].params["x"] == 1.0
    # Deep copy: mutating warm-start should not alter tracked strategy.
    warm[0].params["x"] = 99.0
    assert manager.active[s1].strategy.params["x"] == 1.0


def test_record_fitness_unknown_id_raises() -> None:
    manager = StrategyLifecycleManager()
    with pytest.raises(KeyError):
        manager.record_fitness("missing", 0.5)


def test_retire_unknown_id_raises() -> None:
    manager = StrategyLifecycleManager()
    with pytest.raises(KeyError):
        manager.retire("missing")


def test_family_tree_unknown_id_returns_empty() -> None:
    manager = StrategyLifecycleManager()
    assert manager.family_tree("missing") == []


def test_age_adjusted_fitness_no_history_is_zero() -> None:
    manager = StrategyLifecycleManager()
    sid = manager.register(Strategy({"x": 1.0}))
    assert manager.age_adjusted_fitness(sid) == 0.0


def test_age_adjusted_fitness_unknown_id_raises() -> None:
    manager = StrategyLifecycleManager()
    with pytest.raises(KeyError):
        manager.age_adjusted_fitness("missing")
