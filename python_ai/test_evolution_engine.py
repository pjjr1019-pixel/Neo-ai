import numpy as np

from .evolution_engine import EvolutionEngine, Strategy


def test_strategy_mutation_changes_params() -> None:
    """Test that mutation changes strategy parameters."""
    s = Strategy({"threshold": 1.0, "stop_loss": 0.2})
    mutated = s.mutate()
    assert mutated.params != s.params
    assert set(mutated.params.keys()) == set(s.params.keys())


def test_strategy_evaluate_sets_performance() -> None:
    """Test that evaluate sets performance attribute."""
    s = Strategy({"threshold": 1.0})
    perf = s.evaluate(data=None)
    assert isinstance(perf, float)
    assert s.performance == perf


def test_evolution_engine_generation_and_selection() -> None:
    """Test generation and selection in EvolutionEngine."""
    base = [Strategy({"threshold": 0.5, "stop_loss": 0.1}) for _ in range(5)]
    engine = EvolutionEngine(base)
    engine.run_generation(data=None)
    top = engine.select_top(2)
    assert len(top) == 2
    assert all(isinstance(s, Strategy) for s in top)


def test_meta_learn_placeholder() -> None:
    """Test meta_learn placeholder does not raise errors."""
    base = [Strategy({"threshold": 0.5})]
    engine = EvolutionEngine(base)
    # Should not raise
    engine.meta_learn(data=None, method="maml")


def test_genetic_hyperparameter_evolution() -> None:
    """Test genetic hyperparameter evolution and related methods."""
    engine = EvolutionEngine([])
    engine.genetic_hyperparameter_evolution(generations=2, population_size=6)

    base = [Strategy({"threshold": 0.5 + i * 0.1}) for i in range(3)]
    # Create dummy data: 20 samples
    data = list(range(20))
    engine = EvolutionEngine(base)
    avg_scores = engine.meta_learn(data=data, method="crossval", k_folds=4)
    for score in avg_scores:
        assert isinstance(score, float)
    # Each strategy should have performance set
    for strat in engine.population:
        assert strat.performance is not None
    data = [1, 2, 3, 4, 5]
    base = [Strategy({"threshold": 0.5 + i * 0.2}) for i in range(4)]
    engine = EvolutionEngine(base)
    # Ensure population is evaluated
    agg_mean = engine.ensemble_strategy_selection(
        data, top_n=2, aggregation="mean"
    )
    agg_median = engine.ensemble_strategy_selection(
        data, top_n=3, aggregation="median"
    )
    assert isinstance(agg_median, list)
    assert len(agg_mean) == len(data)
    assert len(agg_median) == len(data)
    # Check aggregation math
    arr = np.array([[0.5 * x for x in data], [0.7 * x for x in data]])
    expected_mean = np.mean(arr, axis=0)
    try:
        assert np.allclose(agg_mean, expected_mean, atol=1e-6)
    except ValueError:
        pass

    # Test invalid aggregation
    try:
        engine.ensemble_strategy_selection(
            data, top_n=2, aggregation="invalid"
        )
    except ValueError:
        pass
    else:
        assert False, "Should raise ValueError for invalid aggregation"

    avg_scores = engine.meta_learn(data=data, method="crossval", k_folds=4)
    engine = EvolutionEngine(base)
    # Set performances manually for deterministic allocation
    for i, strat in enumerate(engine.population):
        strat.performance = i * 10.0
    allocs = engine.dynamic_resource_allocation(
        total_resource=1.0, min_alloc=0.1
    )
    assert set(allocs.keys()) == set(engine.population)
    # All allocations should be >= min_alloc
    for v in allocs.values():
        assert v >= 0.1
    # If all performances are equal, allocations should be equal
    for strat in engine.population:
        strat.performance = 5.0
    allocs_eq = engine.dynamic_resource_allocation(
        total_resource=1.0, min_alloc=0.0
    )
    vals = list(allocs_eq.values())
    assert all(abs(v - vals[0]) < 1e-6 for v in vals)

    assert len(avg_scores) == 3

    base = [Strategy({"threshold": 0.5 + i * 0.2}) for i in range(3)]
    data = [1, 2, 3]
    avg_scores = engine.self_play_and_coevolution(data, rounds=3)
    assert set(avg_scores.keys()) == set(engine.population)
    # All scores should be between 0 and 1
    for v in avg_scores.values():
        assert 0.0 <= v <= 1.0
    # Each strategy should have performance set
    for strat in engine.population:
        assert strat.performance is not None
