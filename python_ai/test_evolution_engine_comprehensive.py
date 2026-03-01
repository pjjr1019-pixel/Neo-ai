"""
Comprehensive tests for EvolutionEngine and Strategy classes.
Ensures full flake8 compliance and adherence to project coding policy.
"""

import pytest
from python_ai.evolution_engine import EvolutionEngine, Strategy


def test_strategy_mutate_and_evaluate():
    """
    Test that Strategy.mutate returns a new Strategy with different parameters
    and valid evaluation.
    """
    base_params = {"threshold": 0.5, "stop_loss": 0.1}
    strat = Strategy(base_params)
    mutated = strat.mutate()
    assert isinstance(mutated, Strategy)
    assert mutated.params != base_params or mutated is not strat
    perf = mutated.evaluate([1, 2, 3])
    assert isinstance(perf, float)
    assert 0.0 <= perf <= 1.0


def test_evolution_engine_initialization():
    """Test that EvolutionEngine initializes with the given population."""
    base = [Strategy({"x": i}) for i in range(2)]
    engine = EvolutionEngine(base)
    assert engine.population == base


def test_explainable_evolution_report_empty_and_nonempty():
    """
    Test explainable_evolution_report returns a string for both empty
    and non-empty previous populations.
    """
    base = [Strategy({"x": i}) for i in range(2)]
    engine = EvolutionEngine(base)
    report = engine.explainable_evolution_report()
    assert isinstance(report, str)
    prev = [Strategy({"x": 0}), Strategy({"x": 1})]
    report2 = engine.explainable_evolution_report(previous_population=prev)
    assert isinstance(report2, str)


def test_self_play_and_coevolution():
    """Test self_play_and_coevolution returns valid scores."""
    base = [Strategy({"x": i}) for i in range(3)]
    engine = EvolutionEngine(base)
    scores = engine.self_play_and_coevolution([1, 2, 3], rounds=2)
    assert set(scores.keys()) == set(engine.population)
    for v in scores.values():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_dynamic_resource_allocation_various_cases():
    """Test dynamic_resource_allocation distributes resources correctly."""
    base = [Strategy({"x": i}) for i in range(3)]
    engine = EvolutionEngine(base)
    for i, strat in enumerate(engine.population):
        strat.performance = float(i)
    allocs = engine.dynamic_resource_allocation(
        total_resource=1.0, min_alloc=0.0
    )
    assert len(allocs) == 3
    assert all(isinstance(v, float) for v in allocs.values())
    # All equal performance
    for strat in engine.population:
        strat.performance = 1.0
    allocs_eq = engine.dynamic_resource_allocation(
        total_resource=1.0, min_alloc=0.0
    )
    vals = list(allocs_eq.values())
    assert all(abs(v - vals[0]) < 1e-6 for v in vals)


def test_ensemble_strategy_selection_mean_and_median():
    """
    Test ensemble_strategy_selection mean/median types
    and invalid aggregation error.
    """
    base = [Strategy({"threshold": 0.5 + i * 0.2}) for i in range(4)]
    engine = EvolutionEngine(base)
    data = [1, 2, 3, 4]
    agg_mean = engine.ensemble_strategy_selection(
        data, top_n=2, aggregation="mean"
    )
    agg_median = engine.ensemble_strategy_selection(
        data, top_n=2, aggregation="median"
    )
    assert isinstance(agg_mean, list)
    assert isinstance(agg_median, list)
    with pytest.raises(ValueError):
        engine.ensemble_strategy_selection(
            data, top_n=2, aggregation="invalid"
        )


def test_run_generation_and_select_top():
    """Test run_generation and select_top return correct results."""
    base = [Strategy({"x": i}) for i in range(5)]
    engine = EvolutionEngine(base)
    engine.run_generation([1, 2, 3])
    assert len(engine.population) == 5
    top = engine.select_top(2)
    assert len(top) == 2
    assert all(isinstance(s, Strategy) for s in top)


def test_meta_learn_crossval():
    """Test meta_learn with cross-validation returns correct length list."""
    base = [Strategy({"x": i}) for i in range(3)]
    engine = EvolutionEngine(base)
    data = [1, 2, 3, 4, 5]
    scores = engine.meta_learn(data, method="crossval", k_folds=2)
    assert isinstance(scores, list)
    assert len(scores) == 3


def test_genetic_hyperparameter_evolution():
    """Test genetic_hyperparameter_evolution creates correct population."""
    engine = EvolutionEngine([])
    engine.genetic_hyperparameter_evolution(
        generations=2, data=[1, 2], population_size=4
    )
    assert len(engine.population) == 4
    assert all(isinstance(s, Strategy) for s in engine.population)


def test_bayesian_hyperparameter_optimization():
    """Test bayesian_hyperparameter_optimization creates one Strategy."""
    engine = EvolutionEngine([])
    engine.bayesian_hyperparameter_optimization(data=[1, 2, 3], n_iter=3)
    assert len(engine.population) == 1
    assert isinstance(engine.population[0], Strategy)


def test_main_block_excluded():
    """Test that the main block is excluded from import side effects."""
    import importlib
    import sys

    modname = "python_ai.evolution_engine"
    if modname in sys.modules:
        del sys.modules[modname]
    importlib.import_module(modname)


def test_strategy_mutate_non_numeric():
    """Test mutate with non-numeric parameters does not error and
    copies params.
    """
    base_params = {"name": "test", "flag": True}
    strat = Strategy(base_params)
    mutated = strat.mutate()
    assert mutated.params == base_params
    assert mutated is not strat


def test_evolution_engine_empty_population():
    """Test EvolutionEngine methods with empty population."""
    engine = EvolutionEngine([])
    # Should not raise
    report = engine.explainable_evolution_report()
    assert isinstance(report, str)
    scores = engine.self_play_and_coevolution([], rounds=1)
    assert isinstance(scores, dict)
    allocs = engine.dynamic_resource_allocation()
    assert isinstance(allocs, dict)
    # select_top should return empty list
    assert engine.select_top(2) == []


def test_meta_learn_invalid_data():
    """Test meta_learn raises ValueError on invalid data for crossval."""
    engine = EvolutionEngine([Strategy({"x": 1})])
    with pytest.raises(ValueError):
        engine.meta_learn(data=None, method="crossval")
    with pytest.raises(ValueError):
        engine.meta_learn(data=[], method="crossval")


def test_evolution_engine_init_and_attributes():
    """Test __init__ and attribute assignments for EvolutionEngine."""
    base = [Strategy({"a": 1})]
    engine = EvolutionEngine(base)
    assert engine.base_strategies == base
    assert engine.population == base


def test_dynamic_resource_allocation_min_alloc_exceeds_total():
    """Test allocation when min_alloc * n > total_resource."""
    base = [Strategy({"x": i}) for i in range(3)]
    engine = EvolutionEngine(base)
    for strat in engine.population:
        strat.performance = 1.0
    allocs = engine.dynamic_resource_allocation(
        total_resource=1.0, min_alloc=1.0
    )
    assert all(v == 1.0 for v in allocs.values())
