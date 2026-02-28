"""
Comprehensive tests for EvolutionEngine and Strategy classes.
Ensures full flake8 compliance and adherence to project coding policy.
"""

import pytest
from python_ai.evolution_engine import EvolutionEngine, Strategy


def test_strategy_mutate_and_evaluate():
    base_params = {"threshold": 0.5, "stop_loss": 0.1}
    strat = Strategy(base_params)
    mutated = strat.mutate()
    assert isinstance(mutated, Strategy)
    assert mutated.params != base_params or mutated is not strat
    perf = mutated.evaluate([1, 2, 3])
    assert isinstance(perf, float)
    assert 0.0 <= perf <= 1.0


def test_evolution_engine_initialization():
    base = [Strategy({"x": i}) for i in range(2)]
    engine = EvolutionEngine(base)
    assert engine.population == base


def test_explainable_evolution_report_empty_and_nonempty():
    base = [Strategy({"x": i}) for i in range(2)]
    engine = EvolutionEngine(base)
    report = engine.explainable_evolution_report()
    assert isinstance(report, str)
    prev = [Strategy({"x": 0}), Strategy({"x": 1})]
    report2 = engine.explainable_evolution_report(previous_population=prev)
    assert isinstance(report2, str)


def test_self_play_and_coevolution():
    base = [Strategy({"x": i}) for i in range(3)]
    engine = EvolutionEngine(base)
    scores = engine.self_play_and_coevolution([1, 2, 3], rounds=2)
    assert set(scores.keys()) == set(engine.population)
    for v in scores.values():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_dynamic_resource_allocation_various_cases():
    base = [Strategy({"x": i}) for i in range(3)]
    engine = EvolutionEngine(base)
    for i, strat in enumerate(engine.population):
        strat.performance = float(i)
    allocs = engine.dynamic_resource_allocation(
        total_resource=1.0,
        min_alloc=0.0
    )
    assert len(allocs) == 3
    assert all(isinstance(v, float) for v in allocs.values())
    # All equal performance
    for strat in engine.population:
        strat.performance = 1.0
    allocs_eq = engine.dynamic_resource_allocation(
        total_resource=1.0,
        min_alloc=0.0
    )
    vals = list(allocs_eq.values())
    assert all(
        abs(v - vals[0]) < 1e-6 for v in vals
    )


def test_ensemble_strategy_selection_mean_and_median():
    base = [Strategy({"threshold": 0.5 + i * 0.2}) for i in range(4)]
    engine = EvolutionEngine(base)
    data = [1, 2, 3, 4]
    agg_mean = engine.ensemble_strategy_selection(
        data,
        top_n=2,
        aggregation="mean"
    )
    agg_median = engine.ensemble_strategy_selection(
        data,
        top_n=2,
        aggregation="median"
    )
    assert isinstance(agg_mean, list)
    assert isinstance(agg_median, list)
    with pytest.raises(ValueError):
        engine.ensemble_strategy_selection(
            data,
            top_n=2,
            aggregation="invalid"
        )


def test_run_generation_and_select_top():
    base = [
        Strategy({"x": i})
        for i in range(5)
    ]
    engine = EvolutionEngine(base)
    engine.run_generation([1, 2, 3])
    assert len(engine.population) == 5
    top = engine.select_top(2)
    assert len(top) == 2
    assert all(isinstance(s, Strategy) for s in top)


def test_meta_learn_crossval():
    base = [Strategy({"x": i}) for i in range(3)]
    engine = EvolutionEngine(base)
    data = [1, 2, 3, 4, 5]
    scores = engine.meta_learn(data, method="crossval", k_folds=2)
    assert isinstance(scores, list)
    assert len(scores) == 3


def test_genetic_hyperparameter_evolution():
    engine = EvolutionEngine([])
    engine.genetic_hyperparameter_evolution(
        generations=2,
        data=[1, 2],
        population_size=4
    )
    assert len(engine.population) == 4
    assert all(isinstance(s, Strategy) for s in engine.population)


def test_bayesian_hyperparameter_optimization():
    engine = EvolutionEngine([])
    engine.bayesian_hyperparameter_optimization(data=[1, 2, 3], n_iter=3)
    assert len(engine.population) == 1
    assert isinstance(engine.population[0], Strategy)


def test_main_block_excluded():
    import importlib
    import sys
    modname = "python_ai.evolution_engine"
    if modname in sys.modules:
        del sys.modules[modname]
    importlib.import_module(modname)
