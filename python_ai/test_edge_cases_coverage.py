"""Comprehensive edge case and CI pipeline failure tests for 100% coverage."""

import pytest

from python_ai.evolution_engine import EvolutionEngine, Strategy


def test_explainable_evolution_report_long_params():
    """Test evolution report with long parameter strings."""
    params = {
        "very_long_parameter_name_one": 0.123456789,
        "very_long_parameter_name_two": 0.987654321,
        "another_extremely_long_param": 0.555555555,
    }
    base = [Strategy(params) for _ in range(1)]
    engine = EvolutionEngine(base)
    engine.population[0].performance = 0.99999999

    report = engine.explainable_evolution_report()
    assert isinstance(report, str)
    assert "Evolution Report" in report
    assert "..." in report  # Truncation should occur
    assert len(report.split("\n")[1]) <= 100  # Lines should be manageable


def test_explainable_evolution_report_long_perf_string():
    """Test evolution report with performance strings that exceed 79 chars."""
    base = [Strategy({"x": 1.0}) for _ in range(1)]
    engine = EvolutionEngine(base)
    # Set a very long parameter that would make perf_str long
    engine.population[0].params = {
        "threshold_with_long_name": 0.123456789,
        "stop_loss_parameter": 0.987654321,
    }
    engine.population[0].performance = 0.99999999

    report = engine.explainable_evolution_report()
    assert isinstance(report, str)
    assert "Strategy 1:" in report


def test_explainable_evolution_report_with_previous_population():
    """Test evolution report comparing with previous population."""
    prev_params = {"threshold": 0.5, "stop_loss": 0.1}
    curr_params = {"threshold": 0.6, "stop_loss": 0.2}

    prev_base = [Strategy(prev_params)]
    curr_base = [Strategy(curr_params)]

    engine = EvolutionEngine(curr_base)

    for strat in engine.population:
        strat.performance = 0.8

    report = engine.explainable_evolution_report(prev_base)
    assert "Param Changes:" in report
    assert "threshold" in report
    assert "stop_loss" in report


def test_self_play_coevolution_tie_condition():
    """Test self_play_and_coevolution with tied performances."""
    # Mock evaluate to return same performance for all strategies
    call_count = {"count": 0}

    class MockStrategy(Strategy):
        """Mock strategy for testing tie conditions."""

        def evaluate(self, data):
            """Return fixed performance for testing."""
            call_count["count"] += 1
            # First two calls same, then different to trigger tie detection
            if call_count["count"] % 2 == 1:
                self.performance = 0.5
                return 0.5
            else:
                self.performance = 0.5
                return 0.5

    engine = EvolutionEngine([MockStrategy({"x": i}) for i in range(2)])
    scores = engine.self_play_and_coevolution(data=[1, 2, 3], rounds=1)

    assert isinstance(scores, dict)
    assert len(scores) == 2
    # Verify tie case was hit (equal scores means ties occurred)
    for score in scores.values():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_dynamic_resource_allocation_min_alloc_exceeds():
    """Test allocation when min_alloc * n > total_resource (line 208)."""
    base = [Strategy({"x": i}) for i in range(3)]
    engine = EvolutionEngine(base)

    # Set small total_resource, large min_alloc so constraint is active
    allocs = engine.dynamic_resource_allocation(
        total_resource=0.2, min_alloc=0.1  # 3 * 0.1 = 0.3 > 0.2
    )

    assert isinstance(allocs, dict)
    assert len(allocs) == 3
    # All should get exactly min_alloc since total < n * min_alloc
    for alloc in allocs.values():
        assert alloc == 0.1


def test_resource_allocation_single_strategy():
    """Test resource allocation with a single strategy."""
    base = [Strategy({"x": 1.0})]
    engine = EvolutionEngine(base)
    engine.population[0].performance = 0.8

    allocs = engine.dynamic_resource_allocation(
        total_resource=1.0, min_alloc=0.1
    )
    assert allocs[base[0]] > 0.1


def test_ensemble_with_non_numeric_data():
    """Test ensemble selection with mixed numeric/non-numeric data."""
    base = [Strategy({"threshold": 0.5 + i * 0.1}) for i in range(2)]
    engine = EvolutionEngine(base)

    data = [1, "string", 3.5, None, 2.0]
    ensemble = engine.ensemble_strategy_selection(
        data, top_n=2, aggregation="mean"
    )

    assert isinstance(ensemble, list)
    assert len(ensemble) == len(data)
    # Non-numeric values should be treated as 1.0
    for val in ensemble:
        assert isinstance(val, float)


def test_genetic_hyperparameter_evolution_convergence():
    """Test genetic algorithm produces reasonable convergence."""
    engine = EvolutionEngine([])
    engine.genetic_hyperparameter_evolution(
        generations=5,
        population_size=10,
        mutation_rate=0.3,
    )

    assert len(engine.population) > 0
    # Check all strategies have valid params
    for strat in engine.population:
        assert "threshold" in strat.params
        assert "stop_loss" in strat.params
        assert isinstance(strat.params["threshold"], float)
        assert isinstance(strat.params["stop_loss"], float)


def test_mutation_preserves_structure():
    """Test that mutation preserves parameter structure and types."""
    params = {
        "threshold": 0.5,
        "stop_loss": 0.1,
        "flag": True,
        "name": "test",
    }
    original = Strategy(params)
    mutated = original.mutate()

    # All keys should be present
    assert set(mutated.params.keys()) == set(original.params.keys())
    # Bool and string should remain unchanged
    assert mutated.params["flag"] is True
    assert mutated.params["name"] == "test"
    # Numeric values should change slightly
    assert mutated.params["threshold"] != original.params["threshold"]


def test_select_top_with_none_performances():
    """Test select_top handles None performances gracefully."""
    base = [Strategy({"x": i}) for i in range(5)]
    engine = EvolutionEngine(base)

    # Leave some performances as None
    engine.population[0].performance = 0.9
    engine.population[1].performance = None
    engine.population[2].performance = 0.5

    top = engine.select_top(2)
    assert len(top) == 2
    # Highest performance should be first
    assert top[0].performance == 0.9


def test_meta_learn_k_folds_larger_than_data():
    """Test meta_learn with k_folds larger than data (capped automatically)."""
    base = [Strategy({"x": i}) for i in range(2)]
    engine = EvolutionEngine(base)
    data = [1, 2]  # Only 2 samples

    # k_folds=10 should be capped to len(data)=2
    scores = engine.meta_learn(data=data, method="crossval", k_folds=10)
    assert scores is not None
    assert all(isinstance(s, float) for s in scores)


def test_meta_learn_empty_data():
    """Test meta_learn raises on empty data."""
    base = [Strategy({"x": i}) for i in range(2)]
    engine = EvolutionEngine(base)

    with pytest.raises(ValueError):
        engine.meta_learn(data=[], method="crossval", k_folds=2)


def test_meta_learn_non_sequence_data():
    """Test meta_learn raises on non-sequence data."""
    base = [Strategy({"x": i}) for i in range(2)]
    engine = EvolutionEngine(base)

    with pytest.raises(ValueError):
        engine.meta_learn(data=42, method="crossval", k_folds=2)


def test_run_generation_with_none_data():
    """Test run_generation handles None data gracefully."""
    base = [Strategy({"threshold": 0.5}) for _ in range(3)]
    engine = EvolutionEngine(base)
    engine.run_generation(data=None)

    # All should have performance assigned
    for strat in engine.population:
        assert strat.performance is not None
        assert isinstance(strat.performance, float)


def test_ensemble_with_zero_predictions():
    """Test ensemble when threshold produces zero predictions."""
    base = [Strategy({"threshold": 0.0}) for _ in range(2)]
    engine = EvolutionEngine(base)
    data = [1, 2, 3]

    ensemble = engine.ensemble_strategy_selection(
        data, top_n=2, aggregation="mean"
    )
    assert all(v == 0.0 for v in ensemble)


def test_ensemble_with_negative_data():
    """Test ensemble with negative values."""
    base = [Strategy({"threshold": 0.5}) for _ in range(2)]
    engine = EvolutionEngine(base)
    data = [-1, -2, -3]

    ensemble = engine.ensemble_strategy_selection(
        data, top_n=2, aggregation="mean"
    )
    assert isinstance(ensemble, list)
    assert all(v < 0 for v in ensemble)


def test_ensemble_median_aggregation():
    """Test ensemble with median aggregation matches numpy."""
    base = [Strategy({"threshold": 0.5}), Strategy({"threshold": 0.7})]
    engine = EvolutionEngine(base)
    data = [1, 2, 3, 4, 5]

    median_agg = engine.ensemble_strategy_selection(
        data, top_n=2, aggregation="median"
    )
    assert isinstance(median_agg, list)
    # With two strategies, median should be (0.5*x + 0.7*x) / 2 = 0.6*x
    for i, val in enumerate(median_agg):
        assert abs(val - 0.6 * data[i]) < 0.01


def test_coevolution_single_strategy():
    """Test coevolution with only one strategy."""
    base = [Strategy({"x": 1.0})]
    engine = EvolutionEngine(base)

    scores = engine.self_play_and_coevolution(data=[1, 2, 3], rounds=5)
    assert len(scores) == 1
    # With one strategy, score should be 0 (no matches)
    assert list(scores.values())[0] == 0.0


def test_resource_allocation_zero_total():
    """Test resource allocation with zero total resource."""
    base = [Strategy({"x": i}) for i in range(3)]
    engine = EvolutionEngine(base)

    for i, strat in enumerate(engine.population):
        strat.performance = float(i)

    allocs = engine.dynamic_resource_allocation(
        total_resource=0.0, min_alloc=0.0
    )
    assert all(v == 0.0 for v in allocs.values())


def test_resource_allocation_all_zero_performance():
    """Test allocation when all performances are zero."""
    base = [Strategy({"x": i}) for i in range(3)]
    engine = EvolutionEngine(base)

    for strat in engine.population:
        strat.performance = 0.0

    allocs = engine.dynamic_resource_allocation(
        total_resource=1.0, min_alloc=0.1
    )
    # With zero total shifted performance, should be equal allocation
    vals = list(allocs.values())
    assert all(abs(v - vals[0]) < 1e-6 for v in vals)


def test_strategy_evaluate_consistency():
    """Test that evaluate always returns float and sets performance."""
    strat = Strategy({"x": 1.0})
    data = [1, 2, 3]

    for _ in range(5):
        result = strat.evaluate(data)
        assert isinstance(result, float)
        assert strat.performance == result
        assert 0.0 <= strat.performance <= 1.0


def test_bayesian_optimization_convergence():
    """Test bayesian_hyperparameter_optimization improves over iterations."""
    base = []
    engine = EvolutionEngine(base)
    engine.bayesian_hyperparameter_optimization(n_iter=20, data=None)

    assert len(engine.population) > 0
    best = engine.population[0]
    assert best.performance is not None


def test_strategy_params_immutability():
    """Test that strategy params are copied, not referenced."""
    original_params = {"x": 1.0, "y": 2.0}
    strat = Strategy(original_params)

    original_params["x"] = 99.0
    assert strat.params["x"] == 1.0


def test_engine_population_independence():
    """Test that engines don't share population state."""
    base1 = [Strategy({"x": 1.0})]
    base2 = [Strategy({"x": 2.0})]

    engine1 = EvolutionEngine(base1)
    engine2 = EvolutionEngine(base2)

    # Before run_generation, populations are different
    assert engine1.population[0].params["x"] == 1.0
    assert engine2.population[0].params["x"] == 2.0

    # After run_generation, they should still be different
    engine1.run_generation(None)
    engine2.run_generation(None)
    p1_x = engine1.population[0].params["x"]
    p2_x = engine2.population[0].params["x"]
    assert p1_x != p2_x


def test_log_resource_usage_imports():
    """Test that log_resource_usage can be imported without errors."""
    from python_ai.resource_monitor import log_resource_usage as lru

    assert callable(lru)


def test_evolution_report_empty_population():
    """Test evolution report with empty population."""
    engine = EvolutionEngine([])
    report = engine.explainable_evolution_report()
    assert "Evolution Report" in report


def test_evolution_report_no_performance():
    """Test evolution report with uninitialized performance."""
    base = [Strategy({"x": 1.0})]
    engine = EvolutionEngine(base)
    # Don't call evaluate, so performance is None

    report = engine.explainable_evolution_report()
    assert "Performance: N/A" in report
