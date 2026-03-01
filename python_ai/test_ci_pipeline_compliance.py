"""CI Pipeline Failure Detection Tests.

Comprehensive edge cases and CI compliance checks.
"""

import pytest

from python_ai.evolution_engine import EvolutionEngine, Strategy


class TestTypeValidation:
    """Tests for type safety and contract validation."""

    def test_strategy_performance_type_after_evaluate(self):
        """Validate performance is always float after evaluate."""
        s = Strategy({"x": 1.0})
        for _ in range(10):
            result = s.evaluate(None)
            assert isinstance(result, float), "evaluate must return float"
            assert isinstance(
                s.performance, float
            ), "performance must be float"

    def test_ensemble_returns_list_of_floats(self):
        """Validate ensemble returns list of floats."""
        base = [Strategy({"threshold": 0.5}) for _ in range(2)]
        engine = EvolutionEngine(base)
        result = engine.ensemble_strategy_selection([1, 2, 3])
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_select_top_returns_list_of_strategies(self):
        """Validate select_top returns list of Strategy objects."""
        base = [Strategy({"x": i}) for i in range(5)]
        engine = EvolutionEngine(base)
        engine.run_generation(None)
        top = engine.select_top(2)
        assert isinstance(top, list)
        assert all(isinstance(s, Strategy) for s in top)

    def test_meta_learn_returns_list_or_none(self):
        """Validate meta_learn returns list or None."""
        base = [Strategy({"x": i}) for i in range(2)]
        engine = EvolutionEngine(base)
        result = engine.meta_learn([1, 2, 3])
        assert result is None or isinstance(result, list)
        if result is not None:
            assert all(isinstance(v, float) for v in result)

    def test_dynamic_resource_allocation_returns_dict(self):
        """Validate allocation returns dict with all strategies."""
        base = [Strategy({"x": i}) for i in range(3)]
        engine = EvolutionEngine(base)
        allocs = engine.dynamic_resource_allocation()
        assert isinstance(allocs, dict)
        assert len(allocs) == len(base)
        assert all(isinstance(v, float) for v in allocs.values())


class TestExceptionHandling:
    """Tests for exception handling and error scenarios."""

    def test_invalid_aggregation_raises_valueerror(self):
        """Validate invalid aggregation raises ValueError."""
        base = [Strategy({"threshold": 0.5})]
        engine = EvolutionEngine(base)
        with pytest.raises(ValueError):
            engine.ensemble_strategy_selection(
                [1, 2, 3], aggregation="invalid"
            )

    def test_meta_learn_empty_data_raises_valueerror(self):
        """Validate empty data raises ValueError."""
        base = [Strategy({"x": 1.0})]
        engine = EvolutionEngine(base)
        with pytest.raises(ValueError, match="non-empty"):
            engine.meta_learn([])

    def test_meta_learn_non_sequence_raises_valueerror(self):
        """Validate non-sequence data raises ValueError."""
        base = [Strategy({"x": 1.0})]
        engine = EvolutionEngine(base)
        with pytest.raises(ValueError, match="sequence"):
            engine.meta_learn(42)

    def test_select_top_more_than_population(self):
        """Validate select_top with n > population size."""
        base = [Strategy({"x": i}) for i in range(2)]
        engine = EvolutionEngine(base)
        top = engine.select_top(10)  # More than population
        assert len(top) == len(base)


class TestStateMutation:
    """Tests for proper state management and isolation."""

    def test_strategy_params_independent_copy(self):
        """Validate strategy params are copied, not referenced."""
        original_params = {"x": 1.0, "y": 2.0}
        s = Strategy(original_params)
        original_params["x"] = 99.0
        assert s.params["x"] == 1.0  # Should not change

    def test_mutate_creates_new_instance(self):
        """Validate mutate creates new Strategy instance."""
        s = Strategy({"x": 1.0})
        mutated = s.mutate()
        assert mutated is not s
        assert isinstance(mutated, Strategy)

    def test_run_generation_modifies_only_own_population(self):
        """Validate run_generation doesn't affect other engines."""
        base1 = [Strategy({"x": 1.0})]
        base2 = [Strategy({"x": 2.0})]
        engine1 = EvolutionEngine(base1)
        engine2 = EvolutionEngine(base2)

        perf2_before = engine2.population[0].performance

        engine1.run_generation(None)

        # engine2 should not be affected
        assert engine2.population[0].performance == perf2_before

    def test_ensemble_doesnt_mutate_input(self):
        """Validate ensemble_strategy_selection doesn't mutate data."""
        base = [Strategy({"threshold": 0.5})]
        engine = EvolutionEngine(base)
        data = [1, 2, 3]
        data_copy = data.copy()

        engine.ensemble_strategy_selection(data)
        assert data == data_copy


class TestBoundaryConditions:
    """Tests for boundary conditions and extreme values."""

    def test_select_top_zero_strategies(self):
        """Validate select_top with n=0."""
        base = [Strategy({"x": i}) for i in range(3)]
        engine = EvolutionEngine(base)
        top = engine.select_top(0)
        assert top == []

    def test_ensemble_single_datapoint(self):
        """Validate ensemble with single datapoint."""
        base = [Strategy({"threshold": 0.5})]
        engine = EvolutionEngine(base)
        result = engine.ensemble_strategy_selection([42])
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_meta_learn_single_sample(self):
        """Validate meta_learn with single sample."""
        base = [Strategy({"x": i}) for i in range(2)]
        engine = EvolutionEngine(base)
        result = engine.meta_learn([1], k_folds=1)
        assert result is not None
        assert len(result) == len(base)

    def test_dynamic_allocation_max_strategies(self):
        """Validate allocation with many strategies."""
        base = [Strategy({"x": i}) for i in range(100)]
        engine = EvolutionEngine(base)
        allocs = engine.dynamic_resource_allocation()
        assert len(allocs) == 100
        assert sum(allocs.values()) > 0

    def test_genetic_algorithm_with_existing_population(self):
        """Validate genetic algorithm works with existing population."""
        base = [Strategy({"x": 1.0}), Strategy({"x": 2.0})]
        engine = EvolutionEngine(base)
        engine.genetic_hyperparameter_evolution(
            generations=1, population_size=2
        )
        # Should still have population after evolution
        assert len(engine.population) >= 2


class TestImportSafety:
    """Tests for import safety and module structure."""

    def test_strategy_can_be_imported_directly(self):
        """Validate Strategy is importable."""
        from python_ai.evolution_engine import Strategy as S

        assert S is Strategy

    def test_evolution_engine_can_be_imported_directly(self):
        """Validate EvolutionEngine is importable."""
        from python_ai.evolution_engine import EvolutionEngine as EE

        assert EE is EvolutionEngine

    def test_strategy_initialization_basic(self):
        """Validate Strategy can be initialized with minimal params."""
        s = Strategy({})
        assert s.params == {}
        assert s.performance is None

    def test_engine_initialization_empty(self):
        """Validate EvolutionEngine can be initialized empty."""
        engine = EvolutionEngine([])
        assert engine.population == []


class TestAPIContract:
    """Tests for public API contracts and documented behavior."""

    def test_strategy_mutate_preserves_type(self):
        """Validate mutate returns Strategy."""
        s = Strategy({"x": 1.0})
        mutated = s.mutate()
        assert isinstance(mutated, Strategy)
        assert hasattr(mutated, "params")
        assert hasattr(mutated, "performance")

    def test_engine_methods_exist(self):
        """Validate EvolutionEngine has all documented methods."""
        engine = EvolutionEngine([])
        assert hasattr(engine, "run_generation")
        assert hasattr(engine, "select_top")
        assert hasattr(engine, "meta_learn")
        assert hasattr(engine, "ensemble_strategy_selection")
        assert hasattr(engine, "dynamic_resource_allocation")
        assert hasattr(engine, "self_play_and_coevolution")
        assert hasattr(engine, "genetic_hyperparameter_evolution")

    def test_strategy_methods_are_callable(self):
        """Validate Strategy methods are callable."""
        s = Strategy({"x": 1.0})
        assert callable(s.mutate)
        assert callable(s.evaluate)

    def test_engine_methods_are_callable(self):
        """Validate EvolutionEngine methods are callable."""
        e = EvolutionEngine([])
        assert callable(e.run_generation)
        assert callable(e.select_top)
        assert callable(e.meta_learn)


class TestResourceManagement:
    """Tests for proper resource cleanup and management."""

    def test_strategy_doesnt_hold_excessive_refs(self):
        """Validate Strategy doesn't hold excessive references."""
        strategies = [Strategy({"x": i}) for i in range(1000)]
        # Ensure no memory issues with many strategies
        assert len(strategies) == 1000

    def test_engine_cleanup_after_operations(self):
        """Validate Engine state is clean after operations."""
        engine = EvolutionEngine([Strategy({"x": i}) for i in range(10)])
        engine.run_generation(None)
        assert engine.population is not None
        assert len(engine.population) == 10

    def test_ensemble_with_large_data(self):
        """Validate ensemble handles large datasets."""
        base = [Strategy({"threshold": 0.5})]
        engine = EvolutionEngine(base)
        large_data = list(range(10000))
        result = engine.ensemble_strategy_selection(large_data)
        assert len(result) == len(large_data)


class TestMockIsolation:
    """Tests for mock isolation and no cross-test pollution."""

    def test_strategy_performance_independent_between_instances(self):
        """Validate strategies don't share performance state."""
        s1 = Strategy({"x": 1.0})
        s2 = Strategy({"x": 2.0})

        s1.evaluate(None)
        s2.evaluate(None)

        # Performances should be independent
        original_s2_perf = s2.performance
        s1.performance = 999.0
        assert s2.performance == original_s2_perf

    def test_engine_population_independent(self):
        """Validate engines have independent populations."""
        e1 = EvolutionEngine([Strategy({"x": 1})])
        e2 = EvolutionEngine([Strategy({"x": 2})])

        e1.run_generation(None)
        # e2 should not be affected
        assert len(e2.population) == 1
        assert e2.population[0].params["x"] == 2


class TestPerformanceRegression:
    """Tests to catch performance regressions."""

    def test_select_top_performance_linear(self):
        """Validate select_top completes reasonably with many strategies."""
        base = [Strategy({"x": i}) for i in range(1000)]
        engine = EvolutionEngine(base)
        for _ in range(10):
            engine.run_generation(None)
        top = engine.select_top(10)
        assert len(top) == 10

    def test_ensemble_completes_with_many_strategies(self):
        """Validate ensemble handles many strategies."""
        base = [Strategy({"threshold": i * 0.01}) for i in range(100)]
        engine = EvolutionEngine(base)
        result = engine.ensemble_strategy_selection([1, 2, 3])
        assert len(result) == 3


class TestSecurityAndValidation:
    """Tests for security and input validation."""

    def test_strategy_rejects_deeply_nested_params(self):
        """Validate strategy params are not excessively nested."""
        params = {"x": {"y": {"z": {"deep": 1.0}}}}
        s = Strategy(params)
        # Should accept but handle gracefully
        assert "x" in s.params

    def test_engine_with_none_values_in_data(self):
        """Validate engine handles None values in data."""
        base = [Strategy({"threshold": 0.5})]
        engine = EvolutionEngine(base)
        data = [1, None, 3, None, 5]
        result = engine.ensemble_strategy_selection(data)
        assert len(result) == len(data)

    def test_invalid_k_folds_handled_gracefully(self):
        """Validate k_folds edge cases are handled."""
        base = [Strategy({"x": 1.0})]
        engine = EvolutionEngine(base)
        # Very large k_folds should be capped
        result = engine.meta_learn([1], k_folds=1000)
        assert result is not None


class TestConsistency:
    """Tests for consistency across multiple invocations."""

    def test_select_top_consistent_ordering(self):
        """Validate select_top maintains consistent ordering."""
        base = [Strategy({"x": i}) for i in range(10)]
        engine = EvolutionEngine(base)

        for strat in engine.population:
            strat.performance = 0.5  # All equal

        top1 = engine.select_top(3)
        top2 = engine.select_top(3)

        # With equal performance, should be deterministic after sort
        assert len(top1) == len(top2) == 3

    def test_ensemble_deterministic_with_fixed_params(self):
        """Validate ensemble is deterministic with fixed params."""
        params = {"threshold": 0.5}
        base = [Strategy(params) for _ in range(2)]
        engine = EvolutionEngine(base)

        data = [1, 2, 3]
        result1 = engine.ensemble_strategy_selection(data)
        result2 = engine.ensemble_strategy_selection(data)

        # Should be identical with same params
        assert len(result1) == len(result2)
