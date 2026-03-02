from python_ai import evolution_engine


def test_strategy_init():
    """Test initialization of Strategy with given parameters."""
    params = {"param1": 1, "param2": 2}
    s = evolution_engine.Strategy(params)
    assert s.params == params


def test_evolution_engine_init():
    """Test initialization of EvolutionEngine with base strategies."""
    base_strategies = [evolution_engine.Strategy({"p": i}) for i in range(3)]
    engine = evolution_engine.EvolutionEngine(base_strategies)
    assert engine.base_strategies == base_strategies


def test_explainable_evolution_report():
    """Test explainable_evolution_report returns a string."""
    base_strategies = [evolution_engine.Strategy({"p": i}) for i in range(2)]
    engine = evolution_engine.EvolutionEngine(base_strategies)
    report = engine.explainable_evolution_report()
    assert isinstance(report, str)
    assert "Evolution Report" in report or len(report) > 0
