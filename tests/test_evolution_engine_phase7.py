"""Phase 7 tests for EvolutionEngine performance-oriented features."""

from typing import Any, Dict, Optional

from python_ai.evolution_engine import EvolutionEngine, Strategy


class _DeterministicStrategy(Strategy):
    """Strategy with deterministic and fast evaluation for tests."""

    def __init__(self, params: Dict[str, Any], score: float) -> None:
        super().__init__(params)
        self._score = score

    def evaluate(self, data: Optional[Dict[str, Any]]) -> float:
        self.performance = self._score
        return self._score


def test_iter_population_and_mutations() -> None:
    base = [Strategy({"x": i}) for i in range(4)]
    engine = EvolutionEngine(base, evaluation_workers=2)
    assert len(list(engine.iter_population())) == 4
    mutations = list(engine.iter_mutations())
    assert len(mutations) == 4
    assert all(isinstance(s, Strategy) for s in mutations)


def test_evaluate_population_parallel_mode() -> None:
    base = [_DeterministicStrategy({"x": i}, score=float(i)) for i in range(5)]
    engine = EvolutionEngine(base, evaluation_workers=4)
    scores = engine.evaluate_population(data=None)
    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(s.performance is not None for s in engine.population)


def test_run_generation_uses_mutation_pipeline() -> None:
    base = [Strategy({"threshold": 0.5, "stop_loss": 0.1}) for _ in range(3)]
    engine = EvolutionEngine(base, evaluation_workers=2)
    before_ids = [id(s) for s in engine.population]
    engine.run_generation(data=None)
    after_ids = [id(s) for s in engine.population]
    assert len(after_ids) == 3
    assert before_ids != after_ids


def test_elo_tournament_selection_sets_normalized_performance() -> None:
    base = [
        _DeterministicStrategy({"x": 0}, score=0.1),
        _DeterministicStrategy({"x": 1}, score=0.4),
        _DeterministicStrategy({"x": 2}, score=0.9),
        _DeterministicStrategy({"x": 3}, score=0.2),
    ]
    engine = EvolutionEngine(base, evaluation_workers=2)
    scores = engine.elo_tournament_selection(
        data=None, rounds=8, k_factor=24.0
    )
    assert set(scores.keys()) == set(base)
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    assert all(s.performance is not None for s in base)


def test_meta_learn_crossval_handles_k_folds_greater_than_data() -> None:
    base = [_DeterministicStrategy({"x": 0}, score=0.5)]
    engine = EvolutionEngine(base)
    scores = engine.meta_learn(data=[1, 2], method="crossval", k_folds=10)
    assert scores is not None
    assert len(scores) == 1
    assert isinstance(scores[0], float)
