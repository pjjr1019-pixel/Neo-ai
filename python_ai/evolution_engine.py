"""
Evolution Engine for Strategy Mutation and Meta-Learning
- Mutates strategies (thresholds, sizing, stop-loss)
- Supports meta-learning (MAML, Reptile)
- Designed for extensibility and compliance with project coding policy
- Wired to BacktestingEngine for real performance evaluation
"""

import copy
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import optuna

from python_ai.backtesting_engine import get_backtesting_engine

logger = logging.getLogger(__name__)


class Strategy:
    """Strategy for evolutionary algorithm with parameters and performance."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize a Strategy with parameters.
        Args:
            params: Dictionary of strategy parameters.
        """
        self.params: Dict[str, Any] = params.copy()
        self.performance: Optional[float] = None

    def mutate(self) -> "Strategy":
        """
        Mutate strategy parameters slightly.
        Returns:
            New mutated Strategy instance.
        """
        new_params: Dict[str, Any] = self.params.copy()
        # Example mutation: tweak thresholds slightly
        for k, v in new_params.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                new_params[k] += random.uniform(-0.1, 0.1) * v  # nosec: B311
        return Strategy(new_params)

    def evaluate(self, data: Optional[Dict[str, Any]]) -> float:
        """Evaluate strategy performance using backtesting engine.

        Args:
            data: Dict with 'ohlcv_data' (prices) and 'signals' (trading
                 signals).

        Returns:
            Fitness score from backtest (0-1 scale).
        """
        if not isinstance(data, dict):
            self.performance = 0.0
            return self.performance

        ohlcv_data = data.get("ohlcv_data", {})
        signals = data.get("signals", [])

        if not ohlcv_data or not signals:
            self.performance = 0.0
            return self.performance

        try:
            backtest_engine = get_backtesting_engine()
            metrics = backtest_engine.run_backtest(
                ohlcv_data,
                signals,
            )
            self.performance = metrics.fitness_score
        except Exception:
            logger.warning(
                "Strategy evaluation failed for params=%s",
                self.params,
                exc_info=True,
            )
            self.performance = 0.0

        return self.performance


class EvolutionEngine:
    """Engine for running evolutionary optimization of strategies."""

    def __init__(
        self,
        base_strategies: Iterable[Strategy],
        evaluation_workers: int = 1,
    ) -> None:
        """Initialize with a base population.

        Args:
            base_strategies: Initial population of strategies.
            evaluation_workers: Number of worker threads used for
                parallel strategy evaluation. ``1`` means sequential.
        """
        self.base_strategies: List[Strategy] = list(base_strategies)
        self.population: List[Strategy] = list(self.base_strategies)
        self.evaluation_workers = max(1, evaluation_workers)

    def iter_population(self) -> Iterator[Strategy]:
        """Yield strategies lazily for large populations."""
        yield from self.population

    def iter_mutations(self) -> Iterator[Strategy]:
        """Yield mutated strategies lazily."""
        for strat in self.population:
            yield strat.mutate()

    @staticmethod
    def _evaluate_strategy(
        payload: Tuple[Strategy, Optional[Dict[str, Any]]],
    ) -> float:
        """Evaluate one strategy payload."""
        strat, data = payload
        return strat.evaluate(data)

    def evaluate_population(
        self,
        data: Optional[Dict[str, Any]],
        workers: Optional[int] = None,
    ) -> List[float]:
        """Evaluate full population, optionally in parallel."""
        if not self.population:
            return []

        n_workers = max(1, workers or self.evaluation_workers)
        if n_workers == 1 or len(self.population) < 2:
            return [s.evaluate(data) for s in self.population]

        payloads = [(s, data) for s in self.population]
        with ThreadPoolExecutor(
            max_workers=min(n_workers, len(self.population))
        ) as pool:
            return list(pool.map(self._evaluate_strategy, payloads))

    def explainable_evolution_report(
        self, previous_population: Optional[List[Strategy]] = None
    ) -> str:
        """
        Generate a human-readable report of the evolution process and
        strategy changes.

        Args:
            previous_population: Optional list of previous strategies
                for comparison.

        Returns:
            String report summarizing evolution, performance, and
            parameter changes.
        """
        lines = []
        lines.append("Evolution Report\n====================")
        for i, strat in enumerate(self.population):
            lines.append(f"Strategy {i + 1}:")
            params_str = f"  Params: {strat.params}"
            if len(params_str) > 79:
                params_str = params_str[:76] + "..."
            lines.append(params_str)
            perf_str: str = (
                f"  Performance: {strat.performance:.4f}"
                if strat.performance is not None
                else "  Performance: N/A"
            )
            if len(perf_str.rstrip()) > 79:
                perf_str = perf_str[:76] + "..."
            lines.append(perf_str)
            if previous_population and i < len(previous_population):
                prev = previous_population[i]
                param_changes = {
                    k: (prev.params.get(k), strat.params.get(k))
                    for k in strat.params
                }
                lines.append("  Param Changes:")
                for k, (old, new) in param_changes.items():
                    if old != new:
                        lines.append(f"    {k}: {old} -> {new}")
            lines.append("")
        return "\n".join(lines)

    def self_play_and_coevolution(
        self, data: Optional[Dict[str, Any]], rounds: int = 5
    ) -> Dict[Strategy, float]:
        """
        Simulate self-play and co-evolution between strategies.

        Each strategy is evaluated against others in a
        round-robin tournament.

        Args:
            data: Input data for evaluation.
            rounds: Number of self-play rounds.

        Returns:
            Dict mapping strategy to average score.
        """
        n: int = len(self.population)
        # Need at least 2 strategies for coevolution
        if n < 2:
            # Assign zero performance to single/empty population
            for strat in self.population:
                strat.performance = 0.0
            return {strat: 0.0 for strat in self.population}

        # Memoize evaluations: each strategy evaluated once
        eval_cache: Dict[int, float] = {}
        for strat in self.population:
            sid = id(strat)
            if sid not in eval_cache:
                eval_cache[sid] = strat.evaluate(data)

        scores: Dict[Strategy, float] = {
            strat: 0.0 for strat in self.population
        }
        for _ in range(rounds):
            for i in range(n - 1):
                strat_a = self.population[i]
                perf_a = eval_cache[id(strat_a)]
                for j in range(i + 1, n):
                    strat_b = self.population[j]
                    perf_b = eval_cache[id(strat_b)]
                    if perf_a > perf_b:
                        scores[strat_a] += 1.0
                    elif perf_b > perf_a:
                        scores[strat_b] += 1.0
                    else:
                        scores[strat_a] += 0.5
                        scores[strat_b] += 0.5
        # Normalize by per-strategy match count.
        matches_per_strategy = max(1, rounds * (n - 1))
        avg_scores: Dict[Strategy, float] = {
            strat: score / matches_per_strategy
            for strat, score in scores.items()
            if strat is not None
        }
        # Assign average score as performance
        for strat, avg in avg_scores.items():
            strat.performance = avg
        return avg_scores

    def elo_tournament_selection(
        self,
        data: Optional[Dict[str, Any]],
        rounds: int = 10,
        k_factor: float = 32.0,
    ) -> Dict[Strategy, float]:
        """Run Elo-style tournaments with sub-quadratic match scheduling.

        This replaces full round-robin O(n^2) pairings with random adjacent
        pairings O(n) per round while still applying competitive pressure.

        Args:
            data: Input data for strategy evaluation.
            rounds: Number of tournament rounds to run.
            k_factor: Elo rating update factor.

        Returns:
            Dict mapping each strategy to normalized Elo score in [0, 1].
        """
        n = len(self.population)
        if n < 2:
            for strat in self.population:
                strat.performance = 0.0
            return {strat: 0.0 for strat in self.population}

        eval_cache: Dict[int, float] = {
            id(strat): strat.evaluate(data) for strat in self.population
        }
        ratings: Dict[Strategy, float] = {
            strat: 1500.0 for strat in self.population
        }
        n_rounds = max(1, rounds)

        for _ in range(n_rounds):
            shuffled = list(self.population)
            random.shuffle(shuffled)  # nosec: B311
            for idx in range(0, len(shuffled) - 1, 2):
                strat_a = shuffled[idx]
                strat_b = shuffled[idx + 1]

                rating_a = ratings[strat_a]
                rating_b = ratings[strat_b]
                expected_a = 1.0 / (
                    1.0 + 10 ** ((rating_b - rating_a) / 400.0)
                )
                expected_b = 1.0 - expected_a

                perf_a = eval_cache[id(strat_a)]
                perf_b = eval_cache[id(strat_b)]
                if perf_a > perf_b:
                    score_a, score_b = 1.0, 0.0
                elif perf_b > perf_a:
                    score_a, score_b = 0.0, 1.0
                else:
                    score_a, score_b = 0.5, 0.5

                ratings[strat_a] = rating_a + k_factor * (score_a - expected_a)
                ratings[strat_b] = rating_b + k_factor * (score_b - expected_b)

        min_rating = min(ratings.values())
        max_rating = max(ratings.values())
        denom = max(1e-12, max_rating - min_rating)
        normalized: Dict[Strategy, float] = {
            strat: (rating - min_rating) / denom
            for strat, rating in ratings.items()
        }
        for strat, score in normalized.items():
            strat.performance = score
        return normalized

    def dynamic_resource_allocation(
        self, total_resource: float = 1.0, min_alloc: float = 0.0
    ) -> Dict[Strategy, float]:
        """
        Dynamically allocate resources to strategies based on
        their performance.

        Args:
            total_resource: Total resource to allocate
                (e.g., capital, CPU time).
            min_alloc: Minimum allocation per strategy.

        Returns:
            Dict mapping strategy to allocated resource.
        """
        # Gather performances, avoid None
        if not self.population:
            return {}
        performances: List[float] = [
            s.performance if s.performance is not None else 0.0
            for s in self.population
        ]
        min_perf: float = min(performances)
        # Shift performances to be non-negative
        shifted: List[float] = [p - min_perf for p in performances]
        total: float | int = sum(shifted)
        n: int = len(self.population)
        allocs = {}
        if total == 0:
            # Equal allocation if all performances are the same
            if total_resource > n * min_alloc:
                equal_share = total_resource / n
            else:
                equal_share = min_alloc
            for strat in self.population:
                allocs[strat] = equal_share
        else:
            for strat, perf in zip(self.population, shifted):
                if total_resource > n * min_alloc:
                    alloc: float = min_alloc + (
                        (total_resource - n * min_alloc) * (perf / total)
                    )
                else:
                    alloc = min_alloc
                allocs[strat] = alloc
        return allocs

    def ensemble_strategy_selection(
        self, data: Any, top_n: int = 3, aggregation: str = "mean"
    ) -> List[float]:
        """
        Select an ensemble of top-N strategies and aggregate
        their predictions.

        Args:
            data: Input data to evaluate strategies.
            top_n: Number of top strategies to include.
            aggregation: Aggregation method ('mean' or 'median').

        Returns:
            Aggregated prediction for each data point.
        """
        # Evaluate all strategies on the data (parallel when enabled).
        eval_data: Optional[Dict[str, Any]] = (
            data if isinstance(data, dict) else None
        )
        self.evaluate_population(eval_data)
        # Select top-N strategies by performance
        top_strats: List[Strategy] = self.select_top(top_n)
        # Each strategy makes a prediction for each data point
        predictions = []
        for strat in top_strats:
            # For demonstration, use a dummy prediction:
            # param threshold * data value
            preds: List[Any] = [
                strat.params.get("threshold", 1.0)
                * (x if isinstance(x, (int, float)) else 1.0)
                for x in data
            ]
            predictions.append(preds)
        # Aggregate predictions
        import numpy as np

        pred_matrix = np.array(predictions)
        if aggregation == "mean":
            agg = np.mean(pred_matrix, axis=0)
        elif aggregation == "median":
            agg = np.median(pred_matrix, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        return list(agg.tolist())

    def run_generation(self, data: Optional[Dict[str, Any]]) -> None:
        """
        Mutate and evaluate all strategies in the population.
        Args:
            data: Input data for evaluation.
        """
        # Mutate lazily, then evaluate (parallel when configured).
        self.population = list(self.iter_mutations())
        self.evaluate_population(data)

    def select_top(self, n: int) -> List[Strategy]:
        """
        Select top-n strategies by performance.
        Args:
            n: Number of top strategies to select.
        Returns:
            List of top Strategy objects.
        """
        # Select top-n strategies by performance
        return sorted(
            self.population, key=lambda s: s.performance or 0, reverse=True
        )[:n]

    def meta_learn(
        self, data: Any, method: str = "crossval", k_folds: int = 5
    ) -> Optional[List[float]]:
        """Meta-learning with cross-validation support.

        Args:
            data: Input data for meta-learning.
            method: Meta-learning method ('crossval', etc.).
            k_folds: Number of folds for cross-validation.

        Returns:
            List of average scores or None.
        """
        if method == "crossval":
            if not data or not hasattr(data, "__len__"):
                raise ValueError(
                    "Data must be a non-empty sequence for "
                    "cross-validation."
                )
            samples: List[Any] = list(data)
            n: int = len(samples)
            actual_k_folds: int = min(k_folds, n)
            results = []
            base_fold_size: int = n // actual_k_folds
            remainder: int = n % actual_k_folds
            start = 0
            for i in range(actual_k_folds):
                fold_size: int = base_fold_size + (1 if i < remainder else 0)
                end = start + fold_size
                val_data = samples[start:end]
                start = end
                fold_scores = []
                for strat in self.population:
                    val_eval_data: Dict[str, Any] = {
                        "ohlcv_data": {"close": val_data},
                        "signals": ["HOLD"] * len(val_data),
                    }
                    val_score: float = strat.evaluate(val_eval_data)
                    fold_scores.append(val_score)
                results.append(fold_scores)
            avg_scores: List[float] = [
                sum(scores) / actual_k_folds for scores in zip(*results)
            ]
            for strat, avg in zip(self.population, avg_scores):
                strat.performance = avg
            return avg_scores
        else:
            return None

    def genetic_hyperparameter_evolution(
        self,
        generations: int,
        data: Optional[Dict[str, Any]] = None,
        population_size: int = 10,
        mutation_rate: float = 0.2,
    ) -> None:
        """Evolve hyperparameters using a simple genetic algorithm.

        Args:
            generations: Number of generations to run.
            data: Input data for evaluation.
            population_size: Size of population.
            mutation_rate: Probability of mutation.
        """
        population: List[Strategy] = [
            Strategy(
                {
                    "threshold": random.uniform(0, 1),  # nosec: B311
                    "stop_loss": random.uniform(0, 0.5),  # nosec: B311
                }
            )
            for _ in range(population_size)
        ]
        for _ in range(generations):
            for strat in population:
                strat.evaluate(data)
            population = sorted(
                population, key=lambda s: s.performance or 0, reverse=True
            )
            survivors: List[Strategy] = population[: population_size // 2]
            offspring = []
            for parent in survivors:
                child = (
                    parent.mutate()
                    if random.random() < mutation_rate  # nosec: B311
                    else copy.deepcopy(parent)
                )
                offspring.append(child)
            population = survivors + offspring
        self.population = population

    def bayesian_hyperparameter_optimization(
        self, data: Optional[Dict[str, Any]] = None, n_iter: int = 10
    ) -> None:
        """Bayesian optimization of strategy hyperparameters via Optuna TPE.

        Uses Tree-structured Parzen Estimator (TPE) to efficiently
        search the strategy parameter space instead of random sampling.

        Args:
            data: Input data for evaluation (passed to Strategy.evaluate).
            n_iter: Number of Optuna trials to run.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            """Single Optuna trial: sample params → evaluate."""
            params: Dict[str, float] = {
                "threshold": trial.suggest_float(
                    "threshold",
                    0.0,
                    1.0,
                ),
                "stop_loss": trial.suggest_float(
                    "stop_loss",
                    0.01,
                    0.5,
                ),
            }
            strat = Strategy(params)
            return strat.evaluate(data)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_iter)

        best_params = study.best_params
        best_strat = Strategy(best_params)
        best_strat.evaluate(data)
        self.population = [best_strat]

        logger.info(
            "Bayesian optimization complete: best_value=%.4f  "
            "params=%s  trials=%d",
            study.best_value,
            best_params,
            len(study.trials),
        )


if __name__ == "__main__":  # pragma: no cover
    base: List[Strategy] = [
        Strategy({"threshold": 0.5, "stop_loss": 0.1}) for _ in range(5)
    ]
    engine = EvolutionEngine(base)
    for _ in range(3):
        engine.run_generation(data=None)
        top: List[Strategy] = engine.select_top(2)
        logger.info(
            "Top strategies: %s",
            [s.params for s in top],
        )
