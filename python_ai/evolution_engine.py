"""
Evolution Engine for Strategy Mutation and Meta-Learning
- Mutates strategies (thresholds, sizing, stop-loss)
- Supports meta-learning (MAML, Reptile)
- Designed for extensibility and compliance with project coding policy
"""

import random
import copy
from typing import List, Dict, Any


class Strategy:
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize a Strategy with parameters.
        Args:
            params: Dictionary of strategy parameters.
        """
        self.params: Dict[str, Any] = params.copy()
        self.performance = None

    def mutate(self) -> "Strategy":
        """
        Mutate strategy parameters slightly.
        Returns:
            New mutated Strategy instance.
        """
        new_params: Dict[str, Any] = self.params.copy()
        # Example mutation: tweak thresholds slightly
        for k, v in new_params.items():
            if isinstance(v, (int, float)):
                new_params[k] += random.uniform(-0.1, 0.1) * v
        return Strategy(new_params)

    def evaluate(self, data) -> float:
        """
        Evaluate strategy performance (stub).
        Args:
            data: Input data for evaluation.
        Returns:
            Random float as performance.
        """
        # Placeholder: implement backtesting logic
        self.performance: float = random.uniform(0, 1)
        return self.performance


class EvolutionEngine:
    def explainable_evolution_report(self, previous_population=None) -> str:
        """
        Generate a human-readable report of the evolution process and
        strategy changes.
        Args:
            previous_population: Optional list of previous strategies for
            comparison.
        Returns:
            String report summarizing evolution, performance, and parameter
            changes.
        """
        """
        Generate a human-readable report of the evolution process and
        strategy changes.
        Args:
            previous_population: Optional list of previous strategies for
            comparison.
        Returns:
            String report summarizing evolution, performance, and parameter
            changes.
        """
        lines = []
        lines.append("Evolution Report\n====================")
        for i, strat in enumerate(self.population):
            lines.append(f"Strategy {i+1}:")
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
                    k: (
                        prev.params.get(k),
                        strat.params.get(k),
                    )
                    for k in strat.params
                }
                lines.append("  Param Changes:")
                for k, (old, new) in param_changes.items():
                    if old != new:
                        lines.append(f"    {k}: {old} -> {new}")
            lines.append("")
        return "\n".join(lines)

    def self_play_and_coevolution(
        self, data, rounds: int = 5
    ) -> Dict[Strategy | None, float]:
        """
        Simulate self-play and co-evolution between strategies.
        Args:
            data: Input data for evaluation.
            rounds: Number of self-play rounds.
        Returns:
            Dict mapping strategy to average score.
        """
        """
        Simulate self-play and co-evolution between strategies.
        Each strategy is evaluated against others in a round-robin tournament.
        Args:
            data: Input data for evaluation.
            rounds: Number of self-play rounds.
        Returns:
            Dict mapping strategy to average score.
        """
        n: int = len(self.population)
        scores: Dict[Strategy | None, float] = {strat: 0.0 for strat in self.population}
        for _ in range(rounds):
            for i, strat_a in enumerate(self.population):
                for j, strat_b in enumerate(self.population):
                    if i == j:
                        continue
                    # Simulate a match:
                    # higher performance wins (random if equal)
                    perf_a: float = strat_a.evaluate(data)
                    perf_b: float = strat_b.evaluate(data)
                    if perf_a > perf_b:
                        scores[strat_a] += 1
                    elif perf_b > perf_a:
                        scores[strat_b] += 1
                    else:
                        # Tie: both get half a point
                        scores[strat_a] += 0.5
                        scores[strat_b] += 0.5
        # Normalize scores by number of matches
        matches: int = rounds * n * (n - 1)
        avg_scores: Dict[Strategy | None, float] = {
            strat: score / matches for strat, score in scores.items()
        }
        # Assign average score as performance
        for strat, avg in avg_scores.items():
            strat.performance = avg
        return avg_scores

    def dynamic_resource_allocation(
        self, total_resource: float = 1.0, min_alloc: float = 0.0
    ):
        """
        Dynamically allocate resources to strategies based on their
        performance.
        Args:
            total_resource: Total resource to allocate (e.g., capital,
            CPU time).
            min_alloc: Minimum allocation per strategy.
        Returns:
            Dict mapping strategy to allocated resource.
        """
        """
        Dynamically allocate resources to strategies based on their
        performance.
        Args:
            total_resource: Total resource to allocate (e.g., capital,
            CPU time).
            min_alloc: Minimum allocation per strategy.
        Returns:
            Dict mapping strategy to allocated resource.
        """
        # Gather performances, avoid None
        performances: List[float] = [
            s.performance if s.performance is not None else 0.0 for s in self.population
        ]
        min_perf: float = min(performances)
        # Shift performances to be non-negative
        shifted: List[float] = [p - min_perf for p in performances]
        total: float | int = sum(shifted)
        n: int = len(self.population)
        allocs = {}
        if total == 0:
            # Equal allocation if all performances are the same
            for strat in self.population:
                allocs[strat] = total_resource / n
        else:
            for strat, perf in zip(self.population, shifted):
                if total_resource > n * min_alloc:
                    alloc: float = min_alloc + (
                        (total_resource - n * min_alloc) * (perf / total)
                    )
                else:
                    alloc: float = min_alloc
                allocs[strat] = alloc
        return allocs

    def ensemble_strategy_selection(
        self, data, top_n: int = 3, aggregation: str = "mean"
    ):
        """
        Selects an ensemble of top-N strategies and aggregates their
        predictions.
        Args:
            data: Input data to evaluate strategies.
            top_n: Number of top strategies to include in the ensemble.
            aggregation: Aggregation method ('mean' or 'median').
        Returns:
            Aggregated prediction for each data point.
        """
        """
        Selects an ensemble of top-N strategies and aggregates their
        predictions.
        Args:
            data: Input data to evaluate strategies.
            top_n: Number of top strategies to include in the ensemble.
            aggregation: Aggregation method ('mean' or 'median').
        Returns:
            Aggregated prediction for each data point.
        """
        # Evaluate all strategies on the data
        for strat in self.population:
            strat.evaluate(data)
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

        predictions = np.array(predictions)
        if aggregation == "mean":
            agg = np.mean(predictions, axis=0)
        elif aggregation == "median":
            agg = np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        return agg.tolist()

    def __init__(self, base_strategies: List[Strategy]) -> None:
        """
        Initialize the EvolutionEngine with a base population of strategies.
        Args:
            base_strategies: List of Strategy objects to start the population.
        """
        self.population: List[Strategy] = base_strategies

    def run_generation(self, data) -> None:
        """
        Mutate and evaluate all strategies in the population.
        Args:
            data: Input data for evaluation.
        """
        # Mutate and evaluate all strategies
        new_population = []
        for strat in self.population:
            mutated: Strategy = strat.mutate()
            mutated.evaluate(data)
            new_population.append(mutated)
        self.population = new_population

    def select_top(self, n: int) -> List[Strategy]:
        """
        Select top-n strategies by performance.
        Args:
            n: Number of top strategies to select.
        Returns:
            List of top Strategy objects.
        """
        # Select top-n strategies by performance
        return sorted(self.population, key=lambda s: s.performance or 0, reverse=True)[
            :n
        ]

    def meta_learn(
        self, data, method: str = "crossval", k_folds: int = 5
    ) -> List[float] | None:
        """
        Meta-learning with cross-validation support.
        Args:
            data: Input data for meta-learning.
            method: Meta-learning method ('crossval', etc.).
            k_folds: Number of folds for cross-validation.
        Returns:
            List of average scores or None.
        """
        """
        Meta-learning with cross-validation support.
        If method == 'crossval', performs k-fold cross-validation on
        current population.
        """
        if method == "crossval":
            if not data or not hasattr(data, "__len__"):
                raise ValueError(
                    "Data must be a non-empty sequence for cross-validation."
                )
            n: int = len(data)
            fold_size: int = max(1, n // k_folds)
            results = []
            for i in range(k_folds):
                start: int = i * fold_size
                end: int = start + fold_size if i < k_folds - 1 else n
                val_idx: List[int] = list(range(start, end))
                train_idx: List[int] = [j for j in range(n) if j not in val_idx]
                train_data = [data[j] for j in train_idx]
                val_data = [data[j] for j in val_idx]
                fold_scores = []
                for strat in self.population:
                    # Train: here just evaluate on train_data for placeholder
                    strat.evaluate(train_data)
                    # Validate: evaluate on val_data
                    val_score: float = strat.evaluate(val_data)
                    fold_scores.append(val_score)
                results.append(fold_scores)
            # Average scores per strategy
            avg_scores: List[float] = [
                sum(scores) / k_folds for scores in zip(*results)
            ]
            # Assign average performance to each strategy
            for strat, avg in zip(self.population, avg_scores):
                strat.performance = avg
            return avg_scores
        else:
            # Placeholder for other meta-learning methods (MAML, Reptile, etc.)
            pass

    def genetic_hyperparameter_evolution(
        self,
        generations: int,
        data=None,
        population_size: int = 10,
        mutation_rate: float = 0.2,
    ) -> None:
        """
        Evolve hyperparameters using a simple genetic algorithm.
        Args:
            generations: Number of generations to run.
            data: Input data for evaluation.
            population_size: Size of population.
            mutation_rate: Probability of mutation.
        """
        """
        Evolve hyperparameters using a simple genetic algorithm.
        """
        population: List[Strategy] = [
            Strategy(
                {
                    "threshold": random.uniform(0, 1),
                    "stop_loss": random.uniform(0, 0.5),
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
                    if random.random() < mutation_rate
                    else copy.deepcopy(parent)
                )
                offspring.append(child)
            population = survivors + offspring
        self.population = population

    def bayesian_hyperparameter_optimization(self, data=None, n_iter: int = 10) -> None:
        """
        Stub for Bayesian optimization of hyperparameters.
        Args:
            data: Input data for evaluation.
            n_iter: Number of iterations.
        """
        """
        Stub for Bayesian optimization of hyperparameters
        (replace with e.g. skopt or optuna for real use).
        """
        best = None
        best_perf = float("-inf")
        for _ in range(n_iter):
            params: Dict[str, float] = {
                "threshold": random.uniform(0, 1),
                "stop_loss": random.uniform(0, 0.5),
            }
            strat = Strategy(params)
            perf: float = strat.evaluate(data)
            if perf > best_perf:
                best: Strategy = strat
                best_perf: float = perf
        self.population: List[None | Strategy] = [best]

    # Example usage
    if __name__ == "__main__":
        from python_ai.evolution_engine import EvolutionEngine

        base: List[Strategy] = [
            Strategy({"threshold": 0.5, "stop_loss": 0.1}) for _ in range(5)
        ]
        engine = EvolutionEngine(base)
        for _ in range(3):
            engine.run_generation(data=None)
            top: List[Strategy] = engine.select_top(2)
            print(top)
            print(f"Top strategies: {[s.params for s in top]}")
