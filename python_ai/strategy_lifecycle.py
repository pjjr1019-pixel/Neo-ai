"""Strategy lifecycle management for evolutionary trading strategies."""

from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Optional

from python_ai.evolution_engine import Strategy


@dataclass
class StrategyLineage:
    """Lineage metadata for one strategy instance."""

    strategy_id: str
    strategy: Strategy
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    created_at: float = field(default_factory=time.time)
    retired_at: Optional[float] = None
    retired_reason: Optional[str] = None
    fitness_history: List[float] = field(default_factory=list)


class StrategyLifecycleManager:
    """Track active/retired strategies, lineage, and warm-start candidates."""

    def __init__(self) -> None:
        """Initialize active and archived strategy registries."""
        self._counter = count(1)
        self.active: Dict[str, StrategyLineage] = {}
        self.archive: Dict[str, StrategyLineage] = {}

    def register(
        self,
        strategy: Strategy,
        *,
        parent_ids: Optional[List[str]] = None,
        generation: int = 0,
    ) -> str:
        """Register a strategy and return its new lifecycle ID."""
        strategy_id = f"strat_{next(self._counter):06d}"
        self.active[strategy_id] = StrategyLineage(
            strategy_id=strategy_id,
            strategy=strategy,
            parent_ids=list(parent_ids or []),
            generation=generation,
        )
        return strategy_id

    def record_fitness(self, strategy_id: str, fitness: float) -> None:
        """Append fitness value to a strategy history."""
        node = self.active.get(strategy_id) or self.archive.get(strategy_id)
        if node is None:
            raise KeyError(f"Unknown strategy_id: {strategy_id}")
        node.fitness_history.append(float(fitness))

    def retire(self, strategy_id: str, reason: str = "replaced") -> None:
        """Move strategy from active pool into museum/archive."""
        node = self.active.pop(strategy_id, None)
        if node is None:
            raise KeyError(f"Unknown active strategy_id: {strategy_id}")
        node.retired_at = time.time()
        node.retired_reason = reason
        self.archive[strategy_id] = node

    def family_tree(self, strategy_id: str) -> List[str]:
        """Return ancestor chain IDs for a strategy."""
        seen: List[str] = []
        queue = deque([strategy_id])
        while queue:
            sid = queue.popleft()
            node = self.active.get(sid) or self.archive.get(sid)
            if node is None:
                continue
            for parent_id in node.parent_ids:
                if parent_id not in seen:
                    seen.append(parent_id)
                    queue.append(parent_id)
        return seen

    def complexity_penalty(
        self,
        strategy: Strategy,
        weight: float = 0.01,
    ) -> float:
        """Compute a penalty favoring simpler parameter sets."""
        complexity = 0.0
        for value in strategy.params.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                complexity += abs(float(value)) * 0.01
            elif isinstance(value, (list, tuple, dict)):
                complexity += 1.0
            else:
                complexity += 0.5
        complexity += len(strategy.params) * 0.1
        return weight * complexity

    def age_adjusted_fitness(
        self,
        strategy_id: str,
        *,
        decay: float = 0.98,
    ) -> float:
        """Return latest fitness adjusted by strategy age decay."""
        node = self.active.get(strategy_id) or self.archive.get(strategy_id)
        if node is None:
            raise KeyError(f"Unknown strategy_id: {strategy_id}")
        if not node.fitness_history:
            return 0.0
        age = max(0, int(time.time() - node.created_at))
        return node.fitness_history[-1] * (decay**age)

    def warm_start_population(self, top_n: int = 5) -> List[Strategy]:
        """Return top historical strategies for warm-start seeding."""
        all_nodes = list(self.active.values()) + list(self.archive.values())

        def score(node: StrategyLineage) -> float:
            """Return latest fitness for ranking warm-start candidates."""
            return node.fitness_history[-1] if node.fitness_history else 0.0

        ranked = sorted(all_nodes, key=score, reverse=True)
        return [
            copy.deepcopy(node.strategy) for node in ranked[: max(0, top_n)]
        ]
