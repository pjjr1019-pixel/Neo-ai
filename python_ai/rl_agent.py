"""
Reinforcement Learning Trading Agent for NEO Hybrid AI.

Implements a Deep Q-Network (DQN) agent with experience
replay and epsilon-greedy exploration for learning
trading policies directly from market features.
"""

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Actions the agent can take.
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
NUM_ACTIONS = 3


@dataclass
class Transition:
    """A single experience tuple.

    Attributes:
        state: Observation before the action.
        action: Integer action taken.
        reward: Scalar reward received.
        next_state: Observation after the action.
        done: Whether the episode ended.
    """

    state: np.ndarray  # type: ignore[type-arg]
    action: int
    reward: float
    next_state: np.ndarray  # type: ignore[type-arg]
    done: bool


class _DQN(nn.Module):
    """Two-hidden-layer Q-network."""

    def __init__(self, state_dim: int, n_actions: int = 3) -> None:
        """Build the network.

        Args:
            state_dim: Dimensionality of the state.
            n_actions: Number of discrete actions.
        """
        super().__init__()
        mid = max(state_dim * 2, 32)
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.Linear(mid, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action.

        Args:
            x: State tensor ``(batch, state_dim)``.

        Returns:
            Q-values ``(batch, n_actions)``.
        """
        return self.net(x)  # type: ignore[no-any-return]


class TradingEnvironment:
    """Simple trading environment over a price series.

    Supports a single asset with discrete actions
    (hold, buy, sell).  Position is binary (flat or
    long).

    Args:
        prices: Chronological price array.
        features: Optional feature matrix aligned
            with *prices*.  If ``None``, raw price
            returns are used as state.
        initial_balance: Starting cash balance.
        transaction_cost: Proportional trading fee.
    """

    def __init__(
        self,
        prices: np.ndarray,  # type: ignore[type-arg]
        features: Optional[np.ndarray] = None,  # type: ignore[type-arg]
        initial_balance: float = 10_000.0,
        transaction_cost: float = 0.001,
    ) -> None:
        """Initialise the environment."""
        self._prices = np.asarray(prices, dtype=float)
        self._features = (
            np.asarray(features, dtype=np.float32)
            if features is not None
            else None
        )
        self._init_balance = initial_balance
        self._tc = transaction_cost
        self._step = 0
        self._position = 0  # 0 = flat, 1 = long
        self._balance = initial_balance
        self._entry_price = 0.0

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state vector."""
        if self._features is not None:
            return self._features.shape[1] + 1  # type: ignore[no-any-return]
        return 2  # return + position

    def reset(self) -> np.ndarray:  # type: ignore[type-arg]
        """Reset the environment.

        Returns:
            Initial state vector.
        """
        self._step = 1  # need at least 1 prior price
        self._position = 0
        self._balance = self._init_balance
        self._entry_price = 0.0
        return self._get_state()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool]:  # type: ignore[type-arg]
        """Execute one step.

        Args:
            action: ``0`` = hold, ``1`` = buy, ``2`` = sell.

        Returns:
            Tuple ``(next_state, reward, done)``.
        """
        price = self._prices[self._step]
        reward = 0.0

        if action == ACTION_BUY and self._position == 0:
            self._position = 1
            self._entry_price = price
            self._balance -= price * self._tc
        elif action == ACTION_SELL and self._position == 1:
            pnl = (price - self._entry_price) / self._entry_price
            reward = pnl - self._tc
            self._position = 0
            self._balance *= 1.0 + reward

        self._step += 1
        done = self._step >= len(self._prices) - 1
        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:  # type: ignore[type-arg]
        """Build the current state vector."""
        if self._features is not None:
            feat = self._features[self._step]
            return np.concatenate([feat, [float(self._position)]])
        ret = self._prices[self._step] / self._prices[self._step - 1] - 1.0
        return np.array(
            [ret, float(self._position)],
            dtype=np.float32,
        )


class DQNAgent:
    """Deep Q-Network trading agent.

    Uses an epsilon-greedy policy with decaying
    exploration, experience replay, and a target
    network for stability.

    Args:
        state_dim: Size of the state vector.
        n_actions: Discrete action count.
        gamma: Discount factor.
        lr: Learning rate.
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Multiplicative decay per episode.
        buffer_size: Replay buffer capacity.
        batch_size: Minibatch size for learning.
        target_update: Episodes between target-net syncs.
    """

    def __init__(
        self,
        state_dim: int = 2,
        n_actions: int = NUM_ACTIONS,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update: int = 10,
    ) -> None:
        """Initialise the agent."""
        self._state_dim = state_dim
        self._n_actions = n_actions
        self._gamma = gamma
        self._epsilon = epsilon_start
        self._eps_end = epsilon_end
        self._eps_decay = epsilon_decay
        self._batch_size = batch_size
        self._target_update = target_update

        self._policy_net = _DQN(state_dim, n_actions)
        self._target_net = _DQN(state_dim, n_actions)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._optimiser = torch.optim.Adam(
            self._policy_net.parameters(), lr=lr
        )
        self._buffer: Deque[Transition] = deque(maxlen=buffer_size)
        self._episode_rewards: List[float] = []
        self._train_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(
        self,
        state: np.ndarray,  # type: ignore[type-arg]
        explore: bool = True,
    ) -> int:
        """Choose an action for the given state.

        Args:
            state: Current state vector.
            explore: Use epsilon-greedy if ``True``.

        Returns:
            Integer action.
        """
        if explore and random.random() < self._epsilon:
            return random.randint(0, self._n_actions - 1)
        tensor = torch.from_numpy(
            np.asarray(state, dtype=np.float32)
        ).unsqueeze(0)
        self._policy_net.eval()
        with torch.no_grad():
            q = self._policy_net(tensor)
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        env: TradingEnvironment,
        episodes: int = 100,
    ) -> Dict[str, Any]:
        """Train the agent on *env* for *episodes*.

        Args:
            env: Trading environment instance.
            episodes: Number of training episodes.

        Returns:
            Dict with ``episode_rewards``,
            ``mean_reward``, ``epsilon``.
        """
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.act(state, explore=True)
                next_state, reward, done = env.step(action)
                self._buffer.append(
                    Transition(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    )
                )
                state = next_state
                total_reward += reward
                self._learn_step()

            self._episode_rewards.append(total_reward)
            self._epsilon = max(
                self._eps_end,
                self._epsilon * self._eps_decay,
            )
            if (ep + 1) % self._target_update == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())

        mean_r = float(np.mean(self._episode_rewards[-episodes:]))
        logger.info(
            "DQN trained %d episodes, mean_reward=%.4f, " "epsilon=%.4f",
            episodes,
            mean_r,
            self._epsilon,
        )
        return {
            "episode_rewards": list(self._episode_rewards[-episodes:]),
            "mean_reward": mean_r,
            "epsilon": self._epsilon,
        }

    def _learn_step(self) -> None:
        """Sample a minibatch and perform one SGD step."""
        if len(self._buffer) < self._batch_size:
            return

        batch = random.sample(list(self._buffer), self._batch_size)
        states = torch.from_numpy(
            np.array([t.state for t in batch], dtype=np.float32)
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long
        ).unsqueeze(1)
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32
        ).unsqueeze(1)
        next_states = torch.from_numpy(
            np.array(
                [t.next_state for t in batch],
                dtype=np.float32,
            )
        )
        dones = torch.tensor(
            [float(t.done) for t in batch],
            dtype=torch.float32,
        ).unsqueeze(1)

        self._policy_net.train()
        q_vals = self._policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = (
                self._target_net(next_states).max(dim=1, keepdim=True).values
            )
            target = rewards + self._gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_vals, target)
        self._optimiser.zero_grad()
        loss.backward()
        self._optimiser.step()
        self._train_steps += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return agent summary.

        Returns:
            Dict with config, training stats, epsilon.
        """
        return {
            "state_dim": self._state_dim,
            "n_actions": self._n_actions,
            "epsilon": self._epsilon,
            "buffer_size": len(self._buffer),
            "train_steps": self._train_steps,
            "episodes": len(self._episode_rewards),
            "mean_reward": (
                float(np.mean(self._episode_rewards))
                if self._episode_rewards
                else 0.0
            ),
        }
