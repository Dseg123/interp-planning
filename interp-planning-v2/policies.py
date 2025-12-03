import numpy as np
from typing import Callable, Optional


class BasePolicy:
    """Base class for policies."""

    def __init__(self, env):
        self.env = env

    def get_action(self, state: np.ndarray, goal_embedding: np.ndarray) -> Optional[int]:
        """
        Get an action given current state and goal.

        Args:
            state: Current N-dimensional state
            goal: Goal N-dimensional state

        Returns:
            Action index, or None if no valid actions
        """
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """Random policy that selects uniformly from available actions."""

    def __init__(self, env, seed: Optional[int] = None):
        super().__init__(env)
        self.rng = np.random.RandomState(seed)

    def get_action(self, state: np.ndarray, goal_embedding: np.ndarray) -> Optional[int]:
        """Select a random valid action."""
        available_actions = self.env.get_available_actions(state)

        if len(available_actions) == 0:
            return None

        return self.rng.choice(available_actions)


class GreedyPolicy(BasePolicy):
    """
    Greedy policy based on learned embeddings.

    Uses softmax over distances in embedding space for exploration.
    """

    def __init__(self,
                 env,
                 s_encoder: Callable[[np.ndarray], np.ndarray],
                 temperature: float = 1.0,
                 seed: Optional[int] = None):
        """
        Initialize greedy policy.

        Args:
            env: Environment
            s_encoder: Function that maps state to embedding (s = A @ f(state))
            temperature: Softmax temperature (lower = more greedy)
            seed: Random seed
        """
        super().__init__(env)
        self.s_encoder = s_encoder
        self.temperature = temperature
        self.rng = np.random.RandomState(seed)

    def get_action(self, state: np.ndarray, goal_embedding: np.ndarray) -> Optional[int]:
        """
        Select action that moves toward goal in embedding space.

        Uses softmax over negative distances for stochastic exploration.
        """
        available_actions = self.env.get_available_actions(state)

        if len(available_actions) == 0:
            return None

        # Compute distances in embedding space for each action
        distances = []
        for action in available_actions:
            # Get next state
            delta = self.env._action_to_delta(action)
            next_state = state + delta

            # Compute distance to goal in embedding space
            next_embedding = self.s_encoder(next_state)
            dist = np.linalg.norm(next_embedding - goal_embedding)
            distances.append(dist)

        distances = np.array(distances)

        # Softmax over negative distances (lower distance = higher probability)
        logits = -distances / self.temperature
        # Numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)

        # Sample action
        action = self.rng.choice(available_actions, p=probabilities)

        return action


class ManhattanPolicy(BasePolicy):
    """
    Policy based on Manhattan distance to goal.

    Uses softmax over distances for exploration.
    """

    def __init__(self,
                 env,
                 temperature: float = 1.0,
                 seed: Optional[int] = None):
        """
        Initialize Manhattan policy.

        Args:
            env: Environment
            temperature: Softmax temperature (lower = more greedy)
            seed: Random seed
        """
        super().__init__(env)
        self.temperature = temperature
        self.rng = np.random.RandomState(seed)

    def get_action(self, state: np.ndarray, goal_embedding: np.ndarray) -> Optional[int]:
        """
        Select action that reduces Manhattan distance to goal.

        Uses softmax over negative distances for stochastic exploration.
        """
        available_actions = self.env.get_available_actions(state)

        if len(available_actions) == 0:
            return None

        # Compute Manhattan distances for each action
        distances = []
        for action in available_actions:
            # Get next state
            delta = self.env._action_to_delta(action)
            next_state = state + delta

            # Compute Manhattan distance to goal
            dist = self.env.get_distance(next_state, goal_embedding)
            distances.append(dist)

        distances = np.array(distances)

        # Softmax over negative distances
        logits = -distances / self.temperature
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)

        # Sample action
        action = self.rng.choice(available_actions, p=probabilities)

        return action
