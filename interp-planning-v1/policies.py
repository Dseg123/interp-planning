import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable
from problems import GraphProblem


class BasePolicy(ABC):
    """
    Base interface for policies that select actions in a GraphProblem environment.
    """

    def __init__(self, problem: GraphProblem):
        """
        Initialize the policy with a GraphProblem.

        Args:
            problem: The GraphProblem environment
        """
        self.problem = problem

    @abstractmethod
    def get_action(self, state: int, goal: np.ndarray, debug: bool = False) -> int:
        """
        Select an action given the current state.

        Args:
            state: The current state (vertex)

        Returns:
            The selected action (target vertex)
        """
        pass


class RandomPolicy(BasePolicy):
    """
    A policy that randomly selects from available actions at each state.
    """

    def __init__(self, problem: GraphProblem, seed: Optional[int] = None):
        """
        Initialize the random policy.

        Args:
            problem: The GraphProblem environment
            seed: Random seed for reproducibility
        """
        super().__init__(problem)
        self.rng = np.random.RandomState(seed)

    def get_action(self, state: int, goal: np.ndarray, debug: bool = False) -> int:
        """
        Randomly select an action from available actions at the given state.

        Args:
            state: The current state (vertex)

        Returns:
            A randomly selected neighboring vertex
        """
        available_actions = self.problem.get_available_actions(state)

        if not available_actions:
            raise ValueError(f"No available actions from state {state}")

        return self.rng.choice(available_actions)


class GreedyPolicy(BasePolicy):
    """
    A policy that selects actions using softmax over distances to a goal in latent space.
    This provides stochastic exploration while biasing toward closer states.
    """

    def __init__(self, problem: GraphProblem, s_encoder: Callable[[int], np.ndarray],
                 temperature: float = 1.0, seed: Optional[int] = None):
        """
        Initialize the greedy policy with a state encoder.

        Args:
            problem: The GraphProblem environment
            s_encoder: Function that maps state (int) to latent embedding (np.ndarray)
            temperature: Temperature for softmax (higher = more exploration)
            seed: Random seed for reproducibility
        """
        super().__init__(problem)
        self.s_encoder = s_encoder
        self.temperature = temperature
        self.rng = np.random.RandomState(seed)

    def get_action(self, state: int, goal: np.ndarray, debug: bool = False) -> int:
        """
        Select an action using softmax over negative distances to the goal in latent space.

        Args:
            state: The current state (vertex)
            goal: The goal embedding in latent space (np.ndarray)

        Returns:
            A stochastically sampled action (neighboring vertex) biased toward closer states
        """
        available_actions = self.problem.get_available_actions(state)

        if not available_actions:
            raise ValueError(f"No available actions from state {state}")

        # Compute L2 distances for all available actions
        distances = []
        for action in available_actions:
            action_encoding = self.s_encoder(action)
            # distance = np.linalg.norm(action_encoding - goal)
            distance = -np.dot(action_encoding, goal)
            distances.append(distance)

        distances = np.array(distances)

        # Convert distances to negative (so closer = higher probability)
        # Apply softmax with temperature
        logits = -distances / self.temperature

        # Softmax: exp(logits) / sum(exp(logits))
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)

        # Sample action according to probabilities
        action = self.rng.choice(available_actions, p=probabilities)
        if debug:
            print(f"State: {state}, Available actions: {available_actions}")
            print(f"Distances: {distances}")
            print(f"Logits: {logits}")
            print(f"Probabilities: {probabilities}")
            print(f"Selected action: {action}")
            
        return action
