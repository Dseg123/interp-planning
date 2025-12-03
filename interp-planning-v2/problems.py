import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, List, Set
from scipy.spatial.distance import cdist


class GridworldProblem(gym.Env):
    """
    N-dimensional gridworld environment with obstacles.

    State space: N-dimensional grid with coordinates in [0, K-1]^N
    Total state space size: K^N

    Obstacles: O random points are sampled. Any state within radius r of an
    obstacle is unreachable.

    Action space: 2N discrete actions (increment or decrement each dimension)
    """

    def __init__(self, K: int, N: int, O: int, r: float, seed: Optional[int] = None):
        """
        Initialize the gridworld problem.

        Args:
            K: Number of discrete values per dimension (coordinates range [0, K-1])
            N: Number of dimensions
            O: Number of obstacles
            r: Radius around obstacles that are unreachable
            seed: Random seed for reproducibility
        """
        super(GridworldProblem, self).__init__()

        self.K = K
        self.N = N
        self.O = O
        self.r = r

        self.rng = np.random.RandomState(seed)

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(2 * N)  # 2N actions: +/- each dimension
        self.observation_space = gym.spaces.Box(
            low=0, high=K-1, shape=(N,), dtype=np.int32
        )

        # Generate obstacles
        self.obstacles = self._generate_obstacles()

        # Initialize state and goal
        self.current_state = None
        self.goal_state = None

    def _generate_obstacles(self) -> np.ndarray:
        """
        Generate O random obstacle locations in the state space.

        Returns:
            Array of shape (O, N) containing obstacle coordinates
        """
        obstacles = self.rng.randint(0, self.K, size=(self.O, self.N))
        return obstacles

    def _is_valid_state(self, state: np.ndarray) -> bool:
        """
        Check if a state is valid (within bounds and not near obstacles).

        Args:
            state: N-dimensional state vector

        Returns:
            True if state is valid, False otherwise
        """
        # Check bounds
        if np.any(state < 0) or np.any(state >= self.K):
            return False

        # Check distance to obstacles
        if self.O > 0:
            distances = np.linalg.norm(self.obstacles - state, axis=1)
            if np.any(distances <= self.r):
                return False

        return True

    def _action_to_delta(self, action: int) -> np.ndarray:
        """
        Convert action index to state delta.

        Action encoding:
        - action in [0, N): increment dimension (action)
        - action in [N, 2N): decrement dimension (action - N)

        Args:
            action: Action index in [0, 2N)

        Returns:
            Delta vector to add to current state
        """
        delta = np.zeros(self.N, dtype=np.int32)

        if action < self.N:
            # Increment dimension 'action'
            delta[action] = 1
        else:
            # Decrement dimension 'action - N'
            delta[action - self.N] = -1

        return delta

    def can_initiate(self, action: int, state: Optional[np.ndarray] = None) -> bool:
        """
        Check if an action can be initiated from a given state.

        An action is initiable if the resulting state would be:
        1. Within bounds [0, K-1]^N
        2. Not within radius r of any obstacle

        Args:
            action: Action index
            state: State to check from (default: current_state)

        Returns:
            True if action is initiable, False otherwise
        """
        if state is None:
            state = self.current_state

        delta = self._action_to_delta(action)
        next_state = state + delta

        return self._is_valid_state(next_state)

    def get_available_actions(self, state: Optional[np.ndarray] = None) -> List[int]:
        """
        Get list of available (initiable) actions from a state.

        Args:
            state: State to check from (default: current_state)

        Returns:
            List of valid action indices
        """
        if state is None:
            state = self.current_state

        available = []
        for action in range(2 * self.N):
            if self.can_initiate(action, state):
                available.append(action)

        return available

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_state is None:
            raise RuntimeError("Must call reset() before step()")

        # Check if action is initiable
        if not self.can_initiate(action):
            # Invalid action - stay in current state with penalty
            return self.current_state.copy(), -1.0, False, {'valid': False}

        # Execute action
        delta = self._action_to_delta(action)
        self.current_state = self.current_state + delta

        # Check if goal reached
        done = np.array_equal(self.current_state, self.goal_state)
        reward = 1.0 if done else -0.01  # Small step penalty

        return self.current_state.copy(), reward, done, {'valid': True}

    def reset(self,
              state: Optional[np.ndarray] = None,
              goal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment to a new start and goal state.

        Uses rejection sampling to ensure start and goal are valid states
        (not near obstacles).

        Args:
            state: Starting state (if None, randomly sample)
            goal: Goal state (if None, randomly sample)

        Returns:
            Tuple of (current_state, goal_state)
        """
        # Sample valid start state
        if state is None:
            max_attempts = 1000
            for _ in range(max_attempts):
                state = self.rng.randint(0, self.K, size=self.N)
                if self._is_valid_state(state):
                    break
            else:
                raise RuntimeError(f"Could not find valid start state after {max_attempts} attempts")
        else:
            state = np.array(state, dtype=np.int32)
            if not self._is_valid_state(state):
                raise ValueError("Provided start state is invalid")

        # Sample valid goal state (different from start)
        if goal is None:
            max_attempts = 1000
            for _ in range(max_attempts):
                goal = self.rng.randint(0, self.K, size=self.N)
                if self._is_valid_state(goal) and not np.array_equal(goal, state):
                    break
            else:
                raise RuntimeError(f"Could not find valid goal state after {max_attempts} attempts")
        else:
            goal = np.array(goal, dtype=np.int32)
            if not self._is_valid_state(goal):
                raise ValueError("Provided goal state is invalid")

        self.current_state = state.copy()
        self.goal_state = goal.copy()

        return self.current_state.copy(), self.goal_state.copy()

    def state_to_index(self, state: np.ndarray) -> int:
        """
        Convert N-dimensional state to a unique integer index.

        Uses base-K encoding: index = sum(state[i] * K^i for i in range(N))

        Args:
            state: N-dimensional state vector

        Returns:
            Integer index in [0, K^N)
        """
        index = 0
        for i in range(self.N):
            index += state[i] * (self.K ** i)
        return index

    def index_to_state(self, index: int) -> np.ndarray:
        """
        Convert integer index to N-dimensional state.

        Inverse of state_to_index.

        Args:
            index: Integer index in [0, K^N)

        Returns:
            N-dimensional state vector
        """
        state = np.zeros(self.N, dtype=np.int32)
        for i in range(self.N):
            state[i] = (index // (self.K ** i)) % self.K
        return state

    def get_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Get Manhattan distance between two states.

        Args:
            state1: First state
            state2: Second state

        Returns:
            Manhattan (L1) distance
        """
        return np.sum(np.abs(state1 - state2))
