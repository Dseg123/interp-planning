import numpy as np
from typing import Callable, Tuple, Optional
from problems import GraphProblem
from policies import BasePolicy
from waypoints import c_waypoint, i_waypoint


class WaypointPlanner:
    """
    A planner that maintains internal state and takes one step at a time toward waypoints.
    """

    def __init__(
        self,
        start: int,
        goal: int,
        env: GraphProblem,
        policy: BasePolicy,
        psi: Callable[[int], np.ndarray],
        A: np.ndarray,
        waypoint_type: str = 'i',
        max_waypoints: int = 100,
        M: int = 50,
        c: float = 1.0,
        eps: float = 1e-3,
        T: float = 1.0
    ):
        """
        Initialize the waypoint planner.

        Args:
            start: Starting state
            goal: Goal state
            env: GraphProblem environment
            policy: Policy to select actions
            psi: State encoder function
            A: Transformation matrix
            waypoint_type: 'c' for c-waypoint or 'i' for i-waypoint
            max_waypoints: Maximum number of waypoints before going directly to goal
            M: Number of samples for c-waypoint
            c: Parameter for i-waypoint
            eps: Distance threshold to consider waypoint reached
            T: Temperature for c-waypoint
        """
        self.start = start
        self.goal = goal
        self.env = env
        self.policy = policy
        self.psi = psi
        self.A = A
        self.waypoint_type = waypoint_type
        self.max_waypoints = max_waypoints
        self.M = M
        self.c = c
        self.eps = eps
        self.T = T

        # Internal state
        self.current_state = start
        self.num_waypoints_used = 0
        self.current_waypoint = None
        self.done = False
        self.trajectory = [start]  # Track trajectory for future state sampling

        # Reset environment and compute initial waypoint
        self.env.reset(start, goal)
        self._update_waypoint()

    def _update_waypoint(self):
        """Update the current waypoint based on current state and goal."""
        if self.num_waypoints_used < self.max_waypoints:
            # Generate new waypoint
            if self.waypoint_type == 'c':
                self.current_waypoint = c_waypoint(
                    self.current_state, self.goal, self.psi, self.A,
                    list(self.env.graph.nodes()), self.M, self.T
                )
            elif self.waypoint_type == 'i':
                self.current_waypoint = i_waypoint(
                    self.current_state, self.goal, self.psi, self.A, self.c
                )
            else:
                raise ValueError("Invalid waypoint type. Must be 'c' or 'i'.")
        else:
            # Max waypoints reached, target goal directly
            self.current_waypoint = self.psi(self.goal)

    def step(self, debug: bool = False) -> Tuple[Optional[int], Optional[np.ndarray], bool]:
        """
        Take one step toward the current waypoint.

        Returns:
            Tuple of (action, waypoint, done):
                - action: The action taken (next state)
                - waypoint: The waypoint we were targeting
                - done: Whether we've reached the goal
        """
        if self.done:
            return None, None, True

        # Check if we've reached the goal
        if self.current_state == self.goal:
            self.done = True
            return None, None, True

        # Get current waypoint
        waypoint = self.current_waypoint

        # Select action using policy
        action = self.policy.get_action(self.current_state, waypoint, debug=debug)

        # Take step in environment
        next_state, valid = self.env.step(action)

        # Update current state and trajectory
        self.current_state = next_state
        self.trajectory.append(next_state)

        # Check if we've reached the current waypoint
        curr_z = self.A @ self.psi(self.current_state)
        if np.linalg.norm(curr_z - waypoint) < self.eps:
            # Waypoint reached, update to next waypoint
            self.num_waypoints_used += 1
            self._update_waypoint()

        # Check if done
        if self.current_state == self.goal:
            self.done = True

        if debug:
            print(f"Current state: {self.current_state}, Waypoint: {waypoint}, Action taken: {action}, Done: {self.done}")

        return action, waypoint, self.done

    def reset(self, start: Optional[int] = None, goal: Optional[int] = None):
        """
        Reset the planner with new start and goal.

        Args:
            start: New starting state (if None, use original start)
            goal: New goal state (if None, use original goal)
        """
        if start is not None:
            self.start = start
        if goal is not None:
            self.goal = goal

        self.current_state = self.start
        self.num_waypoints_used = 0
        self.current_waypoint = None
        self.done = False
        self.trajectory = [self.start]  # Reset trajectory

        self.env.reset(self.start, self.goal)
        self._update_waypoint()

    def sample_future_state(self, current_idx: int, p: float = 0.9) -> Optional[int]:
        """
        Sample a future state from the trajectory using geometric distribution.

        Args:
            current_idx: Index in trajectory of current state
            p: Probability parameter for geometric distribution

        Returns:
            Future state sampled from trajectory, or None if no future states
        """
        max_offset = len(self.trajectory) - current_idx - 1
        if max_offset <= 0:
            return None

        # Sample offset from geometric distribution: P(k) = p * (1-p)^(k-1)
        # np.random.geometric returns k in {1, 2, 3, ...}
        offset = np.random.geometric(p)
        offset = min(offset, max_offset)

        return self.trajectory[current_idx + offset]
