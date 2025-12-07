import numpy as np
from typing import Callable, Tuple, Optional, List
from problems import GridworldProblem
from policies import BasePolicy
from waypoints import c_waypoint, i_waypoint, g_waypoint


class WaypointPlanner:
    """
    A planner that maintains internal state and takes one step at a time toward waypoints.

    Adapted for GridworldProblem with N-dimensional states.
    """

    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        env: GridworldProblem,
        policy: BasePolicy,
        psi: Callable[[np.ndarray], np.ndarray],
        A: np.ndarray,
        waypoint_type: str = 'i',
        max_waypoints: int = 100,
        state_buffer: Optional[List[np.ndarray]] = None,
        M: int = 50,
        c: float = 1.0,
        eps: float = 1e-3,
        T: float = 1.0,
        num_gmm_comps: int = 1,
        num_gmm_iters: int = 10
    ):
        """
        Initialize the waypoint planner.

        Args:
            start: Starting state (N-dimensional array)
            goal: Goal state (N-dimensional array)
            env: GridworldProblem environment
            policy: Policy to select actions
            psi: State encoder function
            A: Transformation matrix
            waypoint_type: 'c' for c-waypoint, 'i' for i-waypoint, or 'n' for no waypoints
            max_waypoints: Maximum number of waypoints before going directly to goal
            state_buffer: Buffer of states for c-waypoint sampling (shared across planners)
            M: Number of samples for c-waypoint
            c: Parameter for i-waypoint
            eps: Distance threshold to consider waypoint reached
            T: Temperature for c-waypoint
        """
        self.start = start.copy()
        self.goal = goal.copy()
        self.env = env
        self.policy = policy
        self.psi = psi
        self.A = A
        self.waypoint_type = waypoint_type
        self.max_waypoints = max_waypoints
        self.state_buffer = state_buffer if state_buffer is not None else []
        self.M = M
        self.c = c
        self.eps = eps
        self.T = T
        self.num_gmm_comps = num_gmm_comps
        self.num_gmm_iters = num_gmm_iters

        # Internal state
        self.current_state = start.copy()
        self.num_waypoints_used = 0
        self.current_waypoint = None
        self.done = False
        self.trajectory = [start.copy()]  # Track trajectory for future state sampling

        # Reset environment and compute initial waypoint
        self.env.reset(start, goal)
        self._update_waypoint()

    def _update_waypoint(self):
        """Update the current waypoint based on current state and goal."""
        if self.num_waypoints_used < self.max_waypoints:
            # Generate new waypoint
            if self.waypoint_type == 'c':
                # Use state buffer for c-waypoint sampling
                if len(self.state_buffer) < self.M:
                    # Not enough states in buffer, use direct goal
                    self.current_waypoint = self.psi(self.goal)
                else:
                    self.current_waypoint = c_waypoint(
                        self.current_state, self.goal, self.psi, self.A,
                        self.state_buffer, self.M, self.T
                    )
            elif self.waypoint_type == 'i':
                self.current_waypoint = i_waypoint(
                    self.current_state, self.goal, self.psi, self.A, self.c
                )
            elif self.waypoint_type == 'g':
                if len(self.state_buffer) < self.M:
                    # Not enough states in buffer, use direct goal
                    self.current_waypoint = self.psi(self.goal)
                else:
                    self.current_waypoint = g_waypoint(
                        self.current_state, self.goal, self.psi, self.A,
                        self.state_buffer, self.M, self.num_gmm_comps, self.T, self.num_gmm_iters
                    )
                
            elif self.waypoint_type == 'n':
                # No waypoints - go directly to goal
                self.current_waypoint = self.psi(self.goal)
            else:
                raise ValueError("Invalid waypoint type. Must be 'c', 'i', or 'n'.")
        else:
            # Max waypoints reached, target goal directly
            self.current_waypoint = self.psi(self.goal)

    def step(self, debug: bool = False) -> Tuple[Optional[int], Optional[np.ndarray], bool]:
        """
        Take one step toward the current waypoint.

        Returns:
            Tuple of (action, waypoint, done):
                - action: The action taken (index)
                - waypoint: The waypoint embedding we were targeting
                - done: Whether we've reached the goal
        """
        if self.done:
            return None, None, True

        # Check if we've reached the goal
        if np.array_equal(self.current_state, self.goal):
            self.done = True
            return None, None, True

        # Get current waypoint
        waypoint = self.current_waypoint

        # Select action using policy
        action = self.policy.get_action(self.current_state, waypoint)

        if action is None:
            # No valid actions available
            self.done = True
            return None, waypoint, True

        # Take step in environment
        next_state, reward, done, info = self.env.step(action)

        # Update current state and trajectory
        self.current_state = next_state.copy()
        self.trajectory.append(next_state.copy())

        # Check if we've reached the current waypoint in embedding space
        curr_z = self.A @ self.psi(self.current_state)
        if np.linalg.norm(curr_z - waypoint) < self.eps:
            # Waypoint reached, update to next waypoint
            self.num_waypoints_used += 1
            self._update_waypoint()

        # Check if done
        if np.array_equal(self.current_state, self.goal):
            self.done = True

        if debug:
            print(f"Current state: {self.current_state}, Waypoint: {waypoint}, "
                  f"Action taken: {action}, Done: {self.done}")

        return action, waypoint, self.done

    def reset(self, start: Optional[np.ndarray] = None, goal: Optional[np.ndarray] = None):
        """
        Reset the planner with new start and goal.

        Args:
            start: New starting state (if None, use original start)
            goal: New goal state (if None, use original goal)
        """
        if start is not None:
            self.start = start.copy()
        if goal is not None:
            self.goal = goal.copy()

        self.current_state = self.start.copy()
        self.num_waypoints_used = 0
        self.current_waypoint = None
        self.done = False
        self.trajectory = [self.start.copy()]  # Reset trajectory

        self.env.reset(self.start, self.goal)
        self._update_waypoint()

    def sample_future_state(self, current_idx: int, p: float = 0.9) -> Optional[np.ndarray]:
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

        return self.trajectory[current_idx + offset].copy()

    def rollout(self, start: np.ndarray, goal: np.ndarray, max_steps: int, num_waypoints: int, temperature: float) -> Tuple[List[np.ndarray], int, bool]:
        """
        Perform a full rollout from `start` to `goal` using at most `num_waypoints` waypoints.

        The planner is reset to `start`/`goal`, its `max_waypoints` is set to
        `num_waypoints` for the duration of the rollout, and `step()` is called
        repeatedly until the planner reaches the goal or `max_steps` is exceeded.

        Args:
            start: Starting state (N-dimensional array)
            goal: Goal state (N-dimensional array)
            max_steps: Maximum number of environment steps to execute
            num_waypoints: Maximum number of waypoints to use during this rollout

        Returns:
            states_visited: Full list of states visited (including start)
            path_length: Number of steps taken (len(states_visited) - 1)
            success: Whether the rollout reached the goal
        """
        # Quick check: if start equals goal, return immediately
        if np.array_equal(start, goal):
            return [start.copy()], 0, True

        # Save old planner setting and reset planner
        old_max_waypoints = self.max_waypoints
        try:
            old_temperature = self.policy.temperature
        except:
            pass
            

        try:
            self.reset(start=start, goal=goal)
            # Use the provided waypoint budget for this rollout
            self.max_waypoints = int(num_waypoints)
            try:
                self.policy.temperature = temperature
            except:
                pass

            steps = 0
            while steps < max_steps and not self.done:
                self.step()
                steps += 1

            states_visited = [s.copy() for s in self.trajectory]
            path_length = max(0, len(states_visited) - 1)
            success = np.array_equal(self.current_state, self.goal)

            return states_visited, path_length, bool(success)
        finally:
            # Restore previous planner configuration
            self.max_waypoints = old_max_waypoints
            try:
                self.policy.temperature = temperature
            except:
                pass
