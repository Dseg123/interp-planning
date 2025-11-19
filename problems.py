import numpy as np
import networkx as nx
from typing import List, Tuple, Optional


class GraphProblem:
    """
    A graph-based environment where an agent navigates through vertices.
    The graph is randomly generated as a connected undirected graph.
    """

    def __init__(self, num_vertices: int, edge_probability: float = 0.3, seed: Optional[int] = None):
        """
        Initialize a random connected graph environment.

        Args:
            num_vertices: Number of vertices in the graph
            edge_probability: Probability of edge creation in ErdQsR�nyi model
            seed: Random seed for reproducibility
        """
        self.num_vertices = num_vertices
        self.edge_probability = edge_probability
        self.rng = np.random.RandomState(seed)

        # Generate a random connected graph
        self.graph = self._generate_connected_graph()

        # Current state and goal (vertices)
        self.current_state = None
        self.goal_state = None
        self.reset()

    def _generate_connected_graph(self) -> nx.Graph:
        """
        Generate a random connected undirected graph using ErdQsR�nyi model.
        If the generated graph is not connected, regenerate until we get a connected one.
        """
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random graph
            G = nx.erdos_renyi_graph(
                self.num_vertices,
                self.edge_probability,
                seed=self.rng.randint(0, 2**32 - 1)
            )

            # Check if connected
            if nx.is_connected(G):
                return G

        # Fallback: create a connected graph by starting with a spanning tree
        # and adding random edges
        G = nx.Graph()
        G.add_nodes_from(range(self.num_vertices))

        # Create a random spanning tree (ensures connectivity)
        vertices = list(range(self.num_vertices))
        self.rng.shuffle(vertices)
        for i in range(1, self.num_vertices):
            # Connect each vertex to a random previous vertex
            j = self.rng.randint(0, i)
            G.add_edge(vertices[i], vertices[j])

        # Add additional random edges
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if not G.has_edge(i, j) and self.rng.random() < self.edge_probability:
                    G.add_edge(i, j)

        return G

    def reset(self, state: Optional[int] = None, goal: Optional[int] = None) -> Tuple[int, int]:
        """
        Reset the environment to a random state and goal (or specified states).

        Args:
            state: Optional specific state to reset to. If None, choose randomly.
            goal: Optional specific goal to set. If None, choose randomly (different from start).

        Returns:
            Tuple of (initial_state, goal_state)
        """
        if state is None:
            self.current_state = self.rng.randint(0, self.num_vertices)
        else:
            if state < 0 or state >= self.num_vertices:
                raise ValueError(f"State {state} is out of bounds [0, {self.num_vertices})")
            self.current_state = state

        if goal is None:
            # Choose a random goal different from the current state
            available_goals = [s for s in range(self.num_vertices) if s != self.current_state]
            self.goal_state = self.rng.choice(available_goals)
        else:
            if goal < 0 or goal >= self.num_vertices:
                raise ValueError(f"Goal {goal} is out of bounds [0, {self.num_vertices})")
            self.goal_state = goal

        return self.current_state, self.goal_state

    def get_available_actions(self, state: Optional[int] = None) -> List[int]:
        """
        Get the list of available actions (neighboring vertices) from a given state.

        Args:
            state: The state to query. If None, use current state.

        Returns:
            List of vertex IDs that are neighbors of the given state
        """
        if state is None:
            state = self.current_state

        if state is None:
            raise ValueError("No current state set. Call reset() first.")

        return list(self.graph.neighbors(state))

    def step(self, action: int) -> Tuple[int, bool]:
        """
        Take a step in the environment by moving to a neighboring vertex.

        Args:
            action: The target vertex to move to (must be a neighbor of current state)

        Returns:
            Tuple of (next_state, valid):
                - next_state: The new current state
                - valid: Whether the action was valid (neighbor exists)
        """
        if self.current_state is None:
            raise ValueError("No current state set. Call reset() first.")

        # Check if action is valid (is a neighbor)
        available_actions = self.get_available_actions()

        if action not in available_actions:
            # Invalid action - don't move
            return self.current_state, False

        # Valid action - move to new state
        self.current_state = action
        return self.current_state, True

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix of the graph.

        Returns:
            A (num_vertices x num_vertices) adjacency matrix
        """
        return nx.to_numpy_array(self.graph)

    def get_num_edges(self) -> int:
        """
        Get the number of edges in the graph.

        Returns:
            Number of edges
        """
        return self.graph.number_of_edges()

    def __repr__(self) -> str:
        return (f"GraphProblem(num_vertices={self.num_vertices}, "
                f"num_edges={self.get_num_edges()}, "
                f"current_state={self.current_state}, "
                f"goal_state={self.goal_state})")
