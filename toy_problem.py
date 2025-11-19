import numpy as np
import networkx as nx
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from typing import Tuple, List


class GraphProblem:
    """
    Represents a graph-based planning problem with random walk dynamics.
    """

    def __init__(
        self,
        n_nodes: int = 50,
        gamma: float = 0.9,
        regular: bool = True,
        degree: int = 4,
        edge_prob: float = 0.15,
        seed: int = None
    ):
        """
        Initialize the graph problem.

        Args:
            n_nodes: Number of nodes in the graph
            gamma: Discount factor for state occupancy measure
            regular: If True, generate d-regular graph; if False, generate Erdos-Renyi graph
            degree: Degree for d-regular graph (used if regular=True, must satisfy n*d is even and d < n)
            edge_prob: Edge probability for Erdos-Renyi graph (used if regular=False)
            seed: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.gamma = gamma
        self.regular = regular
        self.degree = degree
        self.edge_prob = edge_prob
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.graph = None
        self.adjacency_matrix = None
        self.occupancy_matrix = None

        if regular:
            self.generate_random_regular_graph(degree)
        else:
            self.generate_random_connected_graph(edge_prob)

        self.compute_discounted_occupancy()

    def generate_random_regular_graph(self, degree: int = 4) -> nx.Graph:
        """
        Generate a random d-regular connected graph.

        Args:
            degree: Degree for each node (must satisfy n*d is even and d < n)

        Returns:
            d-regular connected graph
        """
        # Check constraints for d-regular graph
        if degree >= self.n_nodes:
            raise ValueError(f"Degree {degree} must be less than n_nodes {self.n_nodes}")

        if (self.n_nodes * degree) % 2 != 0:
            raise ValueError(f"n_nodes * degree must be even. Got {self.n_nodes} * {degree} = {self.n_nodes * degree}")

        # Generate random d-regular graph (guaranteed to be connected for d >= 2)
        G = nx.random_regular_graph(degree, self.n_nodes, seed=self.seed)

        self.graph = G
        self.adjacency_matrix = nx.to_numpy_array(G)
        return G

    def generate_random_connected_graph(self, edge_prob: float = 0.15) -> nx.Graph:
        """
        Generate a random connected graph using Erdos-Renyi model.

        Args:
            edge_prob: Probability of edge creation between nodes

        Returns:
            Connected graph
        """
        # Generate random graph and ensure it's connected
        attempts = 0
        max_attempts = 100
        current_prob = edge_prob

        while attempts < max_attempts:
            G = nx.erdos_renyi_graph(self.n_nodes, current_prob, seed=self.seed if self.seed is None else self.seed + attempts)
            if nx.is_connected(G):
                break
            # If not connected, try again with slightly higher probability
            current_prob += 0.01
            attempts += 1

        if not nx.is_connected(G):
            raise ValueError(f"Failed to generate connected graph after {max_attempts} attempts")

        self.graph = G
        self.adjacency_matrix = nx.to_numpy_array(G)
        return G

    def compute_discounted_occupancy(self) -> np.ndarray:
        """
        Compute discounted state occupancy measure from adjacency matrix.

        The occupancy measure O[i,j] represents the expected discounted number
        of visits to state j starting from state i under a random walk policy.

        O = (I - gamma * P)^(-1)
        where P is the transition probability matrix (row-normalized adjacency)

        Returns:
            Discounted occupancy matrix
        """
        # Normalize adjacency matrix to get transition probabilities
        degree = self.adjacency_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero for isolated nodes
        degree = np.where(degree == 0, 1, degree)
        transition_matrix = self.adjacency_matrix / degree

        # Compute discounted occupancy: (I - gamma * P)^(-1)
        I = np.eye(self.n_nodes)
        occupancy = (1 - self.gamma) * np.linalg.inv(I - self.gamma * transition_matrix)

        self.occupancy_matrix = occupancy
        return occupancy


class InterpolationPlanner:
    """
    A planning method that interpolates in a latent space where distances
    are derived from random walk probabilities on a graph.
    """

    def __init__(self, problem: GraphProblem, seed: int = None, n_components: int = 2):
        """
        Initialize the planner with a graph problem.

        Args:
            problem: GraphProblem instance to plan on
            seed: Random seed for reproducibility
        """
        self.problem = problem
        self.seed = seed
        self.embeddings = None

        self.embed_with_mds(n_components=n_components)

    def embed_with_mds(self, n_components: int = 2) -> np.ndarray:
        """
        Use MDS to embed nodes in Euclidean space where distances match
        the discounted occupancy measure.

        Args:
            n_components: Number of dimensions for embedding

        Returns:
            Node embeddings in Euclidean space
        """
        # Convert occupancy to dissimilarity (distance) matrix
        # Higher occupancy = closer distance, so we invert it
        # Use negative log to convert probabilities to distances
        dissimilarity = -np.log(self.problem.occupancy_matrix + 1e-10)

        # Make it symmetric (average with transpose)
        dissimilarity = (dissimilarity + dissimilarity.T) / 2

        # Apply MDS
        mds = MDS(n_components=n_components, dissimilarity='precomputed',
                  random_state=self.seed, max_iter=1000)
        self.embeddings = mds.fit_transform(dissimilarity)

        return self.embeddings
    
    def next_waypoint(self, start: int, goal: int, frac: float = 0.5) -> int:
        current_pos = self.embeddings[start]
        goal_pos = self.embeddings[goal]

        interp = frac * goal_pos + (1 - frac) * current_pos

        distances = np.linalg.norm(self.embeddings - interp, axis=1)
        distances[start] = np.inf  # Exclude current position

        next_node = np.argmin(distances)

        return next_node

    def plan_trajectory(self, start: int, goal: int, max_steps: int = None) -> List[int]:
        """
        Plan a trajectory from start to goal by iteratively moving to the point
        closest to the halfway interpolation between current position and goal.

        Args:
            start: Start node index
            goal: Goal node index
            max_steps: Maximum number of steps (default: n_nodes)

        Returns:
            List of node indices forming the trajectory
        """
        if max_steps is None:
            max_steps = self.problem.n_nodes

        trajectory = [start]
        current = start

        for _ in range(max_steps):
            if current == goal:
                break

            # Get embeddings of current position and goal
            current_pos = self.embeddings[current]
            goal_pos = self.embeddings[goal]

            # Compute halfway interpolation point
            halfway = 0.5 * current_pos + 0.5 * goal_pos

            # Find the nearest node to the halfway point (excluding current)
            distances = np.linalg.norm(self.embeddings - halfway, axis=1)
            distances[current] = np.inf  # Exclude current position

            # Also exclude already visited nodes to avoid loops
            for visited in trajectory:
                distances[visited] = np.inf

            # Select nearest node
            next_node = np.argmin(distances)

            # If we can't find a valid next node, try to move toward goal
            if distances[next_node] == np.inf:
                distances = np.linalg.norm(self.embeddings - goal_pos, axis=1)
                for visited in trajectory:
                    distances[visited] = np.inf
                next_node = np.argmin(distances)

            trajectory.append(next_node)
            current = next_node

        return trajectory

    def visualize(self, trajectory: List[int] = None, start: int = None, goal: int = None):
        """
        Visualize the graph embedding and trajectory.

        Args:
            trajectory: Optional trajectory to highlight
            start: Start node
            goal: Goal node
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot original graph
        ax1 = axes[0]
        pos = nx.spring_layout(self.problem.graph, seed=self.seed)
        nx.draw(self.problem.graph, pos, ax=ax1, node_size=100, node_color='lightblue',
                edge_color='gray', alpha=0.6)

        if trajectory:
            # Highlight trajectory in original graph
            trajectory_edges = [(trajectory[i], trajectory[i+1])
                               for i in range(len(trajectory)-1)]
            nx.draw_networkx_nodes(self.problem.graph, pos, nodelist=[start],
                                  node_color='green', node_size=200, ax=ax1)
            nx.draw_networkx_nodes(self.problem.graph, pos, nodelist=[goal],
                                  node_color='red', node_size=200, ax=ax1)
            nx.draw_networkx_nodes(self.problem.graph, pos, nodelist=trajectory[1:-1],
                                  node_color='orange', node_size=150, ax=ax1)

        ax1.set_title('Original Graph')
        ax1.axis('off')

        # Plot MDS embedding
        ax2 = axes[1]
        ax2.scatter(self.embeddings[:, 0], self.embeddings[:, 1],
                   c='lightblue', s=100, alpha=0.6)

        if trajectory:
            # Plot trajectory in embedding space
            traj_coords = self.embeddings[trajectory]
            ax2.plot(traj_coords[:, 0], traj_coords[:, 1],
                    'o-', color='orange', linewidth=2, markersize=8,
                    label='Trajectory')
            ax2.scatter(self.embeddings[start, 0], self.embeddings[start, 1],
                       c='green', s=300, marker='*', label='Start', zorder=5)
            ax2.scatter(self.embeddings[goal, 0], self.embeddings[goal, 1],
                       c='red', s=300, marker='*', label='Goal', zorder=5)
            ax2.legend()

        ax2.set_title('MDS Embedding (Occupancy-based distances)')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')

        plt.tight_layout()
        plt.show()


def main():
    """Test the interpolation planner."""
    # Create graph problem with d-regular graph
    print("=== Testing with d-regular graph ===")
    problem_regular = GraphProblem(n_nodes=30, gamma=0.9, regular=True, degree=4, seed=42)

    print(f"Generated {problem_regular.degree}-regular graph with {problem_regular.graph.number_of_nodes()} nodes "
          f"and {problem_regular.graph.number_of_edges()} edges")

    # Verify the occupancy matrix is symmetric
    is_symmetric = np.allclose(problem_regular.occupancy_matrix, problem_regular.occupancy_matrix.T)
    print(f"Occupancy matrix is symmetric: {is_symmetric}")
    print(f"Occupancy matrix shape: {problem_regular.occupancy_matrix.shape}")

    # Create planner with the problem
    print("\nInitializing interpolation planner...")
    planner_regular = InterpolationPlanner(problem_regular, seed=42, n_components=2)
    print(f"Embeddings shape: {planner_regular.embeddings.shape}")

    # Plan trajectory
    start_node = 0
    goal_node = problem_regular.n_nodes - 1
    print(f"\nPlanning trajectory from node {start_node} to node {goal_node}...")
    trajectory_regular = planner_regular.plan_trajectory(start_node, goal_node)
    print(f"Trajectory: {trajectory_regular}")
    print(f"Trajectory length: {len(trajectory_regular)} steps")

    # Visualize
    print("\nVisualizing...")
    planner_regular.visualize(trajectory_regular, start_node, goal_node)

    # Test with non-regular graph
    print("\n\n=== Testing with non-regular (Erdos-Renyi) graph ===")
    problem_nonregular = GraphProblem(n_nodes=30, gamma=0.9, regular=False, edge_prob=0.2, seed=42)

    print(f"Generated random graph with {problem_nonregular.graph.number_of_nodes()} nodes "
          f"and {problem_nonregular.graph.number_of_edges()} edges")

    # Check symmetry
    is_symmetric = np.allclose(problem_nonregular.occupancy_matrix, problem_nonregular.occupancy_matrix.T)
    print(f"Occupancy matrix is symmetric: {is_symmetric}")

    # Create planner
    planner_nonregular = InterpolationPlanner(problem_nonregular, seed=42, n_components=2)

    # Plan trajectory
    trajectory_nonregular = planner_nonregular.plan_trajectory(start_node, goal_node)
    print(f"Trajectory: {trajectory_nonregular}")
    print(f"Trajectory length: {len(trajectory_nonregular)} steps")

    # Visualize
    print("\nVisualizing...")
    planner_nonregular.visualize(trajectory_nonregular, start_node, goal_node)


if __name__ == "__main__":
    main()
