import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Callable
from problems import GraphProblem
from policies import GreedyPolicy
import networkx as nx


def contrastive_loss(psi, A, actions, waypoints):
    """
    Compute contrastive loss for learning representations.

    This follows the C-learning critic loss formulation:
    - Positive pairs: (action[i], waypoint[i]) should have high similarity
    - Negative pairs: (action[i], waypoint[j]) for i != j should have low similarity

    Args:
        psi: State embedding matrix (n x k) - PyTorch tensor
        A: Transformation matrix (k x k) - PyTorch tensor
        actions: List or array of action states (integers) - length batch_size
        waypoints: Waypoint embeddings (batch_size x k) - PyTorch tensor

    Returns:
        Loss value (scalar tensor)
    """
    # Convert actions to tensor if needed
    if not isinstance(actions, torch.Tensor):
        actions = torch.tensor(actions, dtype=torch.long)

    # Get action embeddings: psi(action)
    action_emb = psi[actions]  # (batch_size, k)

    # Transform action embeddings: A @ psi(action)
    # action_emb is (batch_size, k), A is (k, k)
    # We want (batch_size, k) @ (k, k)^T = (batch_size, k)
    sa_repr = action_emb @ A.T  # (batch_size, k)

    # Waypoints are already embeddings
    g_repr = waypoints  # (batch_size, k)

    # Compute logits: <sa_repr[i], g_repr[j]> for all i, j
    # This is the similarity matrix
    logits = torch.einsum('ik,jk->ij', sa_repr, g_repr)  # (batch_size, batch_size)

    # Labels: identity matrix (positive pairs on diagonal)
    batch_size = logits.shape[0]
    labels = torch.eye(batch_size, device=logits.device, dtype=logits.dtype)

    # Binary cross-entropy with logits
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    return loss


def contrastive_loss_numpy(psi, A, actions, waypoints):
    """
    NumPy version of contrastive loss (for reference/testing).

    Args:
        psi: State embedding matrix (n x k) - numpy array
        A: Transformation matrix (k x k) - numpy array
        actions: List or array of action states (integers) - length batch_size
        waypoints: Waypoint embeddings (batch_size x k) - numpy array

    Returns:
        Loss value (scalar)
    """
    # Get action embeddings
    action_emb = psi[actions]  # (batch_size, k)

    # Transform action embeddings
    sa_repr = action_emb @ A.T  # (batch_size, k)

    # Waypoints are already embeddings
    g_repr = waypoints  # (batch_size, k)

    # Compute logits
    logits = np.einsum('ik,jk->ij', sa_repr, g_repr)  # (batch_size, batch_size)

    # Labels: identity matrix
    batch_size = logits.shape[0]
    labels = np.eye(batch_size)

    # Binary cross-entropy with logits
    # BCE = -sum(y * log(sigmoid(x)) + (1-y) * log(1 - sigmoid(x)))
    sigmoid_logits = 1 / (1 + np.exp(-logits))
    loss = -np.mean(
        labels * np.log(sigmoid_logits + 1e-8) +
        (1 - labels) * np.log(1 - sigmoid_logits + 1e-8)
    )

    return loss


def learning_epoch(
    psi: torch.Tensor,
    A: torch.Tensor,
    actions: List[int],
    waypoints: torch.Tensor,
    env: GraphProblem,
    learning_rate: float = 0.01,
    iters_per_epoch: int = 1,
    T: float=1.0
) -> Tuple[torch.Tensor, torch.Tensor, GreedyPolicy]:
    """
    Perform gradient descent on psi and A using contrastive loss.

    Args:
        psi: State embedding matrix (n x k) - PyTorch tensor with requires_grad=True
        A: Transformation matrix (k x k) - PyTorch tensor with requires_grad=True
        actions: List of action states (integers) from batch
        waypoints: Waypoint embeddings (batch_size x k) - PyTorch tensor
        env: GraphProblem environment for creating the policy
        learning_rate: Learning rate for gradient descent
        iters_per_epoch: Number of gradient steps to perform

    Returns:
        Tuple of (updated_psi, updated_A, new_policy)
    """
    # Create optimizer
    optimizer = torch.optim.Adam([psi, A], lr=learning_rate)

    # Run gradient descent for iters_per_epoch steps
    for _ in range(iters_per_epoch):
        optimizer.zero_grad()

        # Compute loss
        loss = contrastive_loss(psi, A, actions, waypoints)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

    # Create new policy with updated parameters
    psi_np = psi.detach().numpy()
    A_np = A.detach().numpy()

    # def psi_fun(x: int) -> np.ndarray:
    #     return psi_np[x, :]

    new_policy = GreedyPolicy(env, lambda x: A_np @ psi_np[x, :], temperature=T)

    return psi, A, new_policy, loss.item()


def compute_transition_matrix(env: GraphProblem) -> np.ndarray:
    """
    Compute the transition matrix P for uniform random walk on the graph.
    P[i, j] = 1/degree(i) if there's an edge from i to j, else 0

    Args:
        env: GraphProblem environment

    Returns:
        Transition matrix P (n x n) where P[i,j] is probability of going from i to j
    """
    n = env.num_vertices
    P = np.zeros((n, n))

    for i in range(n):
        neighbors = list(env.graph.neighbors(i))
        if len(neighbors) > 0:
            # Uniform random walk: equal probability to each neighbor
            prob = 1.0 / len(neighbors)
            for j in neighbors:
                P[i, j] = prob

    return P


def compute_occupancy_matrix(P: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    """
    Compute the discounted occupancy matrix O = (I - gamma * P)^{-1}

    O[i, j] represents the expected discounted number of times we visit state j
    starting from state i under a random policy.

    Args:
        P: Transition matrix (n x n)
        gamma: Discount factor

    Returns:
        Occupancy matrix O (n x n)
    """
    n = P.shape[0]
    I = np.eye(n)
    O = np.linalg.inv(I - gamma * P)
    return O


def compute_stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution V where V = P^T V (or equivalently V^T P = V^T)

    Args:
        P: Transition matrix (n x n)
        method: 'eigenvalue' or 'iterative'

    Returns:
        Stationary distribution V (n,) - probability of being in each state
    """
    n = P.shape[0]

    # Start with uniform distribution
    V = np.ones(n) / n

    # Iterate V = P^T V until convergence
    for _ in range(1000):
        V_new = P.T @ V
        if np.allclose(V_new, V):
            break
        V = V_new

    return V


def initialize_with_mds(env: GraphProblem, k: int, gamma: float = 0.9) -> np.ndarray:
    """
    Initialize state embeddings using MDS on discounted occupancy distances.

    The distance between states i and j is based on their occupancy patterns:
    d(i, j) = ||O[i, :] - O[j, :]|| where O is the discounted occupancy matrix.

    Args:
        env: GraphProblem environment
        k: Embedding dimension
        gamma: Discount factor for occupancy

    Returns:
        State embeddings psi (n x k)
    """
    from sklearn.manifold import MDS

    # Compute transition matrix
    P = compute_transition_matrix(env)

    # Compute occupancy matrix
    O = compute_occupancy_matrix(P, gamma)

    V = compute_stationary_distribution(P)

    # Compute pairwise distances between occupancy vectors
    n = env.num_vertices
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(-2 * np.log((O[i, j] / V[j]) * (np.min(V) / np.max(O))))
            if np.isnan(distances[i, j]):
                print(i, j)
                print(O[i, j], V[j], np.min(V))

    # print("V:", V)
    # print("Distances:", distances)
    # print(np.isnan(distances).any())
    # Apply MDS to get k-dimensional embeddings
    mds = MDS(n_components=k, dissimilarity='precomputed', random_state=42)
    psi = mds.fit_transform(distances)

    return psi
