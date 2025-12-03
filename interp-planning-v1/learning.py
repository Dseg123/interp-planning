import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Callable
from problems import GraphProblem
from policies import GreedyPolicy
import networkx as nx


def contrastive_loss(psi, A, current_states, future_states, l2_reg=0.01, use_infonce=True):
    """
    Compute contrastive loss with L2 regularization.

    This follows the C-learning critic loss formulation:
    - Option 1 (InfoNCE): Uses cross-entropy over softmax similarities
    - Option 2 (BCE): Uses binary cross-entropy with identity matrix labels
    - Adds L2 regularization: average squared norm over embeddings in the batch

    The loss trains A @ psi(s_t) to predict psi(s_t+), where s_t+ is sampled
    from a geometric distribution over future states.

    Args:
        psi: State embedding matrix (n x k) - PyTorch tensor
        A: Transformation matrix (k x k) - PyTorch tensor
        current_states: List or array of current states (integers) - length batch_size
        future_states: List or array of future states (integers) - length batch_size
        l2_reg: L2 regularization coefficient for embedding norms
        use_infonce: If True, use InfoNCE (cross-entropy), otherwise use BCE

    Returns:
        Loss value (scalar tensor)
    """
    # Convert to tensors if needed
    if not isinstance(current_states, torch.Tensor):
        current_states = torch.tensor(current_states, dtype=torch.long)
    if not isinstance(future_states, torch.Tensor):
        future_states = torch.tensor(future_states, dtype=torch.long)

    # Get current state embeddings and transform them: A @ psi(s_t)
    current_emb = psi[current_states]  # (batch_size, k)
    sa_repr = current_emb @ A.T  # (batch_size, k) - prediction of future

    # Get future state embeddings: psi(s_t+)
    future_emb = psi[future_states]  # (batch_size, k)

    # Compute logits: <sa_repr[i], future_emb[j]> for all i, j
    # This is the similarity matrix
    logits = torch.einsum('ik,jk->ij', sa_repr, future_emb)  # (batch_size, batch_size)

    batch_size = logits.shape[0]

    if use_infonce:
        # InfoNCE loss: cross-entropy where positive pair is on diagonal
        labels = torch.arange(batch_size, device=logits.device)
        contrastive_loss_val = F.cross_entropy(logits, labels)
    else:
        # Binary cross-entropy with identity matrix labels (original C-learning)
        labels = torch.eye(batch_size, device=logits.device, dtype=logits.dtype)
        contrastive_loss_val = F.binary_cross_entropy_with_logits(logits, labels)

    # L2 regularization: average squared norm over embeddings in current_states
    # ||psi(s)||^2 for s in current_states
    current_squared_norms = torch.sum(current_emb ** 2, dim=1)  # (batch_size,)
    l2_loss = torch.mean(current_squared_norms)

    # Total loss
    loss = contrastive_loss_val + l2_reg * l2_loss

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
    current_states: List[int],
    future_states: List[int],
    env: GraphProblem,
    learning_rate: float = 0.01,
    iters_per_epoch: int = 1,
    policy_temperature: float = 1.0,
    l2_reg: float = 0.01,
    use_infonce: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, GreedyPolicy, float]:
    """
    Perform gradient descent on psi and A using contrastive loss.

    Args:
        psi: State embedding matrix (n x k) - PyTorch tensor with requires_grad=True
        A: Transformation matrix (k x k) - PyTorch tensor with requires_grad=True
        current_states: List of current states (integers) from batch
        future_states: List of future states (integers) sampled with geometric distribution
        env: GraphProblem environment for creating the policy
        learning_rate: Learning rate for gradient descent
        iters_per_epoch: Number of gradient steps to perform
        policy_temperature: Temperature for policy softmax
        l2_reg: L2 regularization coefficient
        use_infonce: If True, use InfoNCE loss, otherwise use BCE

    Returns:
        Tuple of (updated_psi, updated_A, new_policy, loss)
    """
    # Create optimizer
    optimizer = torch.optim.Adam([psi, A], lr=learning_rate)

    # Run gradient descent for iters_per_epoch steps
    for _ in range(iters_per_epoch):
        optimizer.zero_grad()

        # Compute loss
        loss = contrastive_loss(psi, A, current_states, future_states,
                               l2_reg=l2_reg, use_infonce=use_infonce)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

    # Create new policy with updated parameters
    psi_np = psi.detach().numpy()
    A_np = A.detach().numpy()

    # Policy uses A @ psi to predict future states
    new_policy = GreedyPolicy(env, lambda x: A_np @ psi_np[x, :], temperature=policy_temperature)

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
