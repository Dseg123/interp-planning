import numpy as np
from typing import Optional, Callable, List


def c_waypoint(
    start: np.ndarray,
    goal: np.ndarray,
    psi: Callable[[np.ndarray], np.ndarray],
    A: np.ndarray,
    buffer: List[np.ndarray],
    M: int = 50,
    T: float = 1.0
) -> np.ndarray:
    """
    Compute the C-waypoint for a given start and goal state.

    Args:
        start: Current state (N-dimensional array)
        goal: Goal state (N-dimensional array)
        psi: State embedding function
        A: Transformation matrix
        buffer: List of states to sample from
        M: Number of states to sample
        T: Temperature for softmax

    Returns:
        Waypoint embedding in latent space
    """
    start_emb = A @ psi(start)
    goal_emb = psi(goal)

    # Sample M states from buffer
    num_samples = min(M, len(buffer))
    sampled_indices = np.random.choice(len(buffer), size=num_samples, replace=False)
    sampled_states = [buffer[i] for i in sampled_indices]

    # Compute ||psi(s) - start_emb||^2 + ||Apsi(s) - goal_emb||^2 for each sampled state
    costs = []
    for s in sampled_states:
        s_emb = psi(s)
        cost = np.linalg.norm(s_emb - start_emb)**2 + np.linalg.norm(A @ s_emb - goal_emb)**2
        costs.append(cost)
    costs = np.array(costs)

    # Sample by softmaxing over cost
    logits = -costs / T  # Negative cost for softmax
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    probabilities = exp_logits / np.sum(exp_logits)

    chosen_idx = np.random.choice(len(sampled_states), p=probabilities)
    c_waypoint_state = sampled_states[chosen_idx]

    return psi(c_waypoint_state)


def i_waypoint(
    start: np.ndarray,
    goal: np.ndarray,
    psi: Callable[[np.ndarray], np.ndarray],
    A: np.ndarray,
    c: float
) -> np.ndarray:
    """
    Compute the I-waypoint for a given start and goal state.

    Args:
        start: Current state (N-dimensional array)
        goal: Goal state (N-dimensional array)
        psi: State embedding function
        A: Transformation matrix
        c: Interpolation parameter

    Returns:
        Waypoint embedding in latent space
    """
    sigma_inv = (c/(c+1)) * (A.T @ A) + ((c+1)/c) * np.eye(A.shape[1])
    sigma = np.linalg.inv(sigma_inv)
    mu = sigma @ (A.T @ psi(goal) + A @ psi(start))

    i_waypoint_emb = np.random.multivariate_normal(mu, sigma)
    return i_waypoint_emb
