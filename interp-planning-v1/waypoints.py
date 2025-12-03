import numpy as np
from typing import Optional, Callable


def c_waypoint(start: int, goal: int, psi: Callable[[int], np.ndarray], A: np.ndarray, buffer: np.ndarray, M: int = 50, T: float = 1) -> np.ndarray:
    """
    Compute the C-waypoint for a given start and goal state.
    """
    start_emb = A @ psi(start)
    goal_emb = psi(goal)

    # Sample M states from buffer
    sampled_states = np.random.choice(buffer, size=min(M, len(buffer)), replace=False)

    # Compute ||psi(s) - start_emb||^2 + ||Apsi(s) - goal_emb||^2 for each sampled state
    costs = []
    for s in sampled_states:
        s_emb = psi(s)
        cost = np.linalg.norm(s_emb - start_emb)**2 + np.linalg.norm(A @ s_emb - goal_emb)**2
        costs.append(cost)
    costs = np.array(costs)

    # Sample by softmaxing over cost
    # Softmax: exp(logits) / sum(exp(logits))
    logits = -costs / T  # Negative cost for softmax
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    probabilities = exp_logits / np.sum(exp_logits)
    
    c_waypoint = np.random.choice(sampled_states, p=probabilities)

    return psi(c_waypoint)


def i_waypoint(start: int, goal: int, psi: Callable[[int], np.ndarray], A: np.ndarray, c: float) -> np.ndarray:
    """
    Compute the I-waypoint for a given start and goal state.
    """
    sigma_inv = (c/(c+1)) * (A.T @ A) + ((c+1)/c) * np.eye(A.shape[1])
    sigma = np.linalg.inv(sigma_inv)
    mu = sigma @ (A.T @ psi(goal) + A @ psi(start))

    i_waypoint_emb = np.random.multivariate_normal(mu, sigma)
    return i_waypoint_emb

