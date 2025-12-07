import numpy as np
from typing import Optional, Callable, List
# from sklearn.mixture import GaussianMixture



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

    # print(mu, sigma)

    i_waypoint_emb = np.random.multivariate_normal(mu, sigma)
    return i_waypoint_emb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

def g_waypoint(
    start: np.ndarray,
    goal: np.ndarray,
    psi: Callable[[np.ndarray], np.ndarray],
    A: np.ndarray,
    buffer: List[np.ndarray],
    M: int,
    K: int,              # number of mixture components (default = M)
    T: float = 1.0,
    n_iters: int = 10,
    covariance_type: str = "full"
) -> np.ndarray:
    """
    Compute a GMM-based C-Waypoint (continuous approximation).

    Steps:
    1. Compute importance weights p_i over M sampled buffer states.
    2. Fit a Gaussian Mixture Model over their psi-embeddings using p_i.
    3. Sample once from the GMM to get a continuous waypoint.

    Args:
        start: start state
        goal: goal state
        psi: encoder function psi(s) -> embedding
        A: learned linear operator
        buffer: replay buffer of states
        M: number of states to sample from buffer
        T: temperature for softmax weighting
        K: number of mixture components (defaults to M)
        covariance_type: GMM covariance type ("full", "diag", etc.)

    Returns:
        waypoint_emb: a sampled waypoint in latent space
    """

    # Encode start and goal
    start_emb = A @ psi(start)
    goal_emb = psi(goal)

    # Sample states from buffer
    num_samples = min(M, len(buffer))
    idx = np.random.choice(len(buffer), size=num_samples, replace=False)
    sampled_states = [buffer[i] for i in idx]

    # Compute psi embeddings
    psi_embs = np.array([psi(s) for s in sampled_states])  # shape (M, d)

    # Compute the C-Planning cost for each sampled state
    # cost(s) = ||psi(s) - A psi(start)||^2 + ||A psi(s) - psi(goal)||^2
    costs = np.sum((psi_embs - start_emb) ** 2, axis=1) \
          + np.sum((psi_embs @ A.T - goal_emb) ** 2, axis=1)

    # Softmax probabilities over negative cost
    logits = -costs / (2*T)
    logits -= np.max(logits)
    p = np.exp(logits)
    p /= np.sum(p)

    params = fit_gmm_match_probs(psi_embs, p, K, 
                                 n_iters=n_iters)
    

    # Sample 1 waypoint from the GMM
    waypoint_emb = sample_from_fitted_gmm(params)

    return waypoint_emb

def fit_gmm_match_probs(
    psi_embs: np.ndarray,       # shape (M, d)
    p: np.ndarray,              # shape (M,), sums to 1
    K: int = 5,
    n_iters: int = 10,
    lr: float = 1e-2,
    min_diag: float = 1e-6,
    device: str = "cpu",
) -> dict:
    """
    Fit a K-component GMM so that the discrete probabilities defined by normalizing
    g(psi_i) over the M points match target p as closely as possible (minimize KL).
    Returns dict with learned params (weights, means, cov_cholesky).
    """

    psi = torch.tensor(psi_embs, dtype=torch.float32, device=device)  # (M, d)
    p_tensor = torch.tensor(p.astype(np.float32), dtype=torch.float32, device=device)  # (M,)
    M, d = psi.shape

    # Parameters:
    # - raw_weights: (K,) -> softmax -> mixture weights
    # - means: (K, d)
    # - lower-tri cholesky factors: (K, d, d) represented as unconstrained bilinear params for lower triangle
    raw_weights = nn.Parameter(torch.zeros(K, device=device))
    means = nn.Parameter(torch.randn(K, d, device=device) * (psi.std().item() + 1e-2) + psi.mean(0))
    # We'll represent cholesky factors as lower-tri params stored in a matrix
    L_params = nn.Parameter(torch.stack([torch.eye(d, device=device) for _ in range(K)]) * 0.1)

    optimizer = optim.Adam([raw_weights, means, L_params], lr=lr)

    # Helper to compute multivariate normal density at all psi points
    def gmm_pdf_at_points():
        # compute mixture weights
        weights = torch.softmax(raw_weights, dim=0)  # (K,)

        # For numerical stability, compute logpdfs
        # For each component k compute log N(psi | mean_k, Sigma_k)
        logpdfs = []
        for k in range(K):
            L = torch.tril(L_params[k])  # (d,d) lower triangular
            # ensure positive diagonal
            diag = torch.diag(L)
            # push diag to be > min_diag
            diag = torch.clamp(diag, min=min_diag)
            L = L - torch.diag(torch.diag(L)) + torch.diag(diag)

            # compute Sigma = L L^T
            # logdet Sigma = 2 * sum log diag(L)
            logdet = 2.0 * torch.sum(torch.log(diag + 1e-12))

            diff = psi - means[k]  # (M, d)
            # Solve y = L^{-1} diff^T  -> use triangular solve
            # triangular_solve expects (d, M) RHS
            y, _ = torch.triangular_solve(diff.t(), L, upper=False)
            mahal = torch.sum(y * y, dim=0)  # (M,)
            log_const = -0.5 * (d * np.log(2 * np.pi) + logdet)
            logpdf_k = log_const - 0.5 * mahal  # (M,)
            logpdfs.append(logpdf_k.unsqueeze(0))
        logpdfs = torch.cat(logpdfs, dim=0)  # (K, M)

        # log g(psi_i) = logsumexp_k [ log weight_k + logpdf_k(i) ]
        log_weights = torch.log(weights + 1e-12).unsqueeze(1)  # (K,1)
        log_component = log_weights + logpdfs  # (K,M)
        log_g = torch.logsumexp(log_component, dim=0)  # (M,)
        g = torch.exp(log_g)  # (M,)
        return g, log_g, weights

    # Optimization loop
    for it in range(n_iters):
        print(it)
        optimizer.zero_grad()
        g, log_g, weights = gmm_pdf_at_points()  # g: (M,)
        # normalize to discrete q
        q = g / (g.sum() + 1e-12)  # (M,)

        # loss = KL(p || q) = - sum_i p_i * log q_i  + const (we drop const)
        loss = -torch.sum(p_tensor * torch.log(q + 1e-12))
        loss.backward()
        optimizer.step()

        # (optional) small step to prevent collapse of L diagonals: nudge diagonals up
        with torch.no_grad():
            for k in range(K):
                L = torch.tril(L_params[k])
                diag = torch.diag(L)
                diag_clamped = torch.clamp(diag, min=min_diag)
                newL = L - torch.diag(diag) + torch.diag(diag_clamped)
                L_params[k].copy_(newL)

        # simple early stopping if loss small, optionally print
        if it % 200 == 0:
            loss_val = loss.item()
            # print(f"iter {it}, loss {loss_val:.6f}")

    # Return parameters as numpy
    weights_np = torch.softmax(raw_weights, dim=0).detach().cpu().numpy()
    means_np = means.detach().cpu().numpy()
    L_np = torch.tril(L_params.detach()).cpu().numpy()  # (K, d, d)

    return {
        "weights": weights_np,
        "means": means_np,
        "cholesky": L_np,
        "loss": loss.item()
    }


def sample_from_fitted_gmm(params: dict, n: int = 1) -> np.ndarray:
    """
    Sample n points from the fitted GMM returned by fit_gmm_match_probs.
    """
    weights = params["weights"]
    means = params["means"]
    Ls = params["cholesky"]
    K, d = means.shape

    comps = np.random.choice(K, size=n, p=weights)
    samples = np.zeros((n, d))
    for idx, k in enumerate(comps):
        z = np.random.randn(d)
        L = Ls[k]
        samples[idx] = means[k] + L @ z
    return samples if n > 1 else samples[0]
