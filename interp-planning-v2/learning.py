import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Callable
from problems import GridworldProblem
from policies import GreedyPolicy
import time

class StateEncoder(nn.Module):
    """
    Neural network that encodes N-dimensional states to k-dimensional embeddings.

    Maps state (N-dimensional integer vector) -> embedding (k-dimensional continuous vector)
    """

    def __init__(self, state_dim: int, latent_dim: int, hidden_dims: List[int] = [64, 64]):
        """
        Initialize the state encoder network.

        Args:
            state_dim: Dimension of input state (N)
            latent_dim: Dimension of output embedding (k)
            hidden_dims: List of hidden layer sizes
        """
        super(StateEncoder, self).__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Final layer to latent dimension
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encode states to embeddings.

        Args:
            states: Batch of states (batch_size, state_dim) - float tensor

        Returns:
            Embeddings (batch_size, latent_dim)
        """
        return self.network(states)


def contrastive_loss_with_dual(
    psi_net: StateEncoder,
    A: torch.Tensor,
    log_lambda: torch.Tensor,
    current_states: torch.Tensor,
    future_states: torch.Tensor,
    c_target: float = 1.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, dict]:
    """
    Compute contrastive loss with dual formulation from the paper.

    This implements the loss from the JAX example:
    - phi = A @ psi(x0)
    - psi = psi(xT)
    - l_align: ||phi - psi||^2 (align positive pairs)
    - l_unif: logsumexp over negative pair distances (uniformity)
    - l2: mean squared norm of embeddings
    - dual_loss: log_lambda * (c - l2) to constrain L2 norm

    Args:
        psi_net: Neural network encoder
        A: Transformation matrix (k x k)
        log_lambda: Scalar log of Lagrange multiplier (for dual formulation)
        current_states: Batch of current states (batch_size, state_dim)
        future_states: Batch of future states (batch_size, state_dim)
        c_target: Target L2 norm constraint
        device: Device to use for computation (CPU or CUDA)

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    if device is None:
        device = next(psi_net.parameters()).device
    
    # Move data to device if needed
    current_states = current_states.to(device)
    future_states = future_states.to(device)
    # Get embeddings
    # phi = A @ psi(x0) - transformed current state embeddings
    # psi = psi(xT) - future state embeddings
    current_emb = psi_net(current_states)  # (batch_size, k)
    phi = current_emb @ A.T  # (batch_size, k)

    future_emb = psi_net(future_states)  # (batch_size, k)
    psi = future_emb  # (batch_size, k)

    batch_size = phi.shape[0]

    # L2 regularization: average squared norm
    l2 = (torch.mean(psi**2) + torch.mean(current_emb**2)) / 2

    # Alignment loss: ||phi - psi||^2 for positive pairs
    l_align = torch.sum((phi - psi) ** 2, dim=1)  # (batch_size,)

    # Pairwise distances: ||phi[i] - psi[j]||^2 for all i, j
    # phi[:, None] is (batch_size, 1, k)
    # psi[None] is (1, batch_size, k)
    # Result is (batch_size, batch_size)
    pdist = torch.mean((phi[:, None] - psi[None]) ** 2, dim=-1)

    # Uniformity loss: logsumexp over negative pairs
    # Mask out diagonal (positive pairs) with identity matrix
    I = torch.eye(batch_size, device=phi.device)

    # For each row i: logsumexp over j != i of -pdist[i, j]
    # For each col j: logsumexp over i != j of -pdist[i, j]
    l_unif = (
        torch.logsumexp(-(pdist * (1 - I)), dim=1) +
        torch.logsumexp(-(pdist.T * (1 - I)), dim=1)
    ) / 2.0

    # Combined contrastive loss
    loss = l_align + l_unif
    # diff = phi[:, None, :] - psi[None, :, :]
    # d2 = torch.mean(diff * diff, dim=-1)
    # logits = -d2 / 2
    # targets = torch.arange(batch_size, dtype=torch.long)
    # loss = F.cross_entropy(logits, targets)

    # Accuracy: how often is the closest future state the correct one?
    accuracy = torch.mean((torch.argmin(pdist, dim=1) == torch.arange(batch_size, device=phi.device)).float())

    # Dual loss to constrain L2 norm
    # dual_loss = log_lambda * (c_target - stop_gradient(l2))
    dual_loss = log_lambda * (c_target - l2.detach())

    # Total loss
    # loss + exp(log_lambda) * l2 + dual_loss
    total_loss = (
        loss.mean() +
        torch.exp(log_lambda).detach() * l2 +
        dual_loss
    )

    # Metrics
    metrics = {
        'loss': loss.mean().item(),
        'l_unif': l_unif.mean().item(),
        'l_align': l_align.mean().item(),
        'accuracy': accuracy.item(),
        'l2': l2.item(),
        'lambda': torch.exp(log_lambda).item(),
        'total_loss': total_loss.item()
    }

    return total_loss, metrics


def learning_epoch(
    psi_net: StateEncoder,
    A: torch.Tensor,
    log_lambda: torch.Tensor,
    current_states: List[np.ndarray],
    future_states: List[np.ndarray],
    env: GridworldProblem,
    learning_rate: float = 0.001,
    iters_per_epoch: int = 1,
    policy_temperature: float = 1.0,
    c_target: float = 1.0,
    device: torch.device = None
) -> Tuple[StateEncoder, torch.Tensor, torch.Tensor, GreedyPolicy, dict]:
    """
    Perform gradient descent on psi_net, A, and log_lambda using contrastive loss.

    Args:
        psi_net: Neural network state encoder
        A: Transformation matrix (k x k)
        log_lambda: Log of Lagrange multiplier (scalar)
        current_states: List of current states (N-dimensional arrays)
        future_states: List of future states (N-dimensional arrays)
        env: GridworldProblem environment
        learning_rate: Learning rate for optimization
        iters_per_epoch: Number of gradient steps
        policy_temperature: Temperature for policy softmax
        c_target: Target L2 norm constraint
        device: Device to use for computation (CPU or CUDA)

    Returns:
        Tuple of (psi_net, A, log_lambda, policy, metrics)
    """
    # Determine device
    if device is None:
        device = next(psi_net.parameters()).device
    
    # Convert states to tensors and move to device
    current_states_np = np.array([s for s in current_states], dtype=np.float32)
    future_states_np = np.array([s for s in future_states], dtype=np.float32)

    current_states_tensor = torch.from_numpy(current_states_np).to(device)
    future_states_tensor = torch.from_numpy(future_states_np).to(device)

    # Create optimizer for all parameters
    optimizer = torch.optim.Adam(
        list(psi_net.parameters()) + [A, log_lambda],
        lr=learning_rate
    )

    # Run gradient descent
    for _ in range(iters_per_epoch):
        optimizer.zero_grad()

        # Compute loss
        start = time.time()
        loss, metrics = contrastive_loss_with_dual(
            psi_net, A, log_lambda,
            current_states_tensor, future_states_tensor,
            c_target=c_target,
            device=device
        )
        print("Loss time:", time.time() - start)

        # Backward pass
        start = time.time()
        loss.backward()
        print("Back time:", time.time() - start)

        # Update parameters
        optimizer.step()

    # Create new policy with updated parameters
    def s_encoder(state: np.ndarray) -> np.ndarray:
        """Encode state: A @ psi(state)"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
            emb = psi_net(state_tensor).squeeze(0).cpu().numpy()
            return A.detach().cpu().numpy() @ emb

    new_policy = GreedyPolicy(env, s_encoder, temperature=policy_temperature)

    return psi_net, A, log_lambda, new_policy, metrics


def initialize_encoder_random(state_dim: int, latent_dim: int, hidden_dims: List[int] = [64, 64], device: torch.device = None) -> StateEncoder:
    """
    Initialize state encoder with random weights.

    Args:
        state_dim: Dimension of input state (N)
        latent_dim: Dimension of output embedding (k)
        hidden_dims: List of hidden layer sizes
        device: Device to place the network on (CPU or CUDA)

    Returns:
        Initialized StateEncoder network
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    psi_net = StateEncoder(state_dim, latent_dim, hidden_dims)
    psi_net = psi_net.to(device)

    # Xavier/Kaiming initialization is done by default in PyTorch
    # Can add custom initialization here if needed

    return psi_net
