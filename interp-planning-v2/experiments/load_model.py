#!/usr/bin/env python3
"""
Helper script to load saved models for further testing.
"""

import torch
import pickle
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from learning import StateEncoder, initialize_encoder_random
from problems import GridworldProblem
from policies import GreedyPolicy
from omegaconf import OmegaConf
import copy


def load_model(model_path, verbose=True):
    """
    Load a saved model from a .pt file.

    Args:
        model_path: Path to the .pt model file

    Returns:
        Dictionary containing:
        - psi_net: The neural network encoder
        - A_np: Transformation matrix
        - log_lambda_np: Log lambda parameter
        - config: Configuration used for training
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the checkpoint
    checkpoint = torch.load(model_path, weights_only=False)

    # Extract config
    config = OmegaConf.create(checkpoint['config'])

    # Recreate the neural network with the same architecture
    psi_net = initialize_encoder_random(
        state_dim=config.env.N,
        latent_dim=config.model.k,
        hidden_dims=config.model.hidden_dims
    )

    # Load the trained weights
    psi_net.load_state_dict(checkpoint['psi_net_state_dict'])
    psi_net.eval()  # Set to evaluation mode

    # Load other parameters
    A_np = checkpoint['A_np']
    log_lambda_np = checkpoint['log_lambda_np']
    if verbose:
        print(f"Successfully loaded model from: {model_path}")
        print(f"Configuration:")
        print(f"  Environment: K={config.env.K}, N={config.env.N}, O={config.env.O}")
        print(f"  Model: latent_dim={config.model.k}, hidden_dims={config.model.hidden_dims}")
        print(f"  Waypoint type: {config.planner.waypoint_type}")
        print(f"  Seed: {config.seed}")

    return {
        'psi_net': psi_net,
        'A_np': A_np,
        'log_lambda_np': log_lambda_np,
        'config': config
    }


def load_results(results_path):
    """
    Load results from a .pkl file.

    Args:
        results_path: Path to the .pkl results file

    Returns:
        Dictionary containing results and config
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Successfully loaded results from: {results_path}")
    print(f"Waypoint type: {data['waypoint_type']}")
    print(f"Seed: {data['seed']}")
    print(f"Success rate: {data['results']['success_rate']:.2%}")
    print(f"Mean path length: {data['results']['mean_path_length']:.2f}")

    return data


def create_policy_from_model(model_dict, temperature=0.1):
    """
    Create a GreedyPolicy from a loaded model.

    Args:
        model_dict: Dictionary returned by load_model()
        temperature: Temperature for policy (lower = more greedy)

    Returns:
        GreedyPolicy instance
    """
    psi_net = model_dict['psi_net']
    A_np = model_dict['A_np']
    config = model_dict['config']

    # Create environment (for policy initialization)
    env = GridworldProblem(
        K=config.env.K,
        N=config.env.N,
        O=config.env.O,
        r=config.env.r,
        seed=config.seed
    )

    # Create state encoder function
    def s_encoder(state: np.ndarray) -> np.ndarray:
        """Encode state: A @ psi(state)"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            emb = psi_net(state_tensor).squeeze(0).numpy()
            return A_np @ emb

    # Create policy
    policy = GreedyPolicy(
        copy.deepcopy(env),
        s_encoder=s_encoder,
        temperature=temperature
    )

    return policy, env


def test_model(model_path, num_test_pairs=10, max_steps=300, temperature=0.1):
    """
    Quick test of a loaded model.

    Args:
        model_path: Path to the .pt model file
        num_test_pairs: Number of start/goal pairs to test
        max_steps: Maximum steps per episode
        temperature: Policy temperature
    """
    # Load model
    model_dict = load_model(model_path)

    # Create policy
    policy, env = create_policy_from_model(model_dict, temperature=temperature)

    psi_net = model_dict['psi_net']

    # Run test episodes
    success_count = 0
    path_lengths = []

    print(f"\nTesting model on {num_test_pairs} episodes...")

    for i in range(num_test_pairs):
        start_state, goal_state = env.reset()
        trajectory = [start_state.copy()]

        for _ in range(max_steps):
            current_state = env.current_state

            # Get goal embedding
            with torch.no_grad():
                goal_tensor = torch.from_numpy(goal_state.astype(np.float32)).unsqueeze(0)
                goal_emb = psi_net(goal_tensor).squeeze(0).numpy()

            action = policy.get_action(current_state, goal_emb)
            if action is None:
                break

            next_state, reward, done, info = env.step(action)
            trajectory.append(next_state.copy())

            if done:
                break

        reached_goal = np.array_equal(trajectory[-1], goal_state)
        success_count += reached_goal
        path_lengths.append(len(trajectory) - 1)

        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{num_test_pairs} episodes")

    success_rate = success_count / num_test_pairs
    mean_path_length = np.mean(path_lengths)

    print(f"\nTest Results:")
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Mean Path Length: {mean_path_length:.2f} steps")

    return {
        'success_rate': success_rate,
        'mean_path_length': mean_path_length,
        'path_lengths': path_lengths
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Load and test a saved model')
    parser.add_argument('model_path', type=str, help='Path to the .pt model file')
    parser.add_argument('--num_test_pairs', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--max_steps', type=int, default=300, help='Maximum steps per episode')
    parser.add_argument('--temperature', type=float, default=0.1, help='Policy temperature')
    args = parser.parse_args()

    test_model(args.model_path, args.num_test_pairs, args.max_steps, args.temperature)
