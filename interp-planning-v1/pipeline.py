import numpy as np
import torch
import torch.nn as nn
from problems import GraphProblem
from policies import RandomPolicy, GreedyPolicy
from planning import WaypointPlanner
from learning import learning_epoch, initialize_with_mds
from collections import deque
import networkx as nx
import copy
from typing import Tuple, List
from omegaconf import DictConfig
import hydra
import time


def train(config: DictConfig) -> Tuple[np.ndarray, np.ndarray, GraphProblem, float]:
    """
    Train the interpolative planner using contrastive learning.

    Args:
        config: Hydra configuration object

    Returns:
        Tuple of (A_np, psi_np, env, train_time) - learned parameters, environment, and training time
    """
    start_time = time.time()

    # Create environment
    env = GraphProblem(config.env.n, config.env.p, seed=config.seed)

    # Initialize embeddings with MDS
    psi = initialize_with_mds(env, config.model.k, gamma=config.env.gamma)

    # Initialize learnable parameters
    psi_torch = nn.Parameter(torch.tensor(psi, dtype=torch.float32, requires_grad=True))
    A_torch = nn.Parameter(torch.eye(config.model.k, requires_grad=True))

    # Initialize policy (random at first)
    policy = RandomPolicy(env, seed=config.seed)

    # Create N_T waypoint planners with random start/goal
    planners = []
    for i in range(config.training.N_T):
        start = np.random.randint(0, config.env.n)
        goal = np.random.randint(0, config.env.n)
        while goal == start:
            goal = np.random.randint(0, config.env.n)

        planner = WaypointPlanner(
            start=start,
            goal=goal,
            env=copy.deepcopy(env),
            policy=policy,
            psi=lambda x: psi_torch.detach().numpy()[x, :],
            A=A_torch.detach().numpy(),
            waypoint_type=config.planner.waypoint_type,
            max_waypoints=config.planner.max_waypoints,
            c=config.planner.c,
            T=config.planner.temperature
        )
        planners.append(planner)

    print(f"Created {len(planners)} planners")

    # Initialize replay buffer
    replay_buffer = deque(maxlen=config.training.buffer_size)

    # Training loop
    losses = []

    for epoch in range(config.training.num_epochs):
        # Step each planner forward and collect (current_state, future_state) pairs
        for planner in planners:
            # Step planner
            action, waypoint, done = planner.step()

            # Sample (current_state, future_state) pair from trajectory
            if len(planner.trajectory) >= 2:
                # Randomly sample a position in the trajectory (not the last one)
                current_idx = np.random.randint(0, len(planner.trajectory) - 1)
                current_state = planner.trajectory[current_idx]

                # Sample future state from trajectory using geometric distribution
                future_state = planner.sample_future_state(current_idx, p=config.env.gamma)

                if future_state is not None:
                    # Store (current_state, future_state) pair
                    replay_buffer.append((current_state, future_state))

            # Reset if done
            if done:
                start = np.random.randint(0, config.env.n)
                goal = np.random.randint(0, config.env.n)
                while goal == start:
                    goal = np.random.randint(0, config.env.n)
                planner.reset(start, goal)

        # Learning step
        if (epoch > 0 and
            epoch % config.training.learn_frequency == 0 and
            len(replay_buffer) >= config.training.batch_size):

            # Sample batch
            indices = np.random.choice(
                len(replay_buffer),
                size=config.training.batch_size,
                replace=False
            )
            batch = [replay_buffer[i] for i in indices]

            # Extract current and future states
            current_states = [s for s, _ in batch]
            future_states = [s_plus for _, s_plus in batch]

            # Run learning
            psi_torch, A_torch, policy, loss_val = learning_epoch(
                psi_torch, A_torch, current_states, future_states, env,
                learning_rate=config.training.learning_rate,
                iters_per_epoch=config.training.iters_per_epoch,
                policy_temperature=config.planner.temperature,
                l2_reg=config.training.l2_reg,
                use_infonce=config.training.use_infonce
            )

            # Update planners
            psi_np = psi_torch.detach().numpy()
            A_np = A_torch.detach().numpy()

            for planner in planners:
                planner.policy = policy
                planner.psi = lambda x: psi_np[x, :]
                planner.A = A_np

            losses.append(loss_val)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Buffer size = {len(replay_buffer)}, Loss = {loss_val:.4f}")

    train_time = time.time() - start_time
    print(f"Training complete! Time: {train_time:.2f}s")

    # Return final parameters
    A_np = A_torch.detach().numpy()
    psi_np = psi_torch.detach().numpy()

    return A_np, psi_np, env, train_time


def evaluate(A_np: np.ndarray,
             psi_np: np.ndarray,
             env: GraphProblem,
             config: DictConfig) -> dict:
    """
    Evaluate the learned policy on random start/goal pairs.

    Args:
        A_np: Learned transformation matrix
        psi_np: Learned state embeddings
        env: GraphProblem environment
        config: Hydra configuration object

    Returns:
        Dictionary with evaluation metrics
    """
    eval_start_time = time.time()

    def f_encoder(x: int) -> np.ndarray:
        return psi_np[x, :]

    def s_encoder(x: int) -> np.ndarray:
        return A_np @ f_encoder(x)

    test_policy = GreedyPolicy(
        copy.deepcopy(env),
        s_encoder=s_encoder,
        temperature=config.eval.temperature
    )

    # Test the learned policy on many start/goal pairs
    num_test_pairs = config.eval.num_test_pairs
    num_trials_per_pair = config.eval.num_trials_per_pair

    all_lens = []
    all_true_lens = []
    success_count = 0
    total_trials = 0

    for _ in range(num_test_pairs):
        start = np.random.randint(0, config.env.n)
        goal = np.random.randint(0, config.env.n)
        while goal == start:
            goal = np.random.randint(0, config.env.n)

        # Get optimal path length
        true_len = len(nx.shortest_path(env.graph, source=start, target=goal)) - 1
        all_true_lens.append(true_len)

        trial_lens = []
        for _ in range(num_trials_per_pair):
            test_trajectory = [start]
            max_test_steps = config.eval.max_steps

            for _ in range(max_test_steps):
                action = test_policy.get_action(test_trajectory[-1], f_encoder(goal))
                if action is not None:
                    test_trajectory.append(action)
                if action == goal:
                    break

            reached_goal = test_trajectory[-1] == goal
            success_count += int(reached_goal)
            total_trials += 1
            trial_lens.append(len(test_trajectory) - 1)  # -1 to get number of steps

        all_lens.append(trial_lens)

    # Compute statistics
    success_rate = success_count / total_trials
    all_lens_flat = [l for trial in all_lens for l in trial]
    mean_len = np.mean(all_lens_flat)
    mean_true_len = np.mean(all_true_lens)

    # Compute efficiency (only for successful trajectories)
    successful_lens = [l for l, true_l in zip(all_lens_flat, all_true_lens * num_trials_per_pair) if l < config.eval.max_steps]
    if successful_lens:
        efficiency = np.mean([true_l / l for l, true_l in zip(successful_lens, all_true_lens * num_trials_per_pair) if l < config.eval.max_steps])
    else:
        efficiency = 0.0

    # Compute MSE between path length and optimal path length (includes all trials, even failed ones)
    all_true_lens_expanded = all_true_lens * num_trials_per_pair
    mse_path_length = np.mean([(l - true_l) ** 2 for l, true_l in zip(all_lens_flat, all_true_lens_expanded)])

    eval_time = time.time() - eval_start_time

    results = {
        'success_rate': success_rate,
        'mean_path_length': mean_len,
        'mean_optimal_length': mean_true_len,
        'efficiency': efficiency,
        'mse_path_length': mse_path_length,
        'all_lens': all_lens,
        'all_true_lens': all_true_lens,
        'eval_time': eval_time
    }

    print(f"\nEvaluation Results:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Mean Path Length: {mean_len:.2f}")
    print(f"Mean Optimal Length: {mean_true_len:.2f}")
    print(f"Efficiency: {efficiency:.2%}")
    print(f"MSE Path Length: {mse_path_length:.2f}")
    print(f"Evaluation Time: {eval_time:.2f}s")

    return results


def train_and_evaluate(config: DictConfig) -> dict:
    """
    Train and evaluate the interpolative planner.

    Args:
        config: Hydra configuration object

    Returns:
        Dictionary with evaluation results (includes train_time and eval_time)
    """
    print("Starting training...")
    A_np, psi_np, env, train_time = train(config)

    print("\nStarting evaluation...")
    results = evaluate(A_np, psi_np, env, config)

    # Add training time to results
    results['train_time'] = train_time
    results['total_time'] = train_time + results['eval_time']

    print(f"\nTotal Time: {results['total_time']:.2f}s (Train: {train_time:.2f}s, Eval: {results['eval_time']:.2f}s)")

    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    """Main entry point for Hydra."""
    results = train_and_evaluate(config)
    return results


if __name__ == "__main__":
    main()
