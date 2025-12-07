import numpy as np
import torch
import torch.nn as nn
from problems import GridworldProblem
from policies import RandomPolicy, GreedyPolicy
from planning import WaypointPlanner
from learning import learning_epoch, initialize_encoder_random, StateEncoder
from collections import deque
import copy
from typing import Tuple, List, Optional
from omegaconf import DictConfig
import hydra
import time


def evaluate_with_policy(policy, psi_net: StateEncoder, A_np: np.ndarray,
                         env: GridworldProblem, config: DictConfig, num_pairs: int = 10) -> dict:
    """
    Evaluate a policy on random start/goal pairs.

    Args:
        policy: The policy to evaluate (GreedyPolicy or RandomPolicy)
        psi_net: Neural network encoder for computing embeddings
        A_np: Transformation matrix
        env: GridworldProblem environment
        config: Hydra configuration object
        num_pairs: Number of start/goal pairs to test

    Returns:
        Dictionary with success_rate and mean_path_length
    """
    success_count = 0
    total_trials = 0
    all_lens = []

    for _ in range(num_pairs):
        start_state, goal_state = env.reset()

        # Single trial per pair (for faster evaluation during training)
        env.reset(start_state, goal_state)
        trajectory = [start_state.copy()]
        max_steps = config.eval.max_steps

        for _ in range(max_steps):
            current_state = env.current_state

            # Compute goal embedding using psi_net
            with torch.no_grad():
                goal_tensor = torch.from_numpy(goal_state.astype(np.float32)).unsqueeze(0)
                goal_emb = psi_net(goal_tensor).squeeze(0).numpy()

            # Get action from policy
            action = policy.get_action(current_state, goal_emb)

            if action is None:
                break

            next_state, reward, done, info = env.step(action)
            trajectory.append(next_state.copy())

            if done:
                break

        reached_goal = np.array_equal(trajectory[-1], goal_state)
        success_count += int(reached_goal)
        total_trials += 1
        all_lens.append(len(trajectory) - 1)

    success_rate = success_count / total_trials if total_trials > 0 else 0.0
    mean_path_length = np.mean(all_lens) if all_lens else 0.0

    return {
        'success_rate': success_rate,
        'mean_path_length': mean_path_length
    }


def train(config: DictConfig) -> Tuple[StateEncoder, np.ndarray, np.ndarray, GridworldProblem, float, dict]:
    """
    Train the interpolative planner using contrastive learning.

    Args:
        config: Hydra configuration object

    Returns:
        Tuple of (psi_net, A_np, log_lambda_np, env, train_time, training_history)
    """
    start_time = time.time()

    # Initialize training history tracking
    training_history = {
        'epochs': [],
        'loss': [],
        'accuracy': [],
        'l2': [],
        'learning_rate': [],
        'temperature': [],
        'success_rate': [],
        'mean_path_length': [],
        'eval_epochs': []
    }

    # Create environment
    env = GridworldProblem(
        K=config.env.K,
        N=config.env.N,
        O=config.env.O,
        r=config.env.r,
        seed=config.seed
    )

    # Initialize neural network encoder
    psi_net = initialize_encoder_random(
        state_dim=config.env.N,
        latent_dim=config.model.k,
        hidden_dims=config.model.hidden_dims
    )

    # Initialize learnable parameters
    A_torch = nn.Parameter(torch.eye(config.model.k, requires_grad=True))
    log_lambda = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    # Initialize policy (random at first)
    policy = RandomPolicy(env, seed=config.seed)

    # Create N_T waypoint planners with random start/goal
    planners = []
    for i in range(config.training.N_T):
        start_state, goal_state = env.reset()

        # Create psi function that uses the network
        def make_psi(net):
            def psi_fn(state):
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
                    return net(state_tensor).squeeze(0).numpy()
            return psi_fn

        planner = WaypointPlanner(
            start=start_state,
            goal=goal_state,
            env=copy.deepcopy(env),
            policy=policy,
            psi=make_psi(psi_net),
            A=A_torch.detach().numpy(),
            waypoint_type=config.planner.waypoint_type,
            max_waypoints=config.planner.max_waypoints,
            state_buffer=[],  # Will be updated from replay buffer
            M=config.planner.M,
            c=config.training.c_target,  # Use same c as L2 constraint
            eps=config.planner.eps,
            T=config.planner.waypoint_temp,
            num_gmm_comps=config.planner.num_gmm_comps,
            num_gmm_iters=config.planner.num_gmm_iters
        )
        planners.append(planner)

    print(f"Created {len(planners)} planners")

    # Initialize replay buffer (store full trajectories)
    replay_buffer = deque(maxlen=config.training.buffer_size)

    # Helper: sample a future state from a trajectory using geometric distribution
    def sample_future_from_traj(traj: List[np.ndarray], current_idx: int, p: float = 0.9) -> Optional[np.ndarray]:
        max_offset = len(traj) - current_idx - 1
        if max_offset <= 0:
            return None
        offset = np.random.geometric(p)
        offset = min(offset, max_offset)
        return traj[current_idx + offset].copy()

    # Get scheduling parameters
    initial_lr = config.training.learning_rate
    min_lr = initial_lr * 0.01
    initial_temp = config.training.temperature
    min_temp = config.eval.temperature

    # Training loop
    for epoch in range(config.training.num_epochs):
        # For each planner, roll it forward to produce a full trajectory and store it
        for planner in planners:
            # Roll planner forward from its current state to goal (or until max steps)
            traj, path_len, success = planner.rollout(
                start=planner.current_state,
                goal=planner.goal,
                max_steps=config.training.max_steps,
                num_waypoints=config.planner.max_waypoints
            )

            # Store the full trajectory in the replay buffer if it contains at least two states
            if len(traj) >= 2:
                replay_buffer.append(traj)

            # If the planner finished (reached goal or became done), reset with a new random pair
            if planner.done:
                start_state, goal_state = env.reset()
                planner.reset(start_state, goal_state)

        # Learning step
        if (epoch > 0 and
            epoch % config.training.learn_frequency == 0 and
            len(replay_buffer) >= config.training.batch_size):

            # Compute cosine schedule for learning rate and temperature
            progress = epoch / config.training.num_epochs
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))

            current_lr = min_lr + (initial_lr - min_lr) * cosine_factor
            current_temp = min_temp + (initial_temp - min_temp) * cosine_factor

            # Sample batch of trajectories (with replacement to increase diversity)
            if len(replay_buffer) == 0:
                continue

            indices = np.random.choice(
                len(replay_buffer),
                size=config.training.batch_size,
                replace=True
            )
            traj_batch = [replay_buffer[i] for i in indices]

            # From each trajectory, randomly sample a current state and a geometrically-sampled future state
            pairs = []
            for traj in traj_batch:
                if len(traj) < 2:
                    continue
                current_idx = np.random.randint(0, len(traj) - 1)
                current_state = traj[current_idx]
                future_state = sample_future_from_traj(traj, current_idx, p=config.env.gamma)
                if future_state is not None:
                    pairs.append((current_state.copy(), future_state.copy()))

            if len(pairs) == 0:
                continue

            # If we collected fewer pairs than batch size, learning_epoch will run on smaller batch
            current_states = [s for s, _ in pairs]
            future_states = [s_plus for _, s_plus in pairs]

            # Run learning with scheduled learning rate and temperature
            psi_net, A_torch, log_lambda, policy, metrics = learning_epoch(
                psi_net, A_torch, log_lambda,
                current_states, future_states, env,
                learning_rate=current_lr,
                iters_per_epoch=config.training.iters_per_epoch,
                policy_temperature=current_temp,
                c_target=config.training.c_target
            )

            # Update planners with new policy and parameters
            A_np = A_torch.detach().numpy()

            def make_psi(net):
                def psi_fn(state):
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
                        return net(state_tensor).squeeze(0).numpy()
                return psi_fn

            # Extract all states from replay buffer (each entry is a trajectory)
            state_buffer = []
            for traj in replay_buffer:
                for s in traj:
                    state_buffer.append(s.copy())

            for planner in planners:
                planner.policy = policy
                planner.psi = make_psi(psi_net)
                planner.A = A_np
                planner.state_buffer = state_buffer  # Update shared buffer from replay buffer

            # Track training metrics
            training_history['epochs'].append(epoch)
            training_history['loss'].append(metrics['total_loss'])
            training_history['accuracy'].append(metrics['accuracy'])
            training_history['l2'].append(metrics['l2'])
            training_history['learning_rate'].append(current_lr)
            training_history['temperature'].append(current_temp)

            # Periodic evaluation during training using the current policy
            eval_freq = getattr(config.training, 'eval_frequency', None)
            if eval_freq is not None and epoch % eval_freq == 0 and epoch > 0:
                print(f"\n--- Evaluating at epoch {epoch} ---")

                # Evaluate using the current policy with proper embeddings
                eval_results = evaluate_with_policy(policy, psi_net, A_np, env, config, num_pairs=10)
                training_history['eval_epochs'].append(epoch)
                training_history['success_rate'].append(eval_results['success_rate'])
                training_history['mean_path_length'].append(eval_results['mean_path_length'])

                print(f"Success Rate: {eval_results['success_rate']:.2%}, "
                      f"Mean Path Length: {eval_results['mean_path_length']:.2f}\n")

            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Buffer size = {len(replay_buffer)}, "
                      f"Loss = {metrics['total_loss']:.4f}, "
                      f"Alignment Loss = {metrics['l_align']}, "
                      f"Uniformity Loss = {metrics['l_unif']}, "
                      f"Accuracy = {metrics['accuracy']:.2%}, "
                      f"L2 = {metrics['l2']:.4f}, "
                      f"LR = {current_lr:.6f}, "
                      f"Temp = {current_temp:.3f}")

    train_time = time.time() - start_time
    print(f"Training complete! Time: {train_time:.2f}s")

    # Return final parameters
    A_np = A_torch.detach().numpy()
    log_lambda_np = log_lambda.detach().numpy()

    return psi_net, A_np, log_lambda_np, env, train_time, training_history


def evaluate(psi_net: StateEncoder,
             A_np: np.ndarray,
             env: GridworldProblem,
             config: DictConfig) -> dict:
    """
    Evaluate the learned policy on random start/goal pairs.

    Args:
        psi_net: Learned neural network encoder
        A_np: Learned transformation matrix
        env: GridworldProblem environment
        config: Hydra configuration object

    Returns:
        Dictionary with evaluation metrics
    """
    eval_start_time = time.time()

    def s_encoder(state: np.ndarray) -> np.ndarray:
        """Encode state: A @ psi(state)"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            emb = psi_net(state_tensor).squeeze(0).numpy()
            return A_np @ emb

    test_policy = GreedyPolicy(
        copy.deepcopy(env),
        s_encoder=s_encoder,
        temperature=config.eval.temperature
    )

    # Test the learned policy on many start/goal pairs
    num_test_pairs = config.eval.num_test_pairs
    num_trials_per_pair = config.eval.num_trials_per_pair

    all_lens = []
    all_manhattan_lens = []
    success_count = 0
    total_trials = 0

    for _ in range(num_test_pairs):
        start_state, goal_state = env.reset()

        # Get Manhattan distance as lower bound
        manhattan_len = env.get_distance(start_state, goal_state)
        all_manhattan_lens.append(manhattan_len)

        trial_lens = []
        for _ in range(num_trials_per_pair):
            env.reset(start_state, goal_state)
            test_trajectory = [start_state.copy()]
            max_test_steps = config.eval.max_steps

            for _ in range(max_test_steps):
                current_state = env.current_state

                # Get goal embedding for policy
                with torch.no_grad():
                    goal_tensor = torch.from_numpy(goal_state.astype(np.float32)).unsqueeze(0)
                    goal_emb = psi_net(goal_tensor).squeeze(0).numpy()

                action = test_policy.get_action(current_state, goal_emb)
                if action is None:
                    break

                next_state, reward, done, info = env.step(action)
                test_trajectory.append(next_state.copy())

                if done:
                    break

            reached_goal = np.array_equal(test_trajectory[-1], goal_state)
            success_count += int(reached_goal)
            total_trials += 1
            trial_lens.append(len(test_trajectory) - 1)  # -1 to get number of steps

        all_lens.append(trial_lens)

    # Compute statistics
    success_rate = success_count / total_trials if total_trials > 0 else 0.0
    all_lens_flat = [l for trial in all_lens for l in trial]
    mean_len = np.mean(all_lens_flat) if all_lens_flat else 0.0
    mean_manhattan_len = np.mean(all_manhattan_lens) if all_manhattan_lens else 0.0

    # Compute efficiency (ratio of Manhattan distance to actual path length, only for successful trajectories)
    successful_lens = [l for l, manhattan_l in zip(all_lens_flat, all_manhattan_lens * num_trials_per_pair) if l < config.eval.max_steps]
    if successful_lens:
        efficiency = np.mean([manhattan_l / l for l, manhattan_l in zip(successful_lens, all_manhattan_lens * num_trials_per_pair) if l < config.eval.max_steps])
    else:
        efficiency = 0.0

    # Compute MSE between path length and Manhattan distance (includes all trials, even failed ones)
    all_manhattan_lens_expanded = all_manhattan_lens * num_trials_per_pair
    mse_path_length = np.mean([(l - manhattan_l) ** 2 for l, manhattan_l in zip(all_lens_flat, all_manhattan_lens_expanded)]) if all_lens_flat else 0.0

    eval_time = time.time() - eval_start_time

    results = {
        'success_rate': success_rate,
        'mean_path_length': mean_len,
        'mean_manhattan_length': mean_manhattan_len,
        'efficiency': efficiency,
        'mse_path_length': mse_path_length,
        'all_lens': all_lens,
        'all_manhattan_lens': all_manhattan_lens,
        'eval_time': eval_time
    }

    print(f"\nEvaluation Results:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Mean Path Length: {mean_len:.2f}")
    print(f"Mean Manhattan Length: {mean_manhattan_len:.2f}")
    print(f"Efficiency: {efficiency:.2%}")
    print(f"MSE Path Length: {mse_path_length:.2f}")
    print(f"Evaluation Time: {eval_time:.2f}s")

    return results


def train_and_evaluate(config: DictConfig, return_models: bool = False):
    """
    Train and evaluate the interpolative planner.

    Args:
        config: Hydra configuration object
        return_models: If True, returns (results, models) tuple where models contains psi_net, A_np, log_lambda_np, env

    Returns:
        Dictionary with evaluation results (includes train_time, eval_time, and training_history)
        OR tuple of (results, models) if return_models=True
    """
    print("Starting training...")
    psi_net, A_np, log_lambda_np, env, train_time, training_history = train(config)

    print("\nStarting evaluation...")
    results = evaluate(psi_net, A_np, env, config)

    # Add training time and history to results
    results['train_time'] = train_time
    results['total_time'] = train_time + results['eval_time']
    results['training_history'] = training_history

    print(f"\nTotal Time: {results['total_time']:.2f}s (Train: {train_time:.2f}s, Eval: {results['eval_time']:.2f}s)")

    if return_models:
        models = {
            'psi_net': psi_net,
            'A_np': A_np,
            'log_lambda_np': log_lambda_np,
            'env': env
        }
        return results, models
    else:
        return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    """Main entry point for Hydra."""
    results = train_and_evaluate(config)
    return results


if __name__ == "__main__":
    main()
