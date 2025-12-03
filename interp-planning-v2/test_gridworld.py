"""Test script for GridworldProblem environment."""

import numpy as np
from problems import GridworldProblem
from policies import RandomPolicy, ManhattanPolicy


def test_basic_gridworld():
    """Test basic gridworld functionality."""
    print("=" * 80)
    print("Testing GridworldProblem")
    print("=" * 80)

    # Create a small gridworld
    K, N, O, r = 10, 2, 5, 1.5
    env = GridworldProblem(K=K, N=N, O=O, r=r, seed=42)

    print(f"\nEnvironment: K={K}, N={N}, O={O}, r={r}")
    print(f"State space size: {K}^{N} = {K**N}")
    print(f"Action space size: 2*{N} = {2*N}")
    print(f"\nObstacles:")
    for i, obs in enumerate(env.obstacles):
        print(f"  {i}: {obs}")

    # Reset environment
    state, goal = env.reset()
    print(f"\nStart state: {state}")
    print(f"Goal state: {goal}")
    print(f"Manhattan distance: {env.get_distance(state, goal)}")

    # Check available actions
    available = env.get_available_actions(state)
    print(f"\nAvailable actions from start: {available} (out of {2*N})")

    # Test action execution
    print("\n" + "=" * 80)
    print("Testing Random Policy")
    print("=" * 80)

    policy = RandomPolicy(env, seed=42)
    trajectory = [state.copy()]

    for step in range(20):
        action = policy.get_action(env.current_state, goal)
        if action is None:
            print(f"Step {step}: No valid actions!")
            break

        next_state, reward, done, info = env.step(action)
        trajectory.append(next_state.copy())

        print(f"Step {step}: action={action}, state={next_state}, "
              f"reward={reward:.3f}, done={done}, dist={env.get_distance(next_state, goal)}")

        if done:
            print(f"\nReached goal in {step + 1} steps!")
            break

    print(f"\nTrajectory length: {len(trajectory)}")


def test_manhattan_policy():
    """Test Manhattan distance policy."""
    print("\n" + "=" * 80)
    print("Testing Manhattan Policy")
    print("=" * 80)

    # Create a small gridworld
    K, N, O, r = 10, 2, 3, 1.0
    env = GridworldProblem(K=K, N=N, O=O, r=r, seed=123)

    print(f"\nEnvironment: K={K}, N={N}, O={O}, r={r}")

    print("Obstacles:", env.obstacles)

    # Reset environment
    state, goal = env.reset()
    print(f"Start state: {state}")
    print(f"Goal state: {goal}")
    print(f"Initial distance: {env.get_distance(state, goal)}")

    # Test Manhattan policy
    policy = ManhattanPolicy(env, temperature=1, seed=123)

    for step in range(30):
        action = policy.get_action(env.current_state, goal)
        if action is None:
            print(f"Step {step}: No valid actions!")
            break

        next_state, reward, done, info = env.step(action)

        print(f"Step {step}: action={action}, state={next_state}, "
              f"dist={env.get_distance(next_state, goal)}")

        if done:
            print(f"\nReached goal in {step + 1} steps!")
            break


def test_state_indexing():
    """Test state to index conversion."""
    print("\n" + "=" * 80)
    print("Testing State Indexing")
    print("=" * 80)

    K, N = 5, 3
    env = GridworldProblem(K=K, N=N, O=0, r=0.0, seed=42)

    print(f"\nEnvironment: K={K}, N={N}")
    print(f"Total states: {K**N}")

    # Test a few conversions
    test_states = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([4, 4, 4]),
        np.array([2, 3, 1]),
    ]

    print("\nState <-> Index conversions:")
    for state in test_states:
        index = env.state_to_index(state)
        reconstructed = env.index_to_state(index)
        match = np.array_equal(state, reconstructed)
        print(f"  {state} -> {index} -> {reconstructed} (match: {match})")


if __name__ == "__main__":
    test_basic_gridworld()
    test_manhattan_policy()
    test_state_indexing()
