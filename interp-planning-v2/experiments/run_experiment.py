#!/usr/bin/env python3
"""
Run experiment with model saving enabled.
Uses Hydra for configuration, just like pipeline.py but saves models.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pickle
from omegaconf import DictConfig, OmegaConf
import hydra
from pipeline import train_and_evaluate


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    """Main entry point for Hydra with model saving."""

    print("="*80)
    print("Starting Experiment")
    print("="*80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(config))

    # Run training and evaluation (get models too)
    results, models = train_and_evaluate(config, return_models=True)

    # Get current working directory (Hydra output dir)
    import os
    output_dir = Path(os.getcwd())

    # Save results
    results_file = output_dir / "results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump({
            'config': OmegaConf.to_container(config),
            'results': results
        }, f)

    # Save model
    model_file = output_dir / "model.pt"
    torch.save({
        'psi_net_state_dict': models['psi_net'].state_dict(),
        'A_np': models['A_np'],
        'log_lambda_np': models['log_lambda_np'],
        'config': OmegaConf.to_container(config)
    }, model_file)

    # Save training history
    history_file = output_dir / "training_history.pkl"
    with open(history_file, 'wb') as f:
        pickle.dump(results['training_history'], f)

    print(f"\nResults saved to: {results_file}")
    print(f"Model saved to: {model_file}")
    print(f"Training history saved to: {history_file}")

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Waypoint Type: {config.planner.waypoint_type}")
    print(f"Seed: {config.seed}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Mean Path Length: {results['mean_path_length']:.2f}")
    print(f"Mean Manhattan Length: {results['mean_manhattan_length']:.2f}")
    print(f"Efficiency: {results['efficiency']:.2%}")
    print(f"Train Time: {results['train_time']:.2f}s")
    print(f"Eval Time: {results['eval_time']:.2f}s")
    print("="*80)

    return results


if __name__ == "__main__":
    main()
