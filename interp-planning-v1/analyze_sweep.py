#!/usr/bin/env python3
"""
Analyze sweep results and find best hyperparameter combinations.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import OmegaConf


def find_sweep_runs(sweep_dir):
    """Find all completed runs in a sweep directory."""
    sweep_path = Path(sweep_dir)

    if not sweep_path.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return []

    # Find all subdirectories (each is a run)
    runs = []
    for run_dir in sweep_path.iterdir():
        if run_dir.is_dir():
            # Check if it has a .hydra directory (indicates completed run)
            hydra_dir = run_dir / ".hydra"
            if hydra_dir.exists():
                runs.append(run_dir)

    return runs


def extract_results(run_dir):
    """Extract configuration and results from a single run."""
    run_path = Path(run_dir)

    # Load config
    config_file = run_path / ".hydra" / "config.yaml"
    if not config_file.exists():
        print(f"Warning: No config found in {run_dir}")
        return None

    config = OmegaConf.load(config_file)

    # Try to load results (might be in different formats)
    result = {
        'run_dir': str(run_path),
        'k': config.model.k,
        'waypoint_type': config.planner.waypoint_type,
        'learning_rate': config.training.learning_rate,
        'l2_reg': config.training.l2_reg,
        'use_infonce': config.training.use_infonce,
        'num_epochs': config.training.num_epochs,
    }

    # Look for output files that might contain results
    # Check for main.log or pipeline output
    log_files = list(run_path.glob("*.log"))

    # Try to parse metrics from log files
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()

                # Parse evaluation results
                if "Success Rate:" in content:
                    for line in content.split('\n'):
                        if "Success Rate:" in line:
                            result['success_rate'] = float(line.split(':')[1].strip().rstrip('%')) / 100
                        elif "Mean Path Length:" in line:
                            result['mean_path_length'] = float(line.split(':')[1].strip())
                        elif "Mean Optimal Length:" in line:
                            result['mean_optimal_length'] = float(line.split(':')[1].strip())
                        elif "Efficiency:" in line:
                            result['efficiency'] = float(line.split(':')[1].strip().rstrip('%')) / 100
                        elif "Total Time:" in line:
                            # Parse "Total Time: 123.45s (Train: 100.00s, Eval: 23.45s)"
                            parts = line.split(':')[1].split('(')[0].strip().rstrip('s')
                            result['total_time'] = float(parts)
        except Exception as e:
            print(f"Warning: Error parsing {log_file}: {e}")
            continue

    return result


def parse_sweep_output(sweep_dir):
    """Parse results from the main sweep output file."""
    sweep_path = Path(sweep_dir)

    # Look for sweep.out file (created by run_sweep.slurm)
    output_file = sweep_path / "sweep.out"
    if not output_file.exists():
        # Try to find SLURM output file
        job_id = sweep_path.name.split('job')[-1]
        slurm_file = Path(f"/scratch/gpfs/TSILVER/de7281/interp_planning/slurm_logs/sweep_{job_id}.out")
        if slurm_file.exists():
            output_file = slurm_file
        else:
            print(f"Could not find output file: {output_file} or {slurm_file}")
            return []

    print(f"Parsing output file: {output_file}")

    results = []
    current_config = None
    current_results = {}

    with open(output_file, 'r') as f:
        for line in f:
            # Detect start of new run
            if '[HYDRA]' in line and '#' in line and ':' in line:
                # Save previous run if it has results
                if current_config is not None and 'success_rate' in current_results:
                    results.append({
                        'config': current_config,
                        **current_results
                    })

                # Extract configuration line
                # Format: [timestamp][HYDRA] 	#N : config params here
                parts = line.split(':', 2)  # Split on first 2 colons
                if len(parts) >= 3:
                    current_config = parts[2].strip()
                    current_results = {}

            # Extract results from output
            elif current_config is not None:
                if "Success Rate:" in line:
                    current_results['success_rate'] = float(line.split(':')[1].strip().rstrip('%')) / 100
                elif "Mean Path Length:" in line:
                    current_results['mean_path_length'] = float(line.split(':')[1].strip())
                elif "Mean Optimal Length:" in line:
                    current_results['mean_optimal_length'] = float(line.split(':')[1].strip())
                elif "Efficiency:" in line:
                    current_results['efficiency'] = float(line.split(':')[1].strip().rstrip('%')) / 100
                elif "MSE Path Length:" in line:
                    current_results['mse_path_length'] = float(line.split(':')[1].strip())
                elif "Total Time:" in line:
                    # Parse "Total Time: 123.45s (Train: 100.00s, Eval: 23.45s)"
                    parts = line.split(':')[1].split('(')[0].strip().rstrip('s')
                    current_results['total_time'] = float(parts)

    # Don't forget the last run
    if current_config is not None and 'success_rate' in current_results:
        results.append({
            'config': current_config,
            **current_results
        })

    return results


def analyze_sweep(sweep_dir):
    """Analyze all runs in a sweep and create summary."""
    print(f"Analyzing sweep directory: {sweep_dir}\n")

    # Parse results from sweep output file
    results = parse_sweep_output(sweep_dir)

    if len(results) == 0:
        print("No results could be extracted from sweep output!")
        return None

    # Create DataFrame
    df = pd.DataFrame(results)

    print(f"Successfully extracted results from {len(results)} runs\n")
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Overall statistics
    print(f"\nSuccess Rate: {df['success_rate'].mean():.2%} ± {df['success_rate'].std():.2%}")
    print(f"Mean Path Length: {df['mean_path_length'].mean():.2f} ± {df['mean_path_length'].std():.2f}")
    if 'mse_path_length' in df.columns:
        print(f"MSE Path Length: {df['mse_path_length'].mean():.2f} ± {df['mse_path_length'].std():.2f}")
    print(f"Efficiency: {df['efficiency'].mean():.2%} ± {df['efficiency'].std():.2%}")

    print("\n" + "=" * 80)
    print("BEST 10 CONFIGURATIONS (by MSE Path Length - LOWER IS BETTER)")
    print("=" * 80)

    # Sort by MSE (lower is better)
    if 'mse_path_length' in df.columns:
        best_mse = df.nsmallest(10, 'mse_path_length')

        for idx, row in best_mse.iterrows():
            print(f"\n#{best_mse.index.get_loc(idx) + 1}")
            print(f"  Config: {row['config']}")
            print(f"  MSE Path Length: {row['mse_path_length']:.2f}")
            print(f"  Mean Path Length: {row['mean_path_length']:.2f}")
            print(f"  Success Rate: {row['success_rate']:.2%}")
            if 'total_time' in row:
                print(f"  Total Time: {row['total_time']:.1f}s")
    else:
        # Fallback to mean path length if MSE not available
        best_path_length = df.nsmallest(10, 'mean_path_length')

        for idx, row in best_path_length.iterrows():
            print(f"\n#{best_path_length.index.get_loc(idx) + 1}")
            print(f"  Config: {row['config']}")
            print(f"  Mean Path Length: {row['mean_path_length']:.2f}")
            print(f"  Success Rate: {row['success_rate']:.2%}")
            if 'total_time' in row:
                print(f"  Total Time: {row['total_time']:.1f}s")

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by Success Rate)")
    print("=" * 80)

    # Sort by success rate
    top_configs = df.nlargest(10, 'success_rate')

    for idx, row in top_configs.iterrows():
        print(f"\n#{top_configs.index.get_loc(idx) + 1}")
        print(f"  Config: {row['config']}")
        print(f"  Success Rate: {row['success_rate']:.2%}")
        if 'mse_path_length' in row:
            print(f"  MSE Path Length: {row['mse_path_length']:.2f}")
        print(f"  Mean Path Length: {row['mean_path_length']:.2f}")
        if 'total_time' in row:
            print(f"  Total Time: {row['total_time']:.1f}s")

    # Save results to CSV
    output_file = Path(sweep_dir) / "sweep_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_sweep.py <sweep_directory>")
        print("\nExample:")
        print("  python analyze_sweep.py /scratch/gpfs/TSILVER/de7281/interp_planning/sweeps/sweep_2024-11-23_job123456")
        sys.exit(1)

    sweep_dir = sys.argv[1]
    df = analyze_sweep(sweep_dir)
