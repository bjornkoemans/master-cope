#!/usr/bin/env python3
"""
Hyperparameter Search
Automates hyperparameter optimization using grid or random search.
Creates config files, trains models, and compares results.
"""

import argparse
import os
import sys
import json
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from itertools import product
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.display import print_colored


def create_config(base_config, params, output_path):
    """
    Create a config file with specific hyperparameters.

    Args:
        base_config: Base configuration dictionary
        params: Dictionary of hyperparameters to override
        output_path: Path to save config file
    """
    config = base_config.copy()

    # Update nested config
    for key, value in params.items():
        if '.' in key:
            # Handle nested keys (e.g., "learning.lr_actor")
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value

    # Save config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_path


def grid_search(param_grid):
    """
    Generate all combinations for grid search.

    Args:
        param_grid: Dictionary mapping parameter names to lists of values

    Returns:
        List of parameter combinations
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # Generate all combinations
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def random_search(param_distributions, n_trials):
    """
    Generate random combinations for random search.

    Args:
        param_distributions: Dictionary mapping parameter names to (min, max) tuples or lists
        n_trials: Number of random combinations to generate

    Returns:
        List of parameter combinations
    """
    combinations = []

    for _ in range(n_trials):
        combo = {}
        for key, dist in param_distributions.items():
            if isinstance(dist, tuple) and len(dist) == 2:
                # Continuous range (min, max)
                combo[key] = random.uniform(dist[0], dist[1])
            elif isinstance(dist, list):
                # Discrete choices
                combo[key] = random.choice(dist)
            else:
                raise ValueError(f"Invalid distribution for {key}: {dist}")
        combinations.append(combo)

    return combinations


def run_experiment(config_path, data_file, distributions_file, output_dir, name):
    """
    Run a single training experiment.

    Args:
        config_path: Path to config file
        data_file: Path to preprocessed data
        distributions_file: Path to fitted distributions
        output_dir: Output directory for experiment
        name: Experiment name

    Returns:
        Tuple of (success, model_path)
    """
    cmd = [
        "python3",
        "scripts/3_train.py",
        "--data", data_file,
        "--distributions", distributions_file,
        "--config", config_path,
        "--output", output_dir,
        "--name", name
    ]

    print_colored(f"Running: {' '.join(cmd)}", "blue")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        model_path = os.path.join(output_dir, name, "models")
        return True, model_path
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Training failed: {e}", "red")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False, None


def hyperparameter_search(
    data_file,
    distributions_file,
    search_type="grid",
    param_grid=None,
    param_distributions=None,
    n_trials=10,
    base_config_path=None,
    output_dir=None,
    parallel=False
):
    """
    Perform hyperparameter search.

    Args:
        data_file: Path to preprocessed data
        distributions_file: Path to fitted distributions
        search_type: "grid" or "random"
        param_grid: Parameter grid for grid search
        param_distributions: Parameter distributions for random search
        n_trials: Number of trials for random search
        base_config_path: Path to base config file
        output_dir: Output directory for search results
        parallel: Whether to run experiments in parallel (not implemented)
    """
    print_colored("="*60, "yellow")
    print_colored("HYPERPARAMETER SEARCH", "yellow")
    print_colored("="*60, "yellow")

    # Load base config
    if base_config_path is None:
        base_config_path = "configs/default.yaml"

    print_colored(f"\n1. Loading base config from {base_config_path}...", "blue")
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    print_colored("   Base config loaded", "green")

    # Generate parameter combinations
    print_colored(f"\n2. Generating parameter combinations ({search_type} search)...", "blue")

    if search_type == "grid":
        if param_grid is None:
            raise ValueError("param_grid must be provided for grid search")
        combinations = grid_search(param_grid)
    elif search_type == "random":
        if param_distributions is None:
            raise ValueError("param_distributions must be provided for random search")
        combinations = random_search(param_distributions, n_trials)
    else:
        raise ValueError(f"Invalid search_type: {search_type}")

    print_colored(f"   Generated {len(combinations)} combinations", "green")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"experiments/hp_search_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    configs_dir = os.path.join(output_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    print_colored(f"\n3. Creating config files in {configs_dir}...", "blue")

    # Create config files
    config_paths = []
    for i, params in enumerate(combinations):
        config_name = f"config_{i+1:03d}.yaml"
        config_path = os.path.join(configs_dir, config_name)
        create_config(base_config, params, config_path)
        config_paths.append((config_path, params))

    print_colored(f"   Created {len(config_paths)} config files", "green")

    # Run experiments
    print_colored(f"\n4. Running {len(combinations)} experiments...", "blue")
    print_colored("="*60, "yellow")

    if parallel:
        print_colored("⚠️  Parallel execution not yet implemented. Running sequentially.", "yellow")

    results = []
    successful_models = []

    for i, (config_path, params) in enumerate(config_paths):
        print_colored(f"\n[{i+1}/{len(combinations)}] Experiment {i+1}", "cyan")
        print_colored(f"   Parameters: {params}", "blue")

        exp_name = f"exp_{i+1:03d}"
        success, model_path = run_experiment(
            config_path=config_path,
            data_file=data_file,
            distributions_file=distributions_file,
            output_dir=output_dir,
            name=exp_name
        )

        result = {
            'experiment_id': i + 1,
            'experiment_name': exp_name,
            'config_path': config_path,
            'parameters': params,
            'success': success,
            'model_path': model_path
        }
        results.append(result)

        if success:
            successful_models.append(model_path)
            print_colored(f"   ✓ Training completed", "green")
        else:
            print_colored(f"   ✗ Training failed", "red")

    print_colored("\n" + "="*60, "yellow")

    # Save search results
    print_colored(f"\n5. Saving search results...", "blue")

    results_file = os.path.join(output_dir, "search_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'search_type': search_type,
            'total_experiments': len(combinations),
            'successful_experiments': len(successful_models),
            'failed_experiments': len(combinations) - len(successful_models),
            'base_config': base_config_path,
            'param_grid': param_grid if search_type == "grid" else None,
            'param_distributions': param_distributions if search_type == "random" else None,
            'n_trials': n_trials if search_type == "random" else None,
            'experiments': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print_colored(f"   Results saved to: {results_file}", "cyan")

    # Print summary
    print_colored(f"\n📊 Search Summary:", "cyan")
    print_colored(f"   Total experiments: {len(combinations)}", "blue")
    print_colored(f"   Successful: {len(successful_models)}", "green")
    print_colored(f"   Failed: {len(combinations) - len(successful_models)}", "red" if len(combinations) - len(successful_models) > 0 else "blue")

    # Compare models if we have successful ones
    if len(successful_models) >= 2:
        print_colored(f"\n6. Comparing {len(successful_models)} successful models...", "blue")

        compare_cmd = [
            "python3",
            "scripts/5_compare_models.py",
            "--models"
        ] + successful_models + [
            "--data", data_file,
            "--distributions", distributions_file,
            "--output", os.path.join(output_dir, "comparison")
        ]

        try:
            subprocess.run(compare_cmd, check=True)
            print_colored("   ✓ Comparison completed", "green")
        except subprocess.CalledProcessError:
            print_colored("   ✗ Comparison failed", "red")

    elif len(successful_models) == 1:
        print_colored("\n⚠️  Only one successful model, skipping comparison", "yellow")
    else:
        print_colored("\n❌ No successful models to compare", "red")

    print_colored("\n" + "="*60, "yellow")
    print_colored("✅ HYPERPARAMETER SEARCH COMPLETE", "green")
    print_colored("="*60, "yellow")

    print_colored(f"\nAll results saved to: {output_dir}", "cyan")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for MAPPO training"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preprocessed data (from phase 1)"
    )
    parser.add_argument(
        "--distributions",
        type=str,
        required=True,
        help="Path to fitted distributions (from phase 2)"
    )
    parser.add_argument(
        "--search-type",
        type=str,
        choices=["grid", "random"],
        default="grid",
        help="Type of search: grid or random (default: grid)"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/default.yaml",
        help="Path to base config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--param-config",
        type=str,
        help="Path to JSON file with parameter grid or distributions"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of trials for random search (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for search results (default: experiments/hp_search_TIMESTAMP)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (not yet implemented)"
    )

    args = parser.parse_args()

    # Load parameter config
    if args.param_config:
        with open(args.param_config, 'r') as f:
            param_config = json.load(f)

        if args.search_type == "grid":
            param_grid = param_config.get("param_grid")
            param_distributions = None
        else:
            param_grid = None
            param_distributions = param_config.get("param_distributions")
    else:
        # Default parameter grid
        print_colored("⚠️  No parameter config provided, using default grid", "yellow")
        param_grid = {
            "learning.lr_actor": [0.0001, 0.0003, 0.001],
            "learning.lr_critic": [0.0001, 0.0003, 0.001],
            "network.actor_hidden_size": [64, 128, 256]
        }
        param_distributions = None

    hyperparameter_search(
        data_file=args.data,
        distributions_file=args.distributions,
        search_type=args.search_type,
        param_grid=param_grid,
        param_distributions=param_distributions,
        n_trials=args.n_trials,
        base_config_path=args.base_config,
        output_dir=args.output,
        parallel=args.parallel
    )
