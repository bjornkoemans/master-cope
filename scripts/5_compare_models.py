#!/usr/bin/env python3
"""
Phase 5: Model Comparison
Compares multiple trained models on the same test data.
Generates comparison statistics and visualizations.
"""

import argparse
import pickle
import os
import sys
import json
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.custom_environment import AgentOptimizerEnvironment, SimulationParameters
from src.algorithms.mappo.agent import MAPPOAgent
from src.core.display import print_colored
from src.utils.duration_fitting import load_fitted_distributions


def get_device():
    """Get compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def evaluate_single_model(model_path, test_env, device, episodes=10):
    """
    Evaluate a single model.

    Args:
        model_path: Path to trained model directory
        test_env: Test environment
        device: Compute device
        episodes: Number of evaluation episodes

    Returns:
        Dictionary with evaluation results
    """
    # Load model
    agent = MAPPOAgent(test_env, device=device)
    agent.load_models(model_path)

    # Run evaluation
    episode_rewards = []
    for ep in range(episodes):
        observations = test_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Select actions (deterministic)
            actions, _ = agent.select_actions(observations, deterministic=True)

            # Step environment
            observations, rewards, dones, truncated, info = test_env.step(actions)

            # Update counters
            reward = sum(rewards.values()) if rewards else 0
            total_reward += reward
            steps += 1

            # Check if episode is done
            done = all(dones.values()) if dones else False

        episode_rewards.append(total_reward)

    # Calculate statistics
    return {
        'model_path': model_path,
        'episode_rewards': episode_rewards,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'median_reward': float(np.median(episode_rewards))
    }


def compare_models(model_paths, data_file, distributions_file, episodes=10, output_dir=None):
    """
    Compare multiple trained models.

    Args:
        model_paths: List of paths to trained model directories
        data_file: Path to preprocessed data
        distributions_file: Path to fitted distributions
        episodes: Number of evaluation episodes per model
        output_dir: Output directory for comparison results
    """
    print_colored("="*60, "yellow")
    print_colored("PHASE 5: MODEL COMPARISON", "yellow")
    print_colored("="*60, "yellow")

    # Validate model paths
    print_colored(f"\n1. Validating {len(model_paths)} model paths...", "blue")
    valid_models = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            valid_models.append(model_path)
            print_colored(f"   ✓ {model_path}", "green")
        else:
            print_colored(f"   ✗ {model_path} (not found)", "red")

    if not valid_models:
        print_colored("\n❌ No valid models found!", "red")
        return None

    print_colored(f"\n   Found {len(valid_models)} valid models", "green")

    # Load preprocessed data
    print_colored(f"\n2. Loading preprocessed data from {data_file}...", "blue")
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    test_data = data_dict['test']
    print_colored(f"   Test set: {len(test_data)} events", "green")

    # Load fitted distributions
    print_colored(f"\n3. Loading fitted distributions...", "blue")
    fitted_distributions = load_fitted_distributions(distributions_file)
    print_colored(f"   Distributions loaded successfully", "green")

    # Initialize environment
    print_colored("\n4. Initializing test environment...", "blue")
    simulation_parameters = SimulationParameters(
        {"start_timestamp": test_data["start_timestamp"].min()}
    )

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"experiments/comparison_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    test_env = AgentOptimizerEnvironment(
        test_data,
        simulation_parameters,
        experiment_dir=output_dir,
        pre_fitted_distributions=fitted_distributions,
    )
    print_colored(f"   Environment initialized with {len(test_env.agents)} agents", "green")

    # Get device
    device = get_device()
    print_colored(f"   Using device: {device}", "green")

    # Evaluate all models
    print_colored(f"\n5. Evaluating {len(valid_models)} models ({episodes} episodes each)...", "blue")
    print_colored("="*60, "yellow")

    all_results = []
    for i, model_path in enumerate(valid_models):
        model_name = os.path.basename(os.path.dirname(model_path))
        print_colored(f"\n[{i+1}/{len(valid_models)}] Evaluating: {model_name}", "cyan")

        results = evaluate_single_model(model_path, test_env, device, episodes)
        results['model_name'] = model_name
        all_results.append(results)

        print_colored(f"   Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}", "green")

    print_colored("\n" + "="*60, "yellow")

    # Create comparison DataFrame
    print_colored("\n6. Generating comparison statistics...", "blue")
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Model': result['model_name'],
            'Mean': result['mean_reward'],
            'Std': result['std_reward'],
            'Min': result['min_reward'],
            'Max': result['max_reward'],
            'Median': result['median_reward']
        })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Mean', ascending=False)

    # Print comparison table
    print_colored("\n📊 Model Comparison Results:", "cyan")
    print_colored("="*80, "yellow")
    print(df.to_string(index=False))
    print_colored("="*80, "yellow")

    # Find best model
    best_model = df.iloc[0]
    print_colored(f"\n🏆 Best Model: {best_model['Model']}", "green")
    print_colored(f"   Mean Reward: {best_model['Mean']:.2f}", "green")

    # Calculate relative performance
    print_colored("\n📈 Relative Performance (vs best):", "cyan")
    for _, row in df.iterrows():
        if row['Model'] == best_model['Model']:
            print_colored(f"   {row['Model']}: 100.0% (baseline)", "green")
        else:
            relative_perf = (row['Mean'] / best_model['Mean']) * 100
            print_colored(f"   {row['Model']}: {relative_perf:.1f}%", "blue")

    # Save results
    results_file = os.path.join(output_dir, "comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'models': all_results,
            'comparison_table': df.to_dict(orient='records'),
            'best_model': {
                'name': best_model['Model'],
                'mean_reward': float(best_model['Mean'])
            },
            'episodes_per_model': episodes,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print_colored(f"\nResults saved to: {results_file}", "cyan")

    # Save comparison table as CSV
    csv_file = os.path.join(output_dir, "comparison_table.csv")
    df.to_csv(csv_file, index=False)
    print_colored(f"Table saved to: {csv_file}", "cyan")

    # Save detailed text summary
    summary_file = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Model Comparison Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Number of models: {len(valid_models)}\n")
        f.write(f"Episodes per model: {episodes}\n")
        f.write(f"Test set size: {len(test_data)} events\n\n")
        f.write("Comparison Table:\n")
        f.write("-"*70 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "-"*70 + "\n\n")
        f.write(f"Best Model: {best_model['Model']}\n")
        f.write(f"Best Mean Reward: {best_model['Mean']:.2f}\n\n")
        f.write("Relative Performance:\n")
        for _, row in df.iterrows():
            if row['Model'] == best_model['Model']:
                f.write(f"  {row['Model']}: 100.0% (baseline)\n")
            else:
                relative_perf = (row['Mean'] / best_model['Mean']) * 100
                f.write(f"  {row['Model']}: {relative_perf:.1f}%\n")

    print_colored(f"Summary saved to: {summary_file}", "cyan")

    print_colored("\n" + "="*60, "yellow")
    print_colored("✅ MODEL COMPARISON COMPLETE", "green")
    print_colored("="*60, "yellow")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5: Compare multiple trained models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        required=True,
        help="Paths to trained model directories (space-separated)"
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
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per model (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for comparison results (default: experiments/comparison_TIMESTAMP)"
    )

    args = parser.parse_args()

    compare_models(
        model_paths=args.models,
        data_file=args.data,
        distributions_file=args.distributions,
        episodes=args.episodes,
        output_dir=args.output
    )
