#!/usr/bin/env python3
"""
Phase 4: Evaluation Only
Evaluates a trained MAPPO model on test data.
Can evaluate multiple models for comparison.
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


def evaluate_model(model_path, data_file, distributions_file, episodes=10, output_dir=None):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to trained model directory
        data_file: Path to preprocessed data
        distributions_file: Path to fitted distributions
        episodes: Number of evaluation episodes
        output_dir: Output directory for evaluation results
    """
    print_colored("="*60, "yellow")
    print_colored("PHASE 4: MODEL EVALUATION", "yellow")
    print_colored("="*60, "yellow")

    # Load preprocessed data
    print_colored(f"\n1. Loading preprocessed data from {data_file}...", "blue")
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    test_data = data_dict['test']
    print_colored(f"   Test set: {len(test_data)} events", "green")

    # Load fitted distributions
    print_colored(f"\n2. Loading fitted distributions...", "blue")
    fitted_distributions = load_fitted_distributions(distributions_file)
    print_colored(f"   Distributions loaded successfully", "green")

    # Initialize environment
    print_colored("\n3. Initializing test environment...", "blue")
    simulation_parameters = SimulationParameters(
        {"start_timestamp": test_data["start_timestamp"].min()}
    )

    # Create evaluation directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(model_path), f"evaluation_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    test_env = AgentOptimizerEnvironment(
        test_data,
        simulation_parameters,
        experiment_dir=output_dir,
        pre_fitted_distributions=fitted_distributions,
    )
    print_colored(f"   Environment initialized with {len(test_env.agents)} agents", "green")

    # Load trained model
    print_colored(f"\n4. Loading trained model from {model_path}...", "blue")
    device = get_device()
    agent = MAPPOAgent(test_env, device=device)
    agent.load_models(model_path)
    print_colored(f"   Model loaded on {device}", "green")

    # Run evaluation
    print_colored(f"\n5. Running {episodes} evaluation episodes...", "green")
    print_colored("="*60, "yellow")

    episode_rewards = []
    for ep in range(episodes):
        print_colored(f"\nEpisode {ep+1}/{episodes}", "cyan")

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
        print_colored(f"Episode {ep+1} reward: {total_reward:.2f} (steps: {steps})", "green")

    print_colored("="*60, "yellow")

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)

    print_colored(f"\n📊 Evaluation Results ({episodes} episodes):", "cyan")
    print_colored(f"   Mean reward: {mean_reward:.2f}", "green")
    print_colored(f"   Std deviation: {std_reward:.2f}", "green")
    print_colored(f"   Min reward: {min_reward:.2f}", "green")
    print_colored(f"   Max reward: {max_reward:.2f}", "green")

    # Save results
    results = {
        'model_path': model_path,
        'data_file': data_file,
        'episodes': episodes,
        'episode_rewards': episode_rewards,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'min_reward': float(min_reward),
        'max_reward': float(max_reward),
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }

    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print_colored(f"\nResults saved to: {results_file}", "cyan")

    # Save summary
    summary_file = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Model Evaluation Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Episodes: {episodes}\n\n")
        f.write(f"Mean Reward: {mean_reward:.2f}\n")
        f.write(f"Std Deviation: {std_reward:.2f}\n")
        f.write(f"Min Reward: {min_reward:.2f}\n")
        f.write(f"Max Reward: {max_reward:.2f}\n\n")
        f.write("Episode Rewards:\n")
        for i, reward in enumerate(episode_rewards):
            f.write(f"  Episode {i+1}: {reward:.2f}\n")

    print_colored(f"Summary saved to: {summary_file}", "cyan")

    print_colored("\n" + "="*60, "yellow")
    print_colored("✅ EVALUATION COMPLETE", "green")
    print_colored("="*60, "yellow")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: Evaluate trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory"
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
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for evaluation results (default: model_dir/evaluation_TIMESTAMP)"
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_file=args.data,
        distributions_file=args.distributions,
        episodes=args.episodes,
        output_dir=args.output
    )
