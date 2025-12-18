#!/usr/bin/env python3
"""
Script to evaluate all model checkpoints from a training run and plot the average reward evolution.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse
from multiprocessing import Pool, cpu_count

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.CustomEnvironment.custom_environment import AgentOptimizerEnvironment
from src.CustomEnvironment.custom_environment.env.custom_environment import (
    SimulationParameters,
)
from src.MAPPO.agent import MAPPOAgent
from src.config import config
from src.display import print_colored
from src.preprocessing.load_data import load_data, split_data
from src.preprocessing.preprocessing import remove_short_cases


def create_environment(test_df):
    """Create the evaluation environment with test data."""
    sim_params = SimulationParameters(
        {"start_timestamp": test_df["start_timestamp"].min()}
    )

    env = AgentOptimizerEnvironment(test_df, sim_params)
    return env


def load_checkpoint_agent(env, checkpoint_path):
    """Load a MAPPO agent from a checkpoint."""
    agent = MAPPOAgent(
        env=env,
        hidden_size=64,
        lr_actor=0.0003,
        lr_critic=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        batch_size=1028,
        num_epochs=5,
        device="cpu",
    )

    agent.load_models(checkpoint_path)
    return agent


def run_single_evaluation(args):
    """Run a single evaluation episode. This function is designed to work with multiprocessing."""
    test_df, checkpoint_path, run_id = args

    # Create environment for this process
    env = create_environment(test_df)

    # Load agent for this process
    agent = load_checkpoint_agent(env, checkpoint_path)

    # Run evaluation
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        actions, _ = agent.select_actions(obs, deterministic=True)
        next_obs, rewards_step, dones, truncated, _ = env.step(actions)

        step_reward = float(sum(rewards_step.values()))
        episode_reward += step_reward

        done = any(list(dones.values()) + list(truncated.values()))
        obs = next_obs

    return episode_reward


def evaluate_checkpoint_parallel(
    test_df, checkpoint_path, num_runs=10, num_processes=None
):
    """Evaluate a checkpoint agent over multiple runs using parallel processing."""
    if num_processes is None:
        num_processes = min(
            cpu_count(), num_runs
        )  # Don't create more processes than runs

    print_colored(
        f"    Running {num_runs} evaluations using {num_processes} processes", "cyan"
    )

    try:
        # Prepare arguments for parallel processing
        args_list = [(test_df, checkpoint_path, i) for i in range(num_runs)]

        # Run evaluations in parallel
        with Pool(processes=num_processes) as pool:
            rewards = pool.map(run_single_evaluation, args_list)

        return rewards

    except Exception as e:
        print_colored(
            f"    Parallel processing failed ({str(e)}), falling back to sequential",
            "yellow",
        )
        # Fall back to sequential processing
        env = create_environment(test_df)
        agent = load_checkpoint_agent(env, checkpoint_path)
        return evaluate_checkpoint(env, agent, num_runs)


def evaluate_checkpoint(env, agent, num_runs=10):
    """Evaluate a checkpoint agent over multiple runs (sequential version - kept for compatibility)."""
    rewards = []

    for run in range(num_runs):
        print_colored(f"    Run {run + 1}/{num_runs}", "cyan")

        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            actions, _ = agent.select_actions(obs, deterministic=True)
            next_obs, rewards_step, dones, truncated, _ = env.step(actions)

            step_reward = float(sum(rewards_step.values()))
            episode_reward += step_reward

            done = any(list(dones.values()) + list(truncated.values()))
            obs = next_obs

        rewards.append(episode_reward)

    return rewards


def extract_checkpoint_number(checkpoint_path):
    """Extract checkpoint number from path."""
    path_str = str(checkpoint_path)
    if "final" in path_str:
        return float("inf")
    elif "best" in path_str:
        return -1
    else:
        match = re.search(r"checkpoint_(\d+)", path_str)
        if match:
            return int(match.group(1))
    return 0


def main():
    """Main evaluation function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate MAPPO training checkpoints")
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Use parallel processing (default: True)",
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Force sequential processing"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes for parallel evaluation",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of evaluation runs per checkpoint",
    )
    args = parser.parse_args()

    # Override parallel if sequential is explicitly requested
    use_parallel = args.parallel and not args.sequential

    experiment_path = "/Users/pepijnbruinaars/Documents/Master/Thesis/AgentOptimizer/experiments/mappo_20250628_105149"
    num_runs = args.num_runs
    num_processes = args.num_processes or cpu_count()

    print_colored("=" * 70, "yellow")
    if use_parallel:
        print_colored("CHECKPOINT EVALUATION - TRAINING PROGRESS (PARALLEL)", "yellow")
        print_colored(
            f"Using {num_processes} parallel processes for evaluation", "blue"
        )
    else:
        print_colored(
            "CHECKPOINT EVALUATION - TRAINING PROGRESS (SEQUENTIAL)", "yellow"
        )
    print_colored("=" * 70, "yellow")

    # Load and split data
    print_colored("Loading and splitting data...", "blue")
    df = load_data(config)
    df = remove_short_cases(df)
    train_df, test_df = split_data(df, split=0.8)

    # Find all numbered checkpoints (skip best/final for cleaner plot)
    checkpoint_paths = []
    for i in range(1, 101):  # checkpoint_1 to checkpoint_100
        checkpoint_dir = os.path.join(experiment_path, f"checkpoint_{i}")
        if os.path.exists(checkpoint_dir):
            checkpoint_paths.append(checkpoint_dir)

    # Sort by checkpoint number
    checkpoint_paths = sorted(checkpoint_paths, key=extract_checkpoint_number)

    print_colored(f"Found {len(checkpoint_paths)} checkpoints to evaluate", "green")

    # Create environment for sequential processing
    env = None
    if not use_parallel:
        env = create_environment(test_df)

    # Evaluate each checkpoint
    results = []

    for i, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint_name = os.path.basename(checkpoint_path)
        checkpoint_num = extract_checkpoint_number(checkpoint_path)

        print_colored(
            f"\nEvaluating checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_name}",
            "yellow",
        )

        try:
            if use_parallel:
                # Use parallel evaluation
                rewards = evaluate_checkpoint_parallel(
                    test_df, checkpoint_path, num_runs, num_processes
                )
            else:
                # Use sequential evaluation
                if env is None:
                    env = create_environment(test_df)
                agent = load_checkpoint_agent(env, checkpoint_path)
                rewards = evaluate_checkpoint(env, agent, num_runs)

            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            results.append(
                {
                    "checkpoint_number": checkpoint_num,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                }
            )

            print_colored(
                f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}", "green"
            )

        except Exception as e:
            print_colored(f"  Error evaluating {checkpoint_name}: {str(e)}", "red")
            continue

    # Convert to DataFrame and sort
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("checkpoint_number")

    # Save results
    results_file = os.path.join(experiment_path, "checkpoint_evaluation_results.csv")
    df_results.to_csv(results_file, index=False)
    print_colored(f"\nResults saved to: {results_file}", "green")

    # Create the plot
    print_colored("\nCreating training progress plot...", "blue")
    create_training_progress_plot(df_results, experiment_path)

    print_colored("\nEvaluation complete!", "green")


def create_training_progress_plot(df_results, experiment_path):
    """Create a clean plot showing training progress and potential overfitting."""

    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Main line plot with error bars
    ax.plot(
        df_results["checkpoint_number"],
        df_results["mean_reward"],
        "b-",
        linewidth=2,
        marker="o",
        markersize=4,
        alpha=0.8,
        label="Mean Reward",
    )

    # Add error bars (standard deviation)
    ax.fill_between(
        df_results["checkpoint_number"],
        df_results["mean_reward"] - df_results["std_reward"],
        df_results["mean_reward"] + df_results["std_reward"],
        alpha=0.2,
        color="blue",
        label="± 1 Std Dev",
    )

    # Find and mark the best performing checkpoint
    best_idx = df_results["mean_reward"].idxmax()
    best_checkpoint = df_results.loc[best_idx]
    ax.scatter(
        best_checkpoint["checkpoint_number"],
        best_checkpoint["mean_reward"],
        color="red",
        s=100,
        zorder=5,
        label=f'Best (CP {int(best_checkpoint["checkpoint_number"])})',
    )

    # Add smoothed trend line to identify overfitting
    if len(df_results) >= 10:
        # Calculate moving average
        window_size = max(5, len(df_results) // 10)
        moving_avg = (
            df_results["mean_reward"].rolling(window=window_size, center=True).mean()
        )
        ax.plot(
            df_results["checkpoint_number"],
            moving_avg,
            "r--",
            linewidth=2,
            alpha=0.7,
            label=f"Trend (MA-{window_size})",
        )

        # Check for potential overfitting (when trend starts declining after peak)
        peak_idx = moving_avg.idxmax()
        if peak_idx < len(df_results) - window_size:
            peak_checkpoint_num = df_results.loc[peak_idx, "checkpoint_number"]
            ax.axvline(
                x=peak_checkpoint_num,
                color="orange",
                linestyle=":",
                linewidth=2,
                alpha=0.7,
                label="Potential Overfitting",
            )

    # Formatting
    ax.set_xlabel("Checkpoint Number", fontsize=12)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.set_title(
        "MAPPO Training Progress: Average Reward vs. Training Checkpoints",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Add some statistics as text
    best_reward = df_results["mean_reward"].max()
    final_reward = df_results["mean_reward"].iloc[-1]
    improvement = (
        best_reward - df_results["mean_reward"].iloc[0] if len(df_results) > 1 else 0
    )

    stats_text = f"Best Reward: {best_reward:.2f}\nFinal Reward: {final_reward:.2f}\nTotal Improvement: {improvement:.2f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(experiment_path, "training_progress_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print_colored(f"Plot saved to: {plot_file}", "green")

    plt.show()


if __name__ == "__main__":
    # This guard is important for multiprocessing to work correctly
    main()
