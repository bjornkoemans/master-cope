#!/usr/bin/env python3
"""
Script to visualize baseline evaluation results.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def load_results(file_path):
    """Load baseline comparison results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def plot_baseline_comparison(results, output_dir=None, display=True):
    """
    Create and save plots comparing baseline agents.

    Args:
        results: Dict with evaluation results
        output_dir: Directory to save plots (if None, will use results directory)
        display: Whether to display plots (vs just saving them)
    """
    # Set up plot styling
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (12, 8),
        }
    )

    # Determine output directory
    if output_dir is None and isinstance(file_path, str):
        output_dir = os.path.dirname(file_path)

    if output_dir is None:
        output_dir = "."

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Mean Rewards Comparison
    plt.figure(figsize=(12, 6))

    # Extract agent names and mean rewards
    agent_names = []
    mean_rewards = []
    std_rewards = []

    for name, data in results.items():
        agent_names.append(name)
        mean_rewards.append(data["mean_reward"])
        std_rewards.append(data["std_reward"])

    # Sort by mean reward (descending)
    sorted_indices = np.argsort(mean_rewards)[::-1]
    agent_names = [agent_names[i] for i in sorted_indices]
    mean_rewards = [mean_rewards[i] for i in sorted_indices]
    std_rewards = [std_rewards[i] for i in sorted_indices]

    # Create bar plot
    colors = sns.color_palette("viridis", len(agent_names))
    bars = plt.bar(
        agent_names, mean_rewards, yerr=std_rewards, capsize=10, color=colors, alpha=0.7
    )

    # Add value labels on top of each bar
    for bar, value in zip(bars, mean_rewards):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.1 * abs(height)),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=10,
        )

    plt.title("Comparison of Mean Rewards Across Agents")
    plt.ylabel("Mean Reward")
    plt.xlabel("Agent")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    mean_rewards_path = os.path.join(output_dir, "mean_rewards_comparison.png")
    plt.savefig(mean_rewards_path, dpi=300, bbox_inches="tight")

    # Plot 2: Reward Distribution Comparison (Box plot)
    plt.figure(figsize=(12, 6))

    # Create box plot data
    box_data = []
    box_labels = []

    for name, data in results.items():
        if "episode_rewards" in data:
            box_data.append(data["episode_rewards"])
            box_labels.append(name)

    # Create box plot
    plt.boxplot(
        box_data,
        patch_artist=True,
        boxprops=dict(alpha=0.7),
        medianprops=dict(color="black", linewidth=1.5),
    )

    # Set the labels after creating the boxplot
    plt.xticks(range(1, len(box_labels) + 1), box_labels)

    plt.title("Distribution of Episode Rewards")
    plt.ylabel("Episode Reward")
    plt.xlabel("Agent")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save plot
    reward_dist_path = os.path.join(output_dir, "reward_distribution_comparison.png")
    plt.savefig(reward_dist_path, dpi=300, bbox_inches="tight")

    # Plot 3: Episode Rewards Over Episodes
    plt.figure(figsize=(12, 6))

    for name, data in results.items():
        if "episode_rewards" in data:
            episodes = list(range(1, len(data["episode_rewards"]) + 1))
            plt.plot(
                episodes,
                data["episode_rewards"],
                marker="o",
                markersize=4,
                linewidth=2,
                label=name,
                alpha=0.7,
            )

    plt.title("Episode Rewards Across Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save plot
    episode_rewards_path = os.path.join(output_dir, "episode_rewards_over_time.png")
    plt.savefig(episode_rewards_path, dpi=300, bbox_inches="tight")

    # Plot 4: Key Metrics Comparison
    plt.figure(figsize=(14, 10))

    # Define metrics to compare
    metrics = ["mean_reward", "median_reward", "min_reward", "max_reward"]
    metric_labels = ["Mean Reward", "Median Reward", "Min Reward", "Max Reward"]

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        metric_values = [results[name][metric] for name in agent_names]

        # Create bar plot for this metric
        bars = axs[i].bar(agent_names, metric_values, color=colors, alpha=0.7)

        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axs[i].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (0.05 * abs(height)),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=0,
                fontsize=9,
            )

        axs[i].set_title(label)
        axs[i].set_xlabel("Agent")
        axs[i].set_ylabel("Value")
        axs[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save plot
    metrics_path = os.path.join(output_dir, "key_metrics_comparison.png")
    plt.savefig(metrics_path, dpi=300, bbox_inches="tight")

    if display:
        plt.show()
    else:
        plt.close("all")

    print(f"Plots saved to: {output_dir}")
    return {
        "mean_rewards_plot": mean_rewards_path,
        "reward_distribution_plot": reward_dist_path,
        "episode_rewards_plot": episode_rewards_path,
        "key_metrics_plot": metrics_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize baseline comparison results"
    )

    parser.add_argument(
        "file_path", type=str, help="Path to the baseline comparison results JSON file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same directory as results file)",
    )

    parser.add_argument(
        "--no-display", action="store_true", help="Don't display plots (only save them)"
    )

    args = parser.parse_args()

    file_path = args.file_path
    results = load_results(file_path)

    plot_baseline_comparison(
        results, output_dir=args.output_dir, display=not args.no_display
    )
