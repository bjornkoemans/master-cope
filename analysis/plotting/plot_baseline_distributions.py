#!/usr/bin/env python3
"""
Script to create distribution comparison plots for different baselines.
Shows distributions, means, and medians for random, best median, ground truth, and MAPPO agents.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def load_baseline_results(baseline_dir: str) -> Dict:
    """Load baseline comparison results from JSON file."""
    results_path = Path(baseline_dir) / "baseline_comparison_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Baseline results not found at {results_path}")

    with open(results_path, "r") as f:
        return json.load(f)


def load_mappo_results(mappo_dir: str) -> Dict:
    """Load MAPPO training results from experiment directory."""
    mappo_path = Path(mappo_dir)

    # Load training summary for basic info
    mappo_results = {}

    # Try to load episode-level rewards from training summary
    training_summary_path = mappo_path / "training_summary.txt"
    episode_rewards = []

    if training_summary_path.exists():
        # Parse episode rewards from training summary
        with open(training_summary_path, "r") as f:
            lines = f.readlines()

        in_episode_section = False
        for line in lines:
            line = line.strip()
            if line == "Episode Rewards:":
                in_episode_section = True
                continue
            elif in_episode_section and line.startswith("Episode "):
                # Parse line like "Episode 0: 123.45"
                try:
                    reward_str = line.split(": ")[1]
                    episode_rewards.append(float(reward_str))
                except (IndexError, ValueError):
                    continue
            elif in_episode_section and (line == "" or line.startswith("Evaluation")):
                # End of episode rewards section
                break

    # Fallback: try to load from individual episode directories
    if not episode_rewards:
        episodes_dir = mappo_path / "episodes"
        if episodes_dir.exists():
            episode_dirs = sorted(
                [
                    d
                    for d in episodes_dir.iterdir()
                    if d.is_dir() and d.name.startswith("episode_")
                ]
            )
            for episode_dir in episode_dirs:
                episode_summary_path = episode_dir / "summary.txt"
                if episode_summary_path.exists():
                    with open(episode_summary_path, "r") as f:
                        for line in f:
                            if line.startswith("Total Reward:"):
                                try:
                                    reward = float(line.split(": ")[1])
                                    episode_rewards.append(reward)
                                    break
                                except (IndexError, ValueError):
                                    pass

    # Convert to numpy array
    episode_rewards_array = (
        np.array(episode_rewards) if episode_rewards else np.array([])
    )

    if len(episode_rewards_array) > 0:
        mappo_results = {
            "MAPPO Agent": {
                "agent_name": "MAPPO Agent",
                "num_episodes": len(episode_rewards_array),
                "mean_reward": float(np.mean(episode_rewards_array)),
                "std_reward": float(np.std(episode_rewards_array)),
                "median_reward": float(np.median(episode_rewards_array)),
                "min_reward": float(np.min(episode_rewards_array)),
                "max_reward": float(np.max(episode_rewards_array)),
                "episode_rewards": episode_rewards_array.tolist(),
                "total_cumulative_reward": float(np.sum(episode_rewards_array)),
            }
        }

    # Also try to load final evaluation results if available
    final_eval_path = mappo_path / "final_evaluation"
    if final_eval_path.exists():
        # Look for evaluation results
        eval_files = list(final_eval_path.glob("*.json"))
        if eval_files:
            with open(eval_files[0], "r") as f:
                eval_data = json.load(f)
                if "episode_rewards" in eval_data:
                    mappo_results["MAPPO Agent (Evaluation)"] = {
                        "agent_name": "MAPPO Agent (Evaluation)",
                        "num_episodes": len(eval_data["episode_rewards"]),
                        "mean_reward": float(np.mean(eval_data["episode_rewards"])),
                        "std_reward": float(np.std(eval_data["episode_rewards"])),
                        "median_reward": float(np.median(eval_data["episode_rewards"])),
                        "min_reward": float(np.min(eval_data["episode_rewards"])),
                        "max_reward": float(np.max(eval_data["episode_rewards"])),
                        "episode_rewards": eval_data["episode_rewards"],
                    }

    return mappo_results


def extract_agent_episode_rewards(
    all_results: Dict,
) -> Tuple[List[str], List[List[float]]]:
    """Extract agent names and their episode rewards from results."""
    agent_names = []
    episode_rewards = []

    for agent_name, data in all_results.items():
        if "episode_rewards" in data and data["episode_rewards"]:
            agent_names.append(agent_name)
            episode_rewards.append(data["episode_rewards"])

    return agent_names, episode_rewards


def extract_metrics_for_comparison(
    all_results: Dict,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Extract agent names, means, medians, and standard deviations."""
    agent_names = []
    means = []
    medians = []
    stds = []

    for agent_name, data in all_results.items():
        agent_names.append(agent_name)
        means.append(data["mean_reward"])
        medians.append(data["median_reward"])
        stds.append(data["std_reward"])

    return agent_names, np.array(means), np.array(medians), np.array(stds)


def plot_baseline_distributions(
    baseline_dir: str,
    mappo_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    display: bool = True,
) -> None:
    """
    Create comprehensive distribution comparison plots similar to plot_log_metrics.

    Args:
        baseline_dir: Directory containing baseline evaluation results
        mappo_dir: Directory containing MAPPO training results (optional)
        output_dir: Directory to save plots (if None, uses baseline_dir)
        display: Whether to display plots
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
            "figure.figsize": (16, 10),
        }
    )

    # Load baseline results
    baseline_results = load_baseline_results(baseline_dir)

    # Load MAPPO results if provided
    all_results = baseline_results.copy()
    if mappo_dir and os.path.exists(mappo_dir):
        mappo_results = load_mappo_results(mappo_dir)
        all_results.update(mappo_results)

    # Determine output directory
    if output_dir is None:
        output_dir = baseline_dir
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for plotting
    agent_names, episode_rewards = extract_agent_episode_rewards(all_results)
    _, means, medians, stds = extract_metrics_for_comparison(all_results)

    if not agent_names:
        print("No episode rewards data found for plotting distributions")
        return

    # Create the main comparison plot (similar to plot_log_metrics structure)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Baseline Agent Performance Distribution Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # Color palette for consistency
    colors = sns.color_palette("Set2", len(agent_names))
    color_dict = dict(zip(agent_names, colors))

    # Plot 1: Violin plots showing full distributions
    ax1 = axes[0, 0]
    parts = ax1.violinplot(
        episode_rewards,
        positions=range(len(episode_rewards)),
        showmeans=True,
        showmedians=True,
    )

    # Color the violin plots
    for i, part in enumerate(parts["bodies"]):
        part.set_facecolor(colors[i])
        part.set_alpha(0.7)

    # Add median and mean points
    ax1.scatter(
        range(len(medians)),
        medians,
        color="red",
        s=50,
        zorder=5,
        label="Median",
        marker="s",
    )
    ax1.scatter(
        range(len(means)), means, color="blue", s=50, zorder=5, label="Mean", marker="o"
    )

    ax1.set_xticks(range(len(agent_names)))
    ax1.set_xticklabels(agent_names, rotation=45, ha="right")
    ax1.set_title("Reward Distributions by Agent")
    ax1.set_ylabel("Episode Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Box plots for detailed quartile information
    ax2 = axes[0, 1]
    box_parts = ax2.boxplot(
        episode_rewards, labels=agent_names, patch_artist=True, notch=True
    )

    # Color the box plots
    for i, patch in enumerate(box_parts["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)

    ax2.set_xticklabels(agent_names, rotation=45, ha="right")
    ax2.set_title("Reward Box Plots (with Quartiles)")
    ax2.set_ylabel("Episode Reward")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mean and Median comparison bar chart
    ax3 = axes[1, 0]
    x = np.arange(len(agent_names))
    width = 0.35

    bars1 = ax3.bar(
        x - width / 2,
        means,
        width,
        label="Mean",
        color=[color_dict[name] for name in agent_names],
        alpha=0.8,
    )
    bars2 = ax3.bar(
        x + width / 2,
        medians,
        width,
        label="Median",
        color=[color_dict[name] for name in agent_names],
        alpha=0.6,
    )

    # Add error bars for standard deviation
    ax3.errorbar(
        x - width / 2, means, yerr=stds, fmt="none", color="black", capsize=5, alpha=0.7
    )

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Mean value
        height1 = bar1.get_height()
        ax3.text(
            bar1.get_x() + bar1.get_width() / 2.0,
            height1 + stds[i] * 0.1,
            f"{means[i]:.1e}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )
        # Median value
        height2 = bar2.get_height()
        ax3.text(
            bar2.get_x() + bar2.get_width() / 2.0,
            height2 + abs(height2) * 0.05,
            f"{medians[i]:.1e}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    ax3.set_xlabel("Agent")
    ax3.set_ylabel("Reward")
    ax3.set_title("Mean vs Median Rewards with Standard Deviation")
    ax3.set_xticks(x)
    ax3.set_xticklabels(agent_names, rotation=45, ha="right")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Statistical summary table as text
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create summary statistics table
    summary_data = []
    for i, name in enumerate(agent_names):
        summary_data.append(
            [
                name,
                f"{means[i]:.2e}",
                f"{medians[i]:.2e}",
                f"{stds[i]:.2e}",
                f"{all_results[name]['min_reward']:.2e}",
                f"{all_results[name]['max_reward']:.2e}",
                f"{all_results[name]['num_episodes']}",
            ]
        )

    # Create table
    table = ax4.table(
        cellText=summary_data,
        colLabels=["Agent", "Mean", "Median", "Std Dev", "Min", "Max", "Episodes"],
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.13, 0.13, 0.13, 0.13, 0.13, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color the header
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Color rows based on agent
    for i, name in enumerate(agent_names):
        color = color_dict[name]
        for j in range(len(summary_data[0])):
            table[(i + 1, j)].set_facecolor(color)
            table[(i + 1, j)].set_alpha(0.3)

    ax4.set_title("Statistical Summary", fontweight="bold", pad=20)

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "baseline_distribution_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Distribution comparison plot saved to: {output_path}")

    if display:
        plt.show()
    else:
        plt.close()

    # Create additional focused plots
    _create_focused_distribution_plot(
        agent_names, episode_rewards, color_dict, output_dir, display
    )
    _create_performance_ranking_plot(all_results, output_dir, display)


def _create_focused_distribution_plot(
    agent_names: List[str],
    episode_rewards: List[List[float]],
    color_dict: Dict,
    output_dir: str,
    display: bool,
) -> None:
    """Create a focused distribution plot with histograms and KDE."""
    plt.figure(figsize=(14, 8))

    # Create subplots for histograms and KDE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Overlapping histograms
    for i, (name, rewards) in enumerate(zip(agent_names, episode_rewards)):
        ax1.hist(
            rewards,
            bins=20,
            alpha=0.6,
            label=name,
            color=color_dict[name],
            density=True,
        )

    ax1.set_xlabel("Episode Reward")
    ax1.set_ylabel("Density")
    ax1.set_title("Reward Distribution Histograms")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: KDE plots
    for i, (name, rewards) in enumerate(zip(agent_names, episode_rewards)):
        if len(rewards) > 1:  # Need at least 2 points for KDE
            sns.kdeplot(data=rewards, label=name, color=color_dict[name], ax=ax2)
        else:
            # For single point, draw a vertical line
            ax2.axvline(
                x=rewards[0],
                color=color_dict[name],
                linestyle="--",
                alpha=0.7,
                label=name,
            )

    ax2.set_xlabel("Episode Reward")
    ax2.set_ylabel("Density")
    ax2.set_title("Reward Distribution Kernel Density Estimates")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "focused_distribution_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Focused distribution plot saved to: {output_path}")

    if display:
        plt.show()
    else:
        plt.close()


def _create_performance_ranking_plot(
    all_results: Dict, output_dir: str, display: bool
) -> None:
    """Create a performance ranking plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract and sort by median performance
    agents_sorted_by_median = sorted(
        all_results.items(), key=lambda x: x[1]["median_reward"], reverse=True
    )
    names_median = [x[0] for x in agents_sorted_by_median]
    medians_sorted = [x[1]["median_reward"] for x in agents_sorted_by_median]

    # Extract and sort by mean performance
    agents_sorted_by_mean = sorted(
        all_results.items(), key=lambda x: x[1]["mean_reward"], reverse=True
    )
    names_mean = [x[0] for x in agents_sorted_by_mean]
    means_sorted = [x[1]["mean_reward"] for x in agents_sorted_by_mean]

    # Colors for ranking
    colors_median = sns.color_palette("viridis", len(names_median))
    colors_mean = sns.color_palette("plasma", len(names_mean))

    # Plot 1: Ranking by median
    bars1 = ax1.barh(range(len(names_median)), medians_sorted, color=colors_median)
    ax1.set_yticks(range(len(names_median)))
    ax1.set_yticklabels(names_median)
    ax1.set_xlabel("Median Reward")
    ax1.set_title("Agent Ranking by Median Performance")
    ax1.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, medians_sorted)):
        ax1.text(
            value + abs(value) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1e}",
            ha="left",
            va="center",
            fontsize=9,
        )

    # Plot 2: Ranking by mean
    bars2 = ax2.barh(range(len(names_mean)), means_sorted, color=colors_mean)
    ax2.set_yticks(range(len(names_mean)))
    ax2.set_yticklabels(names_mean)
    ax2.set_xlabel("Mean Reward")
    ax2.set_title("Agent Ranking by Mean Performance")
    ax2.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, means_sorted)):
        ax2.text(
            value + abs(value) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1e}",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "performance_ranking_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Performance ranking plot saved to: {output_path}")

    if display:
        plt.show()
    else:
        plt.close()


def main():
    """Main function to run the baseline distribution plotting."""
    parser = argparse.ArgumentParser(
        description="Plot baseline distribution comparisons"
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="experiments/baseline_evaluation_20250704_143426",
        help="Directory containing baseline evaluation results",
    )
    parser.add_argument(
        "--mappo-dir",
        type=str,
        default=None,
        help="Directory containing MAPPO training results (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to baseline-dir)",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Don't display plots, just save them"
    )

    args = parser.parse_args()

    # Check if baseline directory exists
    if not os.path.exists(args.baseline_dir):
        print(f"Error: Baseline directory not found: {args.baseline_dir}")
        print("Available baseline evaluation directories:")
        experiments_dir = Path("experiments")
        if experiments_dir.exists():
            for d in experiments_dir.glob("baseline_evaluation_*"):
                print(f"  {d}")
        return

    # Check if MAPPO directory exists (if provided)
    if args.mappo_dir and not os.path.exists(args.mappo_dir):
        print(f"Warning: MAPPO directory not found: {args.mappo_dir}")
        print("Available MAPPO directories:")
        experiments_dir = Path("experiments")
        if experiments_dir.exists():
            for d in experiments_dir.glob("mappo_*"):
                print(f"  {d}")
        args.mappo_dir = None

    # Create the plots
    try:
        plot_baseline_distributions(
            baseline_dir=args.baseline_dir,
            mappo_dir=args.mappo_dir,
            output_dir=args.output_dir,
            display=not args.no_display,
        )
        print("Baseline distribution plots created successfully!")
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
