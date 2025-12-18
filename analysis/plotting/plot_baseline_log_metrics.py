#!/usr/bin/env python3
"""
Script to create log metrics distribution comparison plots for different baselines.
Shows distributions, means, and medians for throughput, waiting, and processing times
across random, best median, ground truth, and MAPPO agents.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def extract_log_metrics_from_episode(
    log_file: Path,
) -> Tuple[List[float], List[float], List[float]]:
    """Extract throughput, waiting, and processing times from a single episode log file."""
    if not log_file.exists():
        return [], [], []

    log_df = pd.read_csv(log_file)
    case_groups = log_df.groupby("case_id")

    throughput_times = []
    waiting_times = []
    processing_times = []

    for case_id, group in case_groups:
        # Convert timestamps
        group_times = group[
            ["task_assigned_time", "task_started_time", "task_completed_time"]
        ].copy()
        for col in group_times.columns:
            group_times[col] = pd.to_datetime(group_times[col], errors="coerce")

        # Calculate metrics (timestamps are now in UTC)
        waiting_times_vec = (
            group_times["task_started_time"] - group_times["task_assigned_time"]
        ).dt.total_seconds() / 60
        processing_times_vec = (
            group_times["task_completed_time"] - group_times["task_started_time"]
        ).dt.total_seconds() / 60
        throughput_times_vec = (
            group_times["task_completed_time"] - group_times["task_assigned_time"]
        ).dt.total_seconds() / 60

        # Sum up the times for the case
        waiting = waiting_times_vec.sum()
        processing = processing_times_vec.sum()
        throughput = throughput_times_vec.sum()

        if not pd.isna(waiting) and waiting >= 0:
            waiting_times.append(waiting)
        if not pd.isna(processing) and processing >= 0:
            processing_times.append(processing)
        if not pd.isna(throughput) and throughput >= 0:
            throughput_times.append(throughput)

    return throughput_times, waiting_times, processing_times


def load_baseline_log_metrics(
    baseline_dir: str,
) -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Load log metrics for all baseline agents from a baseline evaluation directory.

    Returns:
        Dict mapping agent names to their metrics:
        {
            "Random Baseline": {
                "throughput": [[episode1_times], [episode2_times], ...],
                "waiting": [[episode1_times], [episode2_times], ...],
                "processing": [[episode1_times], [episode2_times], ...]
            },
            ...
        }
    """
    baseline_path = Path(baseline_dir)

    # Load baseline results to get agent names and episode count
    results_file = baseline_path / "baseline_comparison_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Baseline results not found at {results_file}")

    with open(results_file, "r") as f:
        baseline_results = json.load(f)

    # Get log files sorted by timestamp
    logs_dir = baseline_path / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found at {logs_dir}")

    log_files = sorted(logs_dir.glob("log_*.csv"))

    # Initialize structure for storing metrics
    agents_metrics = {}
    agent_names = list(baseline_results.keys())

    for agent_name in agent_names:
        agents_metrics[agent_name] = {"throughput": [], "waiting": [], "processing": []}

    # Distribute log files among agents (assuming they were run sequentially)
    num_agents = len(agent_names)
    episodes_per_agent = len(log_files) // num_agents

    for agent_idx, agent_name in enumerate(agent_names):
        start_idx = agent_idx * episodes_per_agent
        end_idx = (
            (agent_idx + 1) * episodes_per_agent
            if agent_idx < num_agents - 1
            else len(log_files)
        )

        agent_log_files = log_files[start_idx:end_idx]

        for log_file in agent_log_files:
            throughput, waiting, processing = extract_log_metrics_from_episode(log_file)

            agents_metrics[agent_name]["throughput"].append(throughput)
            agents_metrics[agent_name]["waiting"].append(waiting)
            agents_metrics[agent_name]["processing"].append(processing)

    return agents_metrics


def load_mappo_log_metrics(
    mappo_dir: str, num_episodes: int = 5
) -> Dict[str, Dict[str, List[List[float]]]]:
    """Load log metrics from a MAPPO experiment directory."""
    mappo_path = Path(mappo_dir)

    # Try to find log files
    logs_dir = mappo_path / "logs"
    test_run_logs_dir = mappo_path / "test_run" / "logs"

    log_files = []

    # Prefer test_run logs if available
    if test_run_logs_dir.exists():
        log_files = sorted(list(test_run_logs_dir.glob("log_*.csv")))
    elif logs_dir.exists():
        log_files = sorted(list(logs_dir.glob("log_*.csv")))

    if not log_files:
        print(f"Warning: No log files found in {mappo_dir}")
        return {}

    # Take the last num_episodes log files for consistency with baseline evaluation
    if len(log_files) > num_episodes:
        log_files = log_files[-num_episodes:]

    mappo_metrics = {"MAPPO Agent": {"throughput": [], "waiting": [], "processing": []}}

    for log_file in log_files:
        throughput, waiting, processing = extract_log_metrics_from_episode(log_file)

        mappo_metrics["MAPPO Agent"]["throughput"].append(throughput)
        mappo_metrics["MAPPO Agent"]["waiting"].append(waiting)
        mappo_metrics["MAPPO Agent"]["processing"].append(processing)

    return mappo_metrics


def flatten_metrics(
    agents_metrics: Dict[str, Dict[str, List[List[float]]]],
) -> Dict[str, Dict[str, List[float]]]:
    """Flatten the nested episode structure to get all times per agent per metric."""
    flattened = {}

    for agent_name, metrics in agents_metrics.items():
        flattened[agent_name] = {}
        for metric_name, episodes in metrics.items():
            # Flatten all episode times into a single list
            all_times = []
            for episode_times in episodes:
                all_times.extend(episode_times)
            flattened[agent_name][metric_name] = all_times

    return flattened


def plot_baseline_log_metrics(
    baseline_dir: str,
    mappo_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    display: bool = True,
) -> None:
    """
    Create log metrics distribution comparison plots similar to plot_log_metrics.

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
        }
    )

    print(f"Loading baseline log metrics from: {baseline_dir}")

    # Load baseline metrics
    try:
        agents_metrics = load_baseline_log_metrics(baseline_dir)
    except Exception as e:
        print(f"Error loading baseline metrics: {e}")
        return

    # Load MAPPO metrics if provided
    if mappo_dir and os.path.exists(mappo_dir):
        print(f"Loading MAPPO log metrics from: {mappo_dir}")
        try:
            # Get number of episodes per agent from baseline data
            baseline_episodes = len(next(iter(agents_metrics.values()))["throughput"])
            mappo_metrics = load_mappo_log_metrics(
                mappo_dir, num_episodes=baseline_episodes
            )
            agents_metrics.update(mappo_metrics)
        except Exception as e:
            print(f"Warning: Could not load MAPPO metrics: {e}")

    # Flatten the metrics for plotting
    flattened_metrics = flatten_metrics(agents_metrics)

    # Determine output directory
    if output_dir is None:
        output_dir = baseline_dir
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    agent_names = list(flattened_metrics.keys())
    metrics = ["throughput", "waiting", "processing"]
    metric_labels = [
        "Throughput Time (minutes)",
        "Waiting Time (minutes)",
        "Processing Time (minutes)",
    ]
    metric_titles = [
        "Case Throughput Time by Agent",
        "Case Waiting Time by Agent",
        "Case Processing Time by Agent",
    ]

    # Color palette for consistency
    colors = sns.color_palette("Set2", len(agent_names))
    color_dict = dict(zip(agent_names, colors))

    # Create the main comparison plot (similar to plot_log_metrics structure)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(
        "Baseline Agent Log Metrics Distribution Comparison",
        fontsize=16,
        fontweight="bold",
    )

    for i, (metric, ylabel, title) in enumerate(
        zip(metrics, metric_labels, metric_titles)
    ):
        ax = axes[i]

        # Prepare data for this metric
        metric_data = []
        valid_agent_names = []
        valid_colors = []

        for agent_name in agent_names:
            if (
                metric in flattened_metrics[agent_name]
                and flattened_metrics[agent_name][metric]
            ):
                metric_data.append(flattened_metrics[agent_name][metric])
                valid_agent_names.append(agent_name)
                valid_colors.append(color_dict[agent_name])

        if not metric_data:
            ax.text(
                0.5,
                0.5,
                f"No {metric} data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            continue

        # Create violin plot
        parts = ax.violinplot(
            metric_data,
            positions=range(len(metric_data)),
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )

        # Color the violin plots
        for j, part in enumerate(parts["bodies"]):
            part.set_facecolor(valid_colors[j])
            part.set_alpha(0.7)

        # Calculate and add median and mean points
        medians = [np.median(data) for data in metric_data]
        means = [np.mean(data) for data in metric_data]

        ax.scatter(
            range(len(medians)),
            medians,
            color="red",
            s=50,
            zorder=5,
            label="Median",
            marker="s",
        )
        ax.scatter(
            range(len(means)),
            means,
            color="blue",
            s=50,
            zorder=5,
            label="Mean",
            marker="o",
        )

        # Set labels and formatting
        ax.set_xticks(range(len(valid_agent_names)))
        ax.set_xticklabels(valid_agent_names, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Add legend only to the first subplot
        if i == 0:
            ax.legend()

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "baseline_log_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Baseline log metrics comparison plot saved to: {output_path}")

    if display:
        plt.show()
    else:
        plt.close()

    # Create additional detailed comparison plot
    _create_detailed_metrics_comparison(
        flattened_metrics, agent_names, color_dict, output_dir, display
    )


def _create_detailed_metrics_comparison(
    flattened_metrics: Dict[str, Dict[str, List[float]]],
    agent_names: List[str],
    color_dict: Dict[str, Any],
    output_dir: str,
    display: bool,
) -> None:
    """Create a detailed comparison plot with statistics."""

    metrics = ["throughput", "waiting", "processing"]
    metric_labels = [
        "Throughput Time (minutes)",
        "Waiting Time (minutes)",
        "Processing Time (minutes)",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Detailed Baseline Agent Log Metrics Analysis", fontsize=16, fontweight="bold"
    )

    # Top row: Box plots for detailed quartile information
    for i, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[0, i]

        # Prepare data for this metric
        metric_data = []
        valid_agent_names = []

        for agent_name in agent_names:
            if (
                metric in flattened_metrics[agent_name]
                and flattened_metrics[agent_name][metric]
            ):
                metric_data.append(flattened_metrics[agent_name][metric])
                valid_agent_names.append(agent_name)

        if not metric_data:
            ax.text(
                0.5,
                0.5,
                f"No {metric} data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            continue

        # Create box plot
        box_parts = ax.boxplot(
            metric_data, tick_labels=valid_agent_names, patch_artist=True, notch=True
        )

        # Color the box plots
        for j, patch in enumerate(box_parts["boxes"]):
            agent_name = valid_agent_names[j]
            patch.set_facecolor(color_dict[agent_name])
            patch.set_alpha(0.7)

        ax.set_xticklabels(valid_agent_names, rotation=45, ha="right")
        ax.set_title(f"{ylabel} - Box Plot")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Bottom row: Statistical summary and comparison
    for i, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[1, i]

        # Prepare statistics
        stats_data = []
        valid_agent_names = []

        for agent_name in agent_names:
            if (
                metric in flattened_metrics[agent_name]
                and flattened_metrics[agent_name][metric]
            ):
                data = flattened_metrics[agent_name][metric]
                if data:
                    stats = {
                        "agent": agent_name,
                        "mean": np.mean(data),
                        "median": np.median(data),
                        "std": np.std(data),
                        "count": len(data),
                    }
                    stats_data.append(stats)
                    valid_agent_names.append(agent_name)

        if not stats_data:
            ax.text(
                0.5,
                0.5,
                f"No {metric} data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            continue

        # Create bar chart comparing means and medians
        x = np.arange(len(stats_data))
        width = 0.35

        means = [s["mean"] for s in stats_data]
        medians = [s["median"] for s in stats_data]
        stds = [s["std"] for s in stats_data]

        ax.bar(
            x - width / 2,
            means,
            width,
            label="Mean",
            color=[color_dict[name] for name in valid_agent_names],
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            medians,
            width,
            label="Median",
            color=[color_dict[name] for name in valid_agent_names],
            alpha=0.6,
        )

        # Add error bars for standard deviation on means
        ax.errorbar(
            x - width / 2,
            means,
            yerr=stds,
            fmt="none",
            color="black",
            capsize=5,
            alpha=0.7,
        )

        ax.set_xlabel("Agent")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} - Mean vs Median")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_agent_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the detailed plot
    output_path = Path(output_dir) / "baseline_log_metrics_detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Detailed baseline log metrics plot saved to: {output_path}")

    if display:
        plt.show()
    else:
        plt.close()


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot baseline log metrics comparison")
    parser.add_argument(
        "baseline_dir", help="Directory containing baseline evaluation results"
    )
    parser.add_argument(
        "--mappo-dir", help="Directory containing MAPPO results (optional)"
    )
    parser.add_argument(
        "--output-dir", help="Output directory for plots (default: baseline_dir)"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Don't display plots, just save"
    )

    args = parser.parse_args()

    plot_baseline_log_metrics(
        baseline_dir=args.baseline_dir,
        mappo_dir=args.mappo_dir,
        output_dir=args.output_dir,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
