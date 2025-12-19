#!/usr/bin/env python3
"""
Script to create violin plots of evaluation log metrics for baseline experiments.
Groups logs by baseline type and shows distributions with medians and means.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def extract_log_metrics_from_episode(
    log_file: Path,
) -> Tuple[List[float], List[float], List[float]]:
    """Extract throughput, waiting, and processing times from a single episode log file."""
    if not log_file.exists():
        return [], [], []

    try:
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

    except Exception as e:
        print(f"Error processing {log_file}: {e}")
        return [], [], []


def load_evaluation_metrics(
    evaluation_dir: str,
) -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Load log metrics from evaluation directory, grouping logs by baseline type.
    Skips first 10 logs, then groups next 40 logs into 4 baselines (10 logs each).

    Returns:
        Dict mapping baseline names to their metrics:
        {
            "Random": {
                "throughput": [[episode1_times], [episode2_times], ...],
                "waiting": [[episode1_times], [episode2_times], ...],
                "processing": [[episode1_times], [episode2_times], ...]
            },
            ...
        }
    """
    evaluation_path = Path(evaluation_dir)
    logs_dir = evaluation_path / "logs"

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found at {logs_dir}")

    # Get all log files sorted by timestamp
    log_files = sorted(logs_dir.glob("log_*.csv"))
    print(f"Found {len(log_files)} log files")

    # Skip first 10 logs and take next 40 for baseline evaluation
    if len(log_files) < 50:
        print(f"Warning: Expected at least 50 log files, but found {len(log_files)}")

    # Skip first 10 logs
    baseline_log_files = log_files[10:50] if len(log_files) >= 50 else log_files[10:]
    print(f"Using {len(baseline_log_files)} log files for baseline evaluation")

    # Define baseline names in order
    baseline_names = ["Random", "Best Median", "Ground Truth", "MAPPO"]

    # Initialize structure for storing metrics
    baselines_metrics = {}
    for baseline_name in baseline_names:
        baselines_metrics[baseline_name] = {
            "throughput": [],
            "waiting": [],
            "processing": [],
        }

    # Group logs by baseline (10 logs per baseline)
    logs_per_baseline = 10

    for i, baseline_name in enumerate(baseline_names):
        start_idx = i * logs_per_baseline
        end_idx = min((i + 1) * logs_per_baseline, len(baseline_log_files))

        if start_idx >= len(baseline_log_files):
            print(f"Warning: Not enough logs for {baseline_name}")
            break

        baseline_logs = baseline_log_files[start_idx:end_idx]
        print(f"Processing {len(baseline_logs)} logs for {baseline_name}")

        for log_file in baseline_logs:
            throughput, waiting, processing = extract_log_metrics_from_episode(log_file)

            baselines_metrics[baseline_name]["throughput"].append(throughput)
            baselines_metrics[baseline_name]["waiting"].append(waiting)
            baselines_metrics[baseline_name]["processing"].append(processing)

    return baselines_metrics


def flatten_metrics(
    baselines_metrics: Dict[str, Dict[str, List[List[float]]]],
) -> Dict[str, Dict[str, List[float]]]:
    """Flatten the nested episode structure to get all times per baseline per metric."""
    flattened = {}

    for baseline_name, metrics in baselines_metrics.items():
        flattened[baseline_name] = {}
        for metric_name, episodes in metrics.items():
            # Flatten all episode times into a single list
            all_times = []
            for episode_times in episodes:
                all_times.extend(episode_times)
            flattened[baseline_name][metric_name] = all_times

    return flattened


def plot_evaluation_metrics(
    evaluation_dir: str, output_dir: str = None, display: bool = True
) -> None:
    """
    Create violin plots comparing baseline metrics from evaluation logs.

    Args:
        evaluation_dir: Directory containing evaluation logs
        output_dir: Directory to save plots (if None, uses evaluation_dir)
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

    print(f"Loading evaluation metrics from: {evaluation_dir}")

    # Load baseline metrics
    try:
        baselines_metrics = load_evaluation_metrics(evaluation_dir)
    except Exception as e:
        print(f"Error loading evaluation metrics: {e}")
        return

    # Flatten the metrics for plotting
    flattened_metrics = flatten_metrics(baselines_metrics)

    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    for baseline_name, metrics in flattened_metrics.items():
        print(f"\n{baseline_name}:")
        for metric_name, values in metrics.items():
            if values:
                print(
                    f"  {metric_name}: {len(values)} data points, "
                    f"mean={np.mean(values):.2f}, median={np.median(values):.2f}"
                )
            else:
                print(f"  {metric_name}: No data")

    # Determine output directory
    if output_dir is None:
        output_dir = evaluation_dir
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    baseline_names = list(flattened_metrics.keys())
    metrics = ["throughput", "waiting", "processing"]
    metric_labels = [
        "Throughput Time (minutes)",
        "Waiting Time (minutes)",
        "Processing Time (minutes)",
    ]
    metric_titles = [
        "Case Throughput Time by Baseline",
        "Case Waiting Time by Baseline",
        "Case Processing Time by Baseline",
    ]

    # Color palette
    colors = sns.color_palette("Set2", len(baseline_names))
    color_dict = dict(zip(baseline_names, colors))

    # Create the main comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(
        "Baseline Evaluation: Log Metrics Distribution Comparison",
        fontsize=16,
        fontweight="bold",
    )

    for i, (metric, ylabel, title) in enumerate(
        zip(metrics, metric_labels, metric_titles)
    ):
        ax = axes[i]

        # Prepare data for this metric
        metric_data = []
        valid_baseline_names = []
        valid_colors = []

        for baseline_name in baseline_names:
            if (
                metric in flattened_metrics[baseline_name]
                and flattened_metrics[baseline_name][metric]
            ):
                metric_data.append(flattened_metrics[baseline_name][metric])
                valid_baseline_names.append(baseline_name)
                valid_colors.append(color_dict[baseline_name])

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
        ax.set_xticks(range(len(valid_baseline_names)))
        ax.set_xticklabels(valid_baseline_names, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Add legend only to the first subplot
        if i == 0:
            ax.legend()

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "evaluation_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nEvaluation metrics comparison plot saved to: {output_path}")

    if display:
        plt.show()
    else:
        plt.close()

    # Create additional detailed comparison
    _create_detailed_evaluation_comparison(
        flattened_metrics, baseline_names, color_dict, output_dir, display
    )


def _create_detailed_evaluation_comparison(
    flattened_metrics: Dict[str, Dict[str, List[float]]],
    baseline_names: List[str],
    color_dict: Dict[str, any],
    output_dir: str,
    display: bool,
) -> None:
    """Create a detailed comparison plot with box plots and statistics."""

    metrics = ["throughput", "waiting", "processing"]
    metric_labels = [
        "Throughput Time (minutes)",
        "Waiting Time (minutes)",
        "Processing Time (minutes)",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Detailed Baseline Evaluation: Log Metrics Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Top row: Box plots for detailed quartile information
    for i, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[0, i]

        # Prepare data for this metric
        metric_data = []
        valid_baseline_names = []

        for baseline_name in baseline_names:
            if (
                metric in flattened_metrics[baseline_name]
                and flattened_metrics[baseline_name][metric]
            ):
                metric_data.append(flattened_metrics[baseline_name][metric])
                valid_baseline_names.append(baseline_name)

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
            metric_data, tick_labels=valid_baseline_names, patch_artist=True, notch=True
        )

        # Color the box plots
        for j, patch in enumerate(box_parts["boxes"]):
            baseline_name = valid_baseline_names[j]
            patch.set_facecolor(color_dict[baseline_name])
            patch.set_alpha(0.7)

        ax.set_xticklabels(valid_baseline_names, rotation=45, ha="right")
        ax.set_title(f"{ylabel} - Box Plot")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Bottom row: Statistical summary and comparison
    for i, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[1, i]

        # Prepare statistics
        stats_data = []
        valid_baseline_names = []

        for baseline_name in baseline_names:
            if (
                metric in flattened_metrics[baseline_name]
                and flattened_metrics[baseline_name][metric]
            ):
                data = flattened_metrics[baseline_name][metric]
                if data:
                    stats = {
                        "baseline": baseline_name,
                        "mean": np.mean(data),
                        "median": np.median(data),
                        "std": np.std(data),
                        "count": len(data),
                    }
                    stats_data.append(stats)
                    valid_baseline_names.append(baseline_name)

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
            color=[color_dict[name] for name in valid_baseline_names],
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            medians,
            width,
            label="Median",
            color=[color_dict[name] for name in valid_baseline_names],
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

        ax.set_xlabel("Baseline")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} - Mean vs Median")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_baseline_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the detailed plot
    output_path = Path(output_dir) / "evaluation_metrics_detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Detailed evaluation metrics plot saved to: {output_path}")

    if display:
        plt.show()
    else:
        plt.close()


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot evaluation metrics comparison")
    parser.add_argument("evaluation_dir", help="Directory containing evaluation logs")
    parser.add_argument(
        "--output-dir", help="Output directory for plots (default: evaluation_dir)"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Don't display plots, just save"
    )

    args = parser.parse_args()

    plot_evaluation_metrics(
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        display=not args.no_display,
    )


if __name__ == "__main__":
    # If running directly, use the specified evaluation directory
    evaluation_dir = "/Users/pepijnbruinaars/Documents/Master/Thesis/AgentOptimizer/experiments/mappo_20250706_150536/evaluation_20250707_123949"

    print("Creating evaluation metrics plots...")
    plot_evaluation_metrics(evaluation_dir, display=True)
