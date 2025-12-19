#!/usr/bin/env python3
"""
Demo script showing how to use the baseline distribution plotting functionality.
"""
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import the plotting function
from plot_baseline_distributions import plot_baseline_distributions


def demo_baseline_plots():
    """Demo the baseline distribution plotting functionality."""
    print("=" * 60)
    print("BASELINE DISTRIBUTION PLOTTING DEMO")
    print("=" * 60)

    # Find available baseline evaluation directories
    experiments_dir = Path("experiments")
    baseline_dirs = list(experiments_dir.glob("baseline_evaluation_*"))
    mappo_dirs = list(experiments_dir.glob("mappo_*"))

    if not baseline_dirs:
        print("No baseline evaluation directories found!")
        print(
            "Run evaluate_baselines.py first to generate baseline evaluation results."
        )
        return

    # Use the most recent baseline evaluation
    latest_baseline = max(baseline_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Using baseline evaluation from: {latest_baseline}")

    # Find a corresponding MAPPO experiment (optional)
    latest_mappo = None
    if mappo_dirs:
        latest_mappo = max(mappo_dirs, key=lambda x: x.stat().st_mtime)
        print(f"Using MAPPO results from: {latest_mappo}")
    else:
        print("No MAPPO experiments found - plotting baselines only")

    # Create the distribution comparison plots
    print("\nGenerating baseline distribution comparison plots...")
    try:
        plot_baseline_distributions(
            baseline_dir=str(latest_baseline),
            mappo_dir=str(latest_mappo) if latest_mappo else None,
            output_dir=None,  # Use baseline directory
            display=False,  # Don't display, just save
        )
        print(f"\nPlots saved to: {latest_baseline}")
        print("Generated files:")
        print("  - baseline_distribution_comparison.png")
        print("  - focused_distribution_comparison.png")
        print("  - performance_ranking_comparison.png")

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback

        traceback.print_exc()


def show_available_experiments():
    """Show available baseline and MAPPO experiments."""
    print("\nAvailable experiments:")
    print("-" * 40)

    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        print("No experiments directory found!")
        return

    # Show baseline evaluations
    baseline_dirs = sorted(experiments_dir.glob("baseline_evaluation_*"))
    print(f"Baseline Evaluations ({len(baseline_dirs)}):")
    for d in baseline_dirs:
        print(f"  {d.name}")

    # Show MAPPO experiments
    mappo_dirs = sorted(experiments_dir.glob("mappo_*"))
    print(f"\nMAPPO Experiments ({len(mappo_dirs)}):")
    for d in mappo_dirs:
        print(f"  {d.name}")

    # Show other evaluations
    other_dirs = sorted(
        [
            d
            for d in experiments_dir.glob("*")
            if d.is_dir() and not d.name.startswith(("baseline_", "mappo_"))
        ]
    )
    if other_dirs:
        print(f"\nOther Experiments ({len(other_dirs)}):")
        for d in other_dirs:
            print(f"  {d.name}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        show_available_experiments()
    else:
        demo_baseline_plots()
