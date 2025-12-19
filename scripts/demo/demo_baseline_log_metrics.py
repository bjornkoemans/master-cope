#!/usr/bin/env python3
"""
Demo script showing how to use the baseline log metrics plotting functionality.
"""
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import the plotting function
from analysis.plot_baseline_log_metrics import plot_baseline_log_metrics


def demo_baseline_log_metrics_plots():
    """Demo the baseline log metrics plotting functionality."""
    print("=" * 70)
    print("BASELINE LOG METRICS DISTRIBUTION PLOTTING DEMO")
    print("=" * 70)

    # Find available baseline evaluation directories
    experiments_dir = Path("experiments")
    baseline_dirs = list(experiments_dir.glob("baseline_evaluation_*"))
    mappo_dirs = list(experiments_dir.glob("mappo_*"))

    if not baseline_dirs:
        print("No baseline evaluation directories found!")
        print(
            "Run `python src/main.py baseline` first to generate baseline evaluation results."
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

    # Create the log metrics comparison plots
    print("\nGenerating baseline log metrics comparison plots...")
    try:
        plot_baseline_log_metrics(
            baseline_dir=str(latest_baseline),
            mappo_dir=str(latest_mappo) if latest_mappo else None,
            output_dir=None,  # Use baseline directory
            display=False,  # Don't display, just save
        )
        print(f"\nPlots saved to: {latest_baseline}")
        print("Generated files:")
        print("  - baseline_log_metrics_comparison.png")
        print("  - baseline_log_metrics_detailed.png")

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


def main():
    """Main function with command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo baseline log metrics plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_baseline_log_metrics.py                    # Plot most recent baseline evaluation
  python demo_baseline_log_metrics.py --list            # Show available experiments
  python demo_baseline_log_metrics.py --baseline DIR    # Plot specific baseline directory
  python demo_baseline_log_metrics.py --baseline DIR --mappo MAPPO_DIR  # Include MAPPO comparison
        """,
    )

    parser.add_argument(
        "--list", action="store_true", help="Show available experiments"
    )
    parser.add_argument("--baseline", help="Specific baseline directory to plot")
    parser.add_argument("--mappo", help="MAPPO directory for comparison")
    parser.add_argument(
        "--display", action="store_true", help="Display plots instead of just saving"
    )

    args = parser.parse_args()

    if args.list:
        show_available_experiments()
    elif args.baseline:
        if not os.path.exists(args.baseline):
            print(f"Error: Baseline directory '{args.baseline}' does not exist!")
            return

        print(f"Plotting baseline log metrics from: {args.baseline}")
        if args.mappo:
            if not os.path.exists(args.mappo):
                print(
                    f"Warning: MAPPO directory '{args.mappo}' does not exist! Proceeding without MAPPO data."
                )
                args.mappo = None
            else:
                print(f"Including MAPPO comparison from: {args.mappo}")

        try:
            plot_baseline_log_metrics(
                baseline_dir=args.baseline,
                mappo_dir=args.mappo,
                output_dir=None,
                display=args.display,
            )
            print("\nPlots generated successfully!")
        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback

            traceback.print_exc()
    else:
        demo_baseline_log_metrics_plots()


if __name__ == "__main__":
    main()
