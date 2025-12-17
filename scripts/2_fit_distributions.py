#!/usr/bin/env python3
"""
Phase 2: Fit Duration Distributions
Fits probability distributions to task durations using training data only.
Run this ONCE per dataset.
"""

import argparse
import pickle
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.display import print_colored
from src.utils.duration_fitting import (
    fit_duration_distributions_on_training_data,
    save_fitted_distributions,
    print_distribution_summary
)


def fit_distributions(data_file, output_file=None):
    """
    Fit duration distributions on training data.

    Args:
        data_file: Path to preprocessed data pickle
        output_file: Path to save fitted distributions
    """
    print_colored("="*60, "yellow")
    print_colored("PHASE 2: FIT DURATION DISTRIBUTIONS", "yellow")
    print_colored("="*60, "yellow")

    # Load preprocessed data
    print_colored(f"\n1. Loading preprocessed data from {data_file}...", "blue")
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    train_data = data_dict['train']
    stats = data_dict.get('stats', {})

    print_colored(f"   Train set: {len(train_data)} events", "green")
    if stats:
        print_colored(f"   Total dataset: {stats.get('total_events', 'N/A')} events", "green")

    # Fit distributions
    print_colored("\n2. Fitting duration distributions on training data...", "blue")
    print_colored("   This may take a few minutes...", "cyan")

    fitted_distributions = fit_duration_distributions_on_training_data(train_data)
    activity_durations_dict, stats_dict, global_activity_medians = fitted_distributions

    # Print summary
    print_colored("\n3. Distribution Summary:", "blue")
    print_distribution_summary(activity_durations_dict, stats_dict)

    # Save fitted distributions
    if output_file is None:
        # Default output path
        dataset_name = Path(data_file).stem.replace('_processed', '')
        output_file = f"data/distributions/{dataset_name}_distributions.pkl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print_colored(f"\n4. Saving fitted distributions to {output_file}...", "blue")
    save_fitted_distributions(*fitted_distributions, output_file)
    print_colored(f"   Saved successfully!", "green")

    print_colored("\n" + "="*60, "yellow")
    print_colored("✅ DISTRIBUTION FITTING COMPLETE", "green")
    print_colored("="*60, "yellow")
    print_colored(f"\nFitted distributions saved to: {output_file}", "cyan")
    print_colored(f"\nNext step: python scripts/3_train.py \\", "cyan")
    print_colored(f"             --data {data_file} \\", "cyan")
    print_colored(f"             --distributions {output_file} \\", "cyan")
    print_colored(f"             --config configs/default.yaml", "cyan")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Fit duration distributions on training data"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preprocessed data pickle (from phase 1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for fitted distributions (default: data/distributions/DATASET_distributions.pkl)"
    )

    args = parser.parse_args()

    fit_distributions(args.data, args.output)
