#!/usr/bin/env python3
"""
Phase 1: Data Preparation
Loads event log and preprocesses it for training.
Run this ONCE per dataset.
"""

import argparse
import pickle
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import config
from src.core.display import print_colored
from src.preprocessing.load_data import load_data, split_data
from src.preprocessing.preprocessing import remove_short_cases


def prepare_data(input_file=None, output_file=None, test_size=0.17, remove_short=True):
    """
    Prepare data for training.

    Args:
        input_file: Path to input CSV (if None, uses config)
        output_file: Path to save processed data
        test_size: Fraction of data for test set
        remove_short: Whether to remove short cases
    """
    print_colored("="*60, "yellow")
    print_colored("PHASE 1: DATA PREPARATION", "yellow")
    print_colored("="*60, "yellow")

    # Use input file or config
    if input_file:
        # Temporarily override config
        original_filename = config["input_filename"]
        config["input_filename"] = input_file

    # Load data
    print_colored(f"\n1. Loading data from {config['input_filename']}...", "blue")
    data = load_data(config)
    print_colored(f"   Loaded {len(data)} events", "green")

    # Remove short cases if requested
    if remove_short:
        print_colored("\n2. Removing short cases (< 3 activities)...", "blue")
        before = len(data)
        data = remove_short_cases(data)
        after = len(data)
        print_colored(f"   Removed {before - after} events ({(before-after)/before*100:.1f}%)", "green")

    # Split data
    print_colored(f"\n3. Splitting data (train/test = {1-test_size:.0%}/{test_size:.0%})...", "blue")
    train, test = split_data(data, test_size=test_size)
    print_colored(f"   Train set: {len(train)} events", "green")
    print_colored(f"   Test set: {len(test)} events", "green")

    # Save processed data
    if output_file is None:
        # Default output path
        dataset_name = Path(config["input_filename"]).stem
        output_file = f"data/processed/{dataset_name}_processed.pkl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print_colored(f"\n4. Saving processed data to {output_file}...", "blue")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'train': train,
            'test': test,
            'config': config.copy(),
            'stats': {
                'total_events': len(data),
                'train_events': len(train),
                'test_events': len(test),
                'test_size': test_size
            }
        }, f)

    print_colored(f"   Saved successfully!", "green")

    # Restore original config if needed
    if input_file:
        config["input_filename"] = original_filename

    print_colored("\n" + "="*60, "yellow")
    print_colored("✅ DATA PREPARATION COMPLETE", "green")
    print_colored("="*60, "yellow")
    print_colored(f"\nProcessed data saved to: {output_file}", "cyan")
    print_colored(f"Next step: python scripts/2_fit_distributions.py --data {output_file}", "cyan")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: Prepare data for training"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file path (if not specified, uses config.py)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output pickle file path (default: data/processed/DATASET_processed.pkl)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.17,
        help="Fraction of data for test set (default: 0.17)"
    )
    parser.add_argument(
        "--keep-short",
        action="store_true",
        help="Keep short cases (< 3 activities)"
    )

    args = parser.parse_args()

    prepare_data(
        input_file=args.input,
        output_file=args.output,
        test_size=args.test_size,
        remove_short=not args.keep_short
    )
