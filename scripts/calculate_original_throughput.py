#!/usr/bin/env python3
"""
Calculate average throughput time from original event log data.
Usage: python scripts/calculate_original_throughput.py [data_file]
"""

import pandas as pd
import sys
import os

def calculate_original_throughput(data_file=None):
    """Calculate average case throughput time from original event log."""

    if data_file is None:
        data_file = "data/input/train_preprocessed.csv"

    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return

    print(f"Analyzing original event log: {data_file}")
    print("=" * 70)

    # Read the data
    df = pd.read_csv(data_file)

    print(f"Total events: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Ensure we have the required columns
    if 'case_id' not in df.columns:
        print("Error: 'case_id' column not found")
        return
    if 'start_timestamp' not in df.columns or 'end_timestamp' not in df.columns:
        print("Error: timestamp columns not found")
        print(f"Available columns: {list(df.columns)}")
        return

    # Group by case to get start and end times
    case_times = df.groupby('case_id').agg({
        'start_timestamp': 'min',  # First event start time
        'end_timestamp': 'max'      # Last event end time
    })

    # Convert to datetime
    case_times['start_timestamp'] = pd.to_datetime(case_times['start_timestamp'])
    case_times['end_timestamp'] = pd.to_datetime(case_times['end_timestamp'])

    # Calculate throughput time (case duration)
    case_times['throughput_time'] = (
        case_times['end_timestamp'] - case_times['start_timestamp']
    ).dt.total_seconds()

    # Calculate statistics
    print(f"\nTotal cases: {len(case_times)}")
    print("\n" + "=" * 70)
    print("ORIGINAL DATA THROUGHPUT STATISTICS")
    print("=" * 70)

    avg_throughput = case_times['throughput_time'].mean()
    median_throughput = case_times['throughput_time'].median()
    min_throughput = case_times['throughput_time'].min()
    max_throughput = case_times['throughput_time'].max()
    std_throughput = case_times['throughput_time'].std()

    print(f"\n⏱️  AVERAGE THROUGHPUT TIME:")
    print(f"   {avg_throughput:.2f} seconds")
    print(f"   {avg_throughput/60:.2f} minutes")
    print(f"   {avg_throughput/3600:.2f} hours")

    print(f"\n📊 MEDIAN THROUGHPUT TIME:")
    print(f"   {median_throughput:.2f} seconds")
    print(f"   {median_throughput/60:.2f} minutes")
    print(f"   {median_throughput/3600:.2f} hours")

    print(f"\nMin throughput: {min_throughput:.2f}s ({min_throughput/60:.2f} min)")
    print(f"Max throughput: {max_throughput:.2f}s ({max_throughput/60:.2f} min)")
    print(f"Std deviation: {std_throughput:.2f}s ({std_throughput/60:.2f} min)")

    # Show distribution
    print("\n" + "=" * 70)
    print("THROUGHPUT TIME DISTRIBUTION")
    print("=" * 70)

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = case_times['throughput_time'].quantile(p/100)
        print(f"  {p}th percentile: {value:.2f}s ({value/60:.2f} min)")

    return avg_throughput

if __name__ == "__main__":
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    calculate_original_throughput(data_file)
