#!/usr/bin/env python3
"""
Calculate average throughput time from experiment log files.
Usage: python scripts/calculate_throughput.py <experiment_dir>
"""

import pandas as pd
import sys
import os
import glob

def calculate_throughput(experiment_dir):
    """Calculate average case throughput time from log files."""

    log_dir = os.path.join(experiment_dir, "logs")

    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        return

    log_files = glob.glob(os.path.join(log_dir, "log_*.csv"))

    if not log_files:
        print(f"Error: No log files found in {log_dir}")
        return

    print(f"Found {len(log_files)} log files")
    print("=" * 70)

    all_throughput_times = []

    for log_file in sorted(log_files):
        try:
            # Read the log file
            df = pd.read_csv(log_file)

            # Use task-level timestamps instead of case-level
            # Group by case_id to get first task assigned and last task completed
            cases = df.groupby('case_id').agg({
                'task_assigned_time': 'min',  # First task assigned
                'task_completed_time': 'max'  # Last task completed
            })

            # Convert to datetime
            cases['task_assigned_time'] = pd.to_datetime(cases['task_assigned_time'])
            cases['task_completed_time'] = pd.to_datetime(cases['task_completed_time'])

            # Calculate throughput time in seconds
            cases['throughput_time'] = (
                cases['task_completed_time'] - cases['task_assigned_time']
            ).dt.total_seconds()

            # Add to overall list
            all_throughput_times.extend(cases['throughput_time'].tolist())

            # Print stats for this log file
            avg_throughput = cases['throughput_time'].mean()
            print(f"{os.path.basename(log_file)}: {avg_throughput:.2f}s ({avg_throughput/60:.2f} min)")

        except Exception as e:
            print(f"Error processing {os.path.basename(log_file)}: {e}")

    # Calculate overall statistics
    if all_throughput_times:
        print("\n" + "=" * 70)
        print("OVERALL STATISTICS")
        print("=" * 70)

        avg_throughput = sum(all_throughput_times) / len(all_throughput_times)
        min_throughput = min(all_throughput_times)
        max_throughput = max(all_throughput_times)

        print(f"Total cases analyzed: {len(all_throughput_times)}")
        print(f"\n⏱️  AVERAGE THROUGHPUT TIME:")
        print(f"   {avg_throughput:.2f} seconds")
        print(f"   {avg_throughput/60:.2f} minutes")
        print(f"   {avg_throughput/3600:.2f} hours")
        print(f"\nMin throughput: {min_throughput:.2f}s ({min_throughput/60:.2f} min)")
        print(f"Max throughput: {max_throughput:.2f}s ({max_throughput/60:.2f} min)")

        # Calculate standard deviation
        mean = avg_throughput
        variance = sum((x - mean) ** 2 for x in all_throughput_times) / len(all_throughput_times)
        std_dev = variance ** 0.5
        print(f"Std deviation: {std_dev:.2f}s ({std_dev/60:.2f} min)")

        return avg_throughput
    else:
        print("No throughput data found")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/calculate_throughput.py <experiment_dir>")
        print("\nExample:")
        print("  python scripts/calculate_throughput.py experiments/mappo_20251221_143028")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    calculate_throughput(experiment_dir)
