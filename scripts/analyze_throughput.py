"""
Analyze experiment results and calculate average throughput time.
Usage: python scripts/analyze_throughput.py <experiment_dir>
"""

import pandas as pd
import sys
import os
import glob

def analyze_experiment(experiment_dir):
    """Analyze experiment and extract throughput metrics."""

    # Check if experiment directory exists
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory '{experiment_dir}' does not exist.")
        return

    print(f"Analyzing experiment: {experiment_dir}")
    print("=" * 60)

    # 1. Check training summary
    summary_file = os.path.join(experiment_dir, "training_summary.txt")
    if os.path.exists(summary_file):
        print("\n📊 TRAINING SUMMARY:")
        print("-" * 60)
        with open(summary_file, 'r') as f:
            print(f.read())

    # 2. Analyze log files
    log_dir = os.path.join(experiment_dir, "logs")
    if os.path.exists(log_dir):
        log_files = glob.glob(os.path.join(log_dir, "log_*.csv"))

        if log_files:
            print("\n📈 LOG FILE ANALYSIS:")
            print("-" * 60)

            for log_file in log_files:
                print(f"\nAnalyzing: {os.path.basename(log_file)}")

                try:
                    # Read the log CSV
                    df = pd.read_csv(log_file)

                    # Display basic info
                    print(f"  Total rows: {len(df)}")
                    print(f"  Columns: {list(df.columns)}")

                    # Calculate throughput time if available
                    if 'completion_timestamp' in df.columns and 'start_timestamp' in df.columns:
                        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
                        df['completion_timestamp'] = pd.to_datetime(df['completion_timestamp'])
                        df['throughput_time'] = (df['completion_timestamp'] - df['start_timestamp']).dt.total_seconds()

                        avg_throughput = df['throughput_time'].mean()
                        print(f"\n  ⏱️  AVERAGE THROUGHPUT TIME: {avg_throughput:.2f} seconds ({avg_throughput/60:.2f} minutes)")
                        print(f"  Min throughput: {df['throughput_time'].min():.2f} seconds")
                        print(f"  Max throughput: {df['throughput_time'].max():.2f} seconds")
                        print(f"  Std deviation: {df['throughput_time'].std():.2f} seconds")

                    # Show first few rows
                    print("\n  First 5 rows:")
                    print(df.head().to_string())

                except Exception as e:
                    print(f"  Error reading log file: {e}")
        else:
            print("\nNo log files found in logs directory.")
    else:
        print(f"\nLogs directory not found: {log_dir}")

    # 3. Check for final evaluation results
    eval_dir = os.path.join(experiment_dir, "final_evaluation")
    if os.path.exists(eval_dir):
        print("\n📋 FINAL EVALUATION RESULTS:")
        print("-" * 60)

        eval_logs = glob.glob(os.path.join(eval_dir, "logs", "log_*.csv"))
        if eval_logs:
            for log_file in eval_logs:
                print(f"\nEvaluation log: {os.path.basename(log_file)}")
                try:
                    df = pd.read_csv(log_file)

                    if 'completion_timestamp' in df.columns and 'start_timestamp' in df.columns:
                        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
                        df['completion_timestamp'] = pd.to_datetime(df['completion_timestamp'])
                        df['throughput_time'] = (df['completion_timestamp'] - df['start_timestamp']).dt.total_seconds()

                        avg_throughput = df['throughput_time'].mean()
                        print(f"  ⏱️  EVALUATION AVERAGE THROUGHPUT TIME: {avg_throughput:.2f} seconds ({avg_throughput/60:.2f} minutes)")
                except Exception as e:
                    print(f"  Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_throughput.py <experiment_dir>")
        print("\nExample:")
        print("  python scripts/analyze_throughput.py experiments/mappo_20251221_143028")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    analyze_experiment(experiment_dir)
