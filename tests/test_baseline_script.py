#!/usr/bin/env python3
"""
Quick test script to run baseline evaluation on a single dataset.
Used to verify the consolidated baseline script works correctly.
"""
import subprocess
import sys


def main():
    """Run a quick test evaluation."""
    print("Running quick baseline evaluation test...")

    # Run the main script with minimal parameters
    cmd = [
        sys.executable,
        "run_multi_dataset_baseline_evaluation.py",
        "--datasets",
        "Train_Preprocessed",  # Use the known working dataset
        "--episodes",
        "5",  # Fewer episodes for quick test
        "--seed",
        "42",
        "--use-test-data",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ Test completed successfully!")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ Test failed!")
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            print("STDOUT:", result.stdout[-500:])

    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running test: {e}")


if __name__ == "__main__":
    main()
