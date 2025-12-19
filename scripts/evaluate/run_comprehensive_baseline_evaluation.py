#!/usr/bin/env python3
"""
Run comprehensive baseline evaluation across all datasets with trained models.
This script runs the consolidated baseline evaluation including MAPPO and QMIX agents.
"""
import os
import sys
import subprocess
from datetime import datetime


def run_comprehensive_evaluation():
    """Run the comprehensive baseline evaluation."""
    print("=" * 80)
    print("COMPREHENSIVE BASELINE EVALUATION")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting evaluation at: {timestamp}")

    # Check for available models
    mappo_model_path = "./models/mappo_final"
    qmix_model_path = "./models/qmix_final"  # Adjust if different

    mappo_available = os.path.exists(mappo_model_path)
    qmix_available = os.path.exists(qmix_model_path)

    print(f"MAPPO model available: {mappo_available} ({mappo_model_path})")
    print(f"QMIX model available: {qmix_available} ({qmix_model_path})")

    # Build command
    cmd = [
        sys.executable,
        "run_multi_dataset_baseline_evaluation.py",
        "--datasets",
        "all",  # Run on all available datasets
        "--episodes",
        "20",  # Full evaluation episodes
        "--seed",
        "42",
        "--use-test-data",  # Use test data for fair comparison
    ]

    # Add trained models if available
    if mappo_available or qmix_available:
        cmd.extend(["--include-trained"])

        if mappo_available:
            cmd.extend(["--mappo-model-path", mappo_model_path])

        if qmix_available:
            cmd.extend(["--qmix-model-path", qmix_model_path])

    print("\nRunning command:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)

    try:
        # Run the evaluation
        subprocess.run(cmd, check=True)

        print("\n" + "=" * 80)
        print("✅ COMPREHENSIVE EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Error running evaluation: {e}")
        return False


def main():
    """Main function."""
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"Working directory: {os.getcwd()}")

    # Run the evaluation
    success = run_comprehensive_evaluation()

    if success:
        print("\n🎉 All baseline evaluations completed!")
        print("\nCheck the experiments/ directory for results.")
        print("\nEach dataset will have its own directory with results files:")
        print("  - baseline_comparison_results_<dataset>.json")
        print("  - multi_dataset_summary_<timestamp>.json")
    else:
        print("\n💥 Evaluation failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
