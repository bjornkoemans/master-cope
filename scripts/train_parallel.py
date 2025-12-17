#!/usr/bin/env python3
"""
Parallel Training Script
Runs multiple training processes simultaneously to utilize all CPU cores.
Each process runs an independent environment simulation on its own core.
"""

import argparse
import multiprocessing as mp
import subprocess
import time
from datetime import datetime
from pathlib import Path

def run_training(config_path, output_dir, name, gpu_id=0):
    """
    Run a single training process.

    Args:
        config_path: Path to config file
        output_dir: Output directory
        name: Experiment name
        gpu_id: GPU ID to use (for multi-GPU setups)
    """
    cmd = [
        "python", "scripts/train.py",
        "--episodes", "50",
        "--algorithm", "mappo",
    ]

    # Set GPU for this process (if multi-GPU)
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {name} on GPU {gpu_id}")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ {name} completed successfully")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ {name} failed")
        print(result.stderr)

    return result.returncode == 0


def parallel_training(num_processes=None):
    """
    Run multiple training processes in parallel.

    Args:
        num_processes: Number of parallel processes (default: CPU count)
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f"╔════════════════════════════════════════════════════════════════╗")
    print(f"║              Parallel MAPPO Training                           ║")
    print(f"╚════════════════════════════════════════════════════════════════╝")
    print(f"")
    print(f"Starting {num_processes} parallel training processes...")
    print(f"Each process will use 1 CPU core for environment simulation")
    print(f"All processes share the same GPU for policy updates")
    print(f"")
    print(f"Expected CPU usage: {num_processes} cores at ~100% during episodes")
    print(f"Expected GPU usage: 70-95% during policy updates")
    print(f"")

    # Create different experiment names
    experiments = [
        ("default", f"exp_{i:02d}")
        for i in range(num_processes)
    ]

    # Run in parallel using multiprocessing
    with mp.Pool(processes=num_processes) as pool:
        start_time = time.time()

        # Start all processes
        results = pool.starmap(
            run_training,
            [(config, "experiments", name, 0) for config, name in experiments]
        )

        elapsed = time.time() - start_time

    # Print summary
    successful = sum(results)
    print(f"")
    print(f"╔════════════════════════════════════════════════════════════════╗")
    print(f"║                    Training Complete                           ║")
    print(f"╚════════════════════════════════════════════════════════════════╝")
    print(f"")
    print(f"Results:")
    print(f"  Successful: {successful}/{num_processes}")
    print(f"  Failed:     {num_processes - successful}/{num_processes}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"")
    print(f"Expected speedup: {num_processes}x faster than sequential")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run parallel MAPPO training to utilize all CPU cores"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: number of CPU cores)"
    )

    args = parser.parse_args()

    parallel_training(num_processes=args.processes)
