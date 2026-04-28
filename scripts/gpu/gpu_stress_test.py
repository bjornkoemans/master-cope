#!/usr/bin/env python3
"""
GPU Stress Test Script
Performs intensive GPU computations for 30 seconds to verify GPU functionality.
"""

import time
import sys

try:
    import torch
except ImportError:
    print("PyTorch not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--break-system-packages"])
    import torch

def stress_test_gpu(duration=30):
    """
    Stress test the GPU by performing intensive matrix operations.

    Args:
        duration: Test duration in seconds (default: 30)
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. GPU stress test cannot run.")
        print("Your system may not have a compatible NVIDIA GPU or CUDA drivers installed.")
        return

    # Get GPU information
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB

    print("=" * 60)
    print("GPU STRESS TEST")
    print("=" * 60)
    print(f"GPU Detected: {gpu_name}")
    print(f"Total Memory: {gpu_memory:.2f} GB")
    print(f"Test Duration: {duration} seconds")
    print("=" * 60)
    print()

    # Set device
    device = torch.device('cuda:0')

    # Create large matrices for intensive computation
    # Adjust size based on available memory
    matrix_size = 4096

    print(f"Initializing {matrix_size}x{matrix_size} matrices on GPU...")

    try:
        # Allocate matrices on GPU
        matrix_a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        matrix_b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)

        print("Matrices allocated successfully")
        print(f"Memory allocated: {(matrix_a.element_size() * matrix_a.nelement() * 2) / (1024**3):.2f} GB")
        print()
        print("Starting stress test...")
        print("Press Ctrl+C to stop early")
        print()

        start_time = time.time()
        iteration = 0

        # Perform intensive GPU operations
        while (time.time() - start_time) < duration:
            # Matrix multiplication (very GPU intensive)
            result = torch.matmul(matrix_a, matrix_b)

            # Additional operations to increase load
            result = torch.sin(result)
            result = torch.exp(result / 1000)  # Scaled to avoid overflow
            result = torch.sqrt(torch.abs(result))

            # Update one matrix to keep computations varied
            matrix_a = result

            iteration += 1

            # Progress update every second
            elapsed = time.time() - start_time
            if iteration % 10 == 0:  # Update periodically
                memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
                utilization = (elapsed / duration) * 100

                print(f"Time: {elapsed:.1f}s | "
                      f"Iterations: {iteration} | "
                      f"Memory: {memory_used:.2f}GB used, {memory_cached:.2f}GB cached | "
                      f"Progress: {utilization:.0f}%")

        # Test completed
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print("STRESS TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total Time: {elapsed:.2f} seconds")
        print(f"Total Iterations: {iteration}")
        print(f"Average Iterations/sec: {iteration/elapsed:.2f}")
        print(f"Peak Memory Usage: {torch.cuda.max_memory_allocated(0) / (1024**3):.2f} GB")
        print()
        print("Your GPU appears to be working correctly.")
        print("=" * 60)

    except RuntimeError as e:
        print(f"\nError during GPU stress test: {e}")
        print("This might indicate insufficient GPU memory or driver issues.")
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\nTest interrupted after {elapsed:.2f} seconds")
        print(f"Completed {iteration} iterations before stopping.")
    finally:
        # Clean up
        torch.cuda.empty_cache()

if __name__ == "__main__":
    stress_test_gpu(duration=30)
