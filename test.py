# cpu_vs_gpu_benchmark.py
"""
Simple PyTorch benchmark comparing CPU vs GPU performance.

Run:
    python cpu_vs_gpu_benchmark.py

Notes:
- Requires PyTorch with CUDA support to test GPU.
- Adjust matrix size or iterations for heavier workloads.
"""

import time
import torch


def benchmark(device, size=4096, iterations=50):
    # Create tensors on the specified device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up (important for GPU)
    for _ in range(5):
        _ = torch.matmul(a, b)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iterations):
        _ = torch.matmul(a, b)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()
    return end - start


def main():
    size = 2048        # matrix dimension
    iterations = 100   # number of matmul operations

    print(f"Matrix size: {size} x {size}")
    print(f"Iterations: {iterations}")
    print("-" * 40)

    # CPU benchmark
    cpu_device = torch.device("cpu")
    cpu_time = benchmark(cpu_device, size, iterations)
    print(f"CPU time: {cpu_time:.4f} seconds")

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_time = benchmark(gpu_device, size, iterations)
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    else:
        print("CUDA not available. GPU test skipped.")


if __name__ == "__main__":
    main()
