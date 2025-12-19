# cpu_vs_accelerator_benchmark.py
"""
PyTorch benchmark comparing CPU vs GPU/Accelerator performance.

Supports:
- CPU (all platforms)
- CUDA (NVIDIA GPUs on Linux/Windows)
- MPS (Apple Silicon on macOS)

Run:
    python cpu_vs_accelerator_benchmark.py
"""

import time
import torch


def get_accelerator_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA (NVIDIA GPU)"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple Silicon GPU)"
    return None, None


def benchmark(device, size=2048, iterations=100):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up
    for _ in range(5):
        _ = torch.matmul(a, b)
    synchronize(device)

    start = time.perf_counter()

    for _ in range(iterations):
        _ = torch.matmul(a, b)

    synchronize(device)
    end = time.perf_counter()

    return end - start


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    size = 2048
    iterations = 100

    print(f"Matrix size: {size} x {size}")
    print(f"Iterations: {iterations}")
    print("-" * 40)

    # CPU
    cpu_device = torch.device("cpu")
    cpu_time = benchmark(cpu_device, size, iterations)
    print(f"CPU time: {cpu_time:.4f} seconds")

    # Accelerator (CUDA or MPS)
    acc_device, acc_name = get_accelerator_device()
    if acc_device is not None:
        acc_time = benchmark(acc_device, size, iterations)
        print(f"{acc_name} time: {acc_time:.4f} seconds")
        print(f"Speedup: {cpu_time / acc_time:.2f}x")
    else:
        print("No GPU/accelerator available. Only CPU tested.")


if __name__ == "__main__":
    main()
