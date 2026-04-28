#!/usr/bin/env python3
"""Diagnose GPU utilization issues during training."""
import torch
import torch.nn as nn
import time

print("=" * 70)
print("PYTORCH + CUDA DIAGNOSTIC")
print("=" * 70)

# Check PyTorch build
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("\nCUDA NOT AVAILABLE.")
    exit(1)

# Check CUDA operations actually run on GPU
print("\n" + "=" * 70)
print("TESTING ACTUAL GPU EXECUTION")
print("=" * 70)

# Test 1: Simple matrix multiplication with GPU monitoring
print("\nTest 1: Matrix multiplication (should show GPU usage)")
size = 8192
x = torch.randn(size, size, device='cuda')
y = torch.randn(size, size, device='cuda')

print("  Running 10 large matrix multiplications...")
print("  → Check nvidia-smi NOW - you should see GPU-Util spike!")

for i in range(10):
    z = x @ y
    torch.cuda.synchronize()  # Force wait for GPU
    print(f"    Iteration {i+1}/10 complete")
    time.sleep(0.5)  # Give time to observe in nvidia-smi

# Test 2: Neural network training
print("\nTest 2: Neural network training (your actual use case)")
print("  Creating networks similar to your MAPPO setup...")

class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

model = TestNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

batch_size = 2048
x_train = torch.randn(batch_size, 100, device='cuda')
y_train = torch.randn(batch_size, 10, device='cuda')

print(f"  Training with batch_size={batch_size} for 100 iterations...")
print("  → Check nvidia-smi NOW - you should see GPU-Util around 30-80%!")

start = time.time()
for i in range(100):
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (i + 1) % 20 == 0:
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"    Iteration {i+1}/100 - Time: {elapsed:.2f}s - Loss: {loss.item():.4f}")

torch.cuda.synchronize()
total_time = time.time() - start
print(f"  Total training time: {total_time:.2f}s")
print(f"  Time per iteration: {total_time/100*1000:.2f}ms")

# Test 3: Check for CPU fallback
print("\n" + "=" * 70)
print("CHECKING FOR CPU FALLBACK")
print("=" * 70)

# Create identical operations on CPU and GPU
print("\nComparing CPU vs GPU speeds (should be very different)...")

# CPU test
model_cpu = TestNet().cpu()
x_cpu = torch.randn(batch_size, 100)
start = time.time()
for _ in range(10):
    _ = model_cpu(x_cpu)
cpu_time = time.time() - start

# GPU test
model_gpu = TestNet().cuda()
x_gpu = torch.randn(batch_size, 100, device='cuda')
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    _ = model_gpu(x_gpu)
torch.cuda.synchronize()
gpu_time = time.time() - start

print(f"  CPU time (10 forward passes): {cpu_time:.4f}s")
print(f"  GPU time (10 forward passes): {gpu_time:.4f}s")
print(f"  Speedup: {cpu_time/gpu_time:.2f}x")

if cpu_time / gpu_time < 2:
    print("\nWARNING: GPU speedup is suspiciously low.")
    print("  Possible issues:")
    print("    1. GPU operations falling back to CPU")
    print("    2. PyTorch not compiled with proper CUDA support")
    print("    3. Driver/CUDA version mismatch")
else:
    print("\nGPU appears to be working correctly.")

print("\n" + "=" * 70)
print("GPU MEMORY STATUS")
print("=" * 70)
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.4f} GB")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print("\nWhile this script is running, check:")
print("  watch -n 0.5 'nvidia-smi; echo; mpstat -P ALL 1 1'")
print("\nIf GPU-Util stays at 0%, there's a problem with your PyTorch/CUDA setup.")
