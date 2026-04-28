#!/usr/bin/env python3
"""Profile where time is spent during training."""
import time
import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("TRAINING BOTTLENECK PROFILER")
print("=" * 70)

# Simulate your data structure
n_samples = 62510
n_agents = 8
batch_size = 2048

print(f"\nSimulating MAPPO training:")
print(f"  Samples: {n_samples}")
print(f"  Agents: {n_agents}")
print(f"  Batch size: {batch_size}")
print(f"  Batches per epoch: {(n_samples + batch_size - 1) // batch_size}")

# Create mock observation structure (like yours)
observations = []
for i in range(n_samples):
    obs_dict = {}
    for agent_id in range(n_agents):
        obs_dict[agent_id] = {
            'case_id': i,
            'activity': np.random.randint(0, 10),
            'queue_length': np.random.randint(0, 5),
            'time_in_system': np.random.random(),
            'next_activity': np.random.randint(0, 10),
        }
    observations.append(obs_dict)

rewards = [np.random.random() for _ in range(n_samples)]

# Mock agents
class MockAgent:
    def __init__(self, id):
        self.id = id

agents = [MockAgent(i) for i in range(n_agents)]

print("\n" + "=" * 70)
print("TIMING BREAKDOWN (1 EPOCH)")
print("=" * 70)

total_start = time.time()

# Step 1: Create random permutation
step_start = time.time()
indices = torch.randperm(n_samples, device='cuda')
step_time = time.time() - step_start
print(f"\n1. Create permutation: {step_time*1000:.2f}ms")

# Step 2: Loop through batches
data_prep_time = 0
gpu_time = 0
num_batches = (n_samples + batch_size - 1) // batch_size

for start_idx in range(0, n_samples, batch_size):
    end_idx = min(start_idx + batch_size, n_samples)
    batch_indices = indices[start_idx:end_idx]

    # DATA PREPARATION (CPU-bound Python)
    prep_start = time.time()

    batch_indices_cpu = batch_indices.cpu().numpy()
    batch_obs = [observations[i] for i in batch_indices_cpu]

    batch_obs_lists = []
    batch_rewards = []

    for j, obs_dict in enumerate(batch_obs):
        obs_list = [obs_dict[agent.id] for agent in agents]
        batch_idx = batch_indices_cpu[j]
        reward = rewards[batch_idx]
        batch_obs_lists.append(obs_list)
        batch_rewards.append(reward)

    data_prep_time += time.time() - prep_start

    # GPU COMPUTATION (simulate your network forward/backward)
    gpu_start = time.time()
    # Simulate processing the batch on GPU
    dummy_tensor = torch.randn(len(batch_obs), 256, device='cuda')
    dummy_output = torch.relu(dummy_tensor @ torch.randn(256, 256, device='cuda'))
    torch.cuda.synchronize()
    gpu_time += time.time() - gpu_start

total_time = time.time() - total_start

print(f"2. Data preparation (Python loops): {data_prep_time*1000:.2f}ms ({data_prep_time/total_time*100:.1f}%)")
print(f"3. GPU computation: {gpu_time*1000:.2f}ms ({gpu_time/total_time*100:.1f}%)")
print(f"4. Other overhead: {(total_time-data_prep_time-gpu_time)*1000:.2f}ms ({(total_time-data_prep_time-gpu_time)/total_time*100:.1f}%)")
print(f"\nTotal epoch time: {total_time:.2f}s")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

if data_prep_time / total_time > 0.7:
    print("\nBOTTLENECK IDENTIFIED: Data preparation (Python loops)")
    print("\nYour H200 is sitting idle while Python prepares data!")
    print("\nSolutions:")
    print("  1. Pre-convert all observations to tensors before training loop")
    print("  2. Use vectorized operations instead of Python loops")
    print("  3. Increase batch size to reduce loop overhead")
    print("  4. Use PyTorch DataLoader with num_workers > 0")
else:
    print("\nGPU is the bottleneck (this is good).")

print("\n" + "=" * 70)
print(f"Your H200 is being utilized only {gpu_time/total_time*100:.1f}% of the time!")
print("=" * 70)
