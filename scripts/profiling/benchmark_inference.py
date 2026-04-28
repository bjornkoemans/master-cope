#!/usr/bin/env python3
"""Benchmark inference speed on CPU vs GPU."""
import sys
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import torch
import time
from agents.mappo.networks import ActorNetwork, CriticNetwork
from gymnasium import spaces
import numpy as np

print("=" * 70)
print("INFERENCE SPEED BENCHMARK: CPU vs GPU")
print("=" * 70)

# Create dummy observation space
obs_space = spaces.Dict({
    'case_id': spaces.Discrete(10000),
    'activity': spaces.Discrete(50),
    'queue_length': spaces.Discrete(20),
    'time_in_system': spaces.Box(low=0, high=100000, shape=(1,)),
    'next_activity': spaces.Discrete(50),
})

action_space = spaces.Discrete(10)

# Create networks
print("\nCreating actor network...")
actor_cpu = ActorNetwork(obs_space, action_space, hidden_size=128, device=torch.device('cpu'))
actor_gpu = ActorNetwork(obs_space, action_space, hidden_size=128, device=torch.device('cuda')).cuda()

print("Creating critic network...")
critic_cpu = CriticNetwork(obs_space, n_agents=8, hidden_size=512, device=torch.device('cpu'))
critic_gpu = CriticNetwork(obs_space, n_agents=8, hidden_size=512, device=torch.device('cuda')).cuda()

# Create dummy observation
dummy_obs = {
    'case_id': 100,
    'activity': 5,
    'queue_length': 3,
    'time_in_system': 1000.0,
    'next_activity': 6,
}

# Create dummy observations list for critic (8 agents)
dummy_obs_list = [dummy_obs for _ in range(8)]

n_iterations = 1000

print(f"\nBenchmarking {n_iterations} forward passes...")
print("=" * 70)

# Benchmark Actor CPU
print("\n1. ACTOR on CPU:")
start = time.time()
for _ in range(n_iterations):
    with torch.no_grad():
        _ = actor_cpu.act(dummy_obs, reward=0.0)
cpu_actor_time = time.time() - start
print(f"   Total: {cpu_actor_time:.3f}s")
print(f"   Per inference: {cpu_actor_time/n_iterations*1000:.3f}ms")
print(f"   Throughput: {n_iterations/cpu_actor_time:.1f} inferences/sec")

# Benchmark Actor GPU
print("\n2. ACTOR on GPU:")
torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    with torch.no_grad():
        _ = actor_gpu.act(dummy_obs, reward=0.0)
torch.cuda.synchronize()
gpu_actor_time = time.time() - start
print(f"   Total: {gpu_actor_time:.3f}s")
print(f"   Per inference: {gpu_actor_time/n_iterations*1000:.3f}ms")
print(f"   Throughput: {n_iterations/gpu_actor_time:.1f} inferences/sec")
print(f"   Speedup: {cpu_actor_time/gpu_actor_time:.2f}x")

# Benchmark Critic CPU
print("\n3. CRITIC on CPU:")
start = time.time()
for _ in range(n_iterations):
    with torch.no_grad():
        _ = critic_cpu(dummy_obs_list, reward=0.0)
cpu_critic_time = time.time() - start
print(f"   Total: {cpu_critic_time:.3f}s")
print(f"   Per inference: {cpu_critic_time/n_iterations*1000:.3f}ms")
print(f"   Throughput: {n_iterations/cpu_critic_time:.1f} inferences/sec")

# Benchmark Critic GPU
print("\n4. CRITIC on GPU:")
torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    with torch.no_grad():
        _ = critic_gpu(dummy_obs_list, reward=0.0)
torch.cuda.synchronize()
gpu_critic_time = time.time() - start
print(f"   Total: {gpu_critic_time:.3f}s")
print(f"   Per inference: {gpu_critic_time/n_iterations*1000:.3f}ms")
print(f"   Throughput: {n_iterations/gpu_critic_time:.1f} inferences/sec")
print(f"   Speedup: {cpu_critic_time/gpu_critic_time:.2f}x")

print("\n" + "=" * 70)
print("SIMULATION ESTIMATE")
print("=" * 70)

steps_per_episode = 62510  # Your typical episode length

# Calculate time for one episode
cpu_total = (cpu_actor_time + cpu_critic_time) / n_iterations * steps_per_episode
gpu_total = (gpu_actor_time + gpu_critic_time) / n_iterations * steps_per_episode

print(f"\nFor {steps_per_episode} steps per episode:")
print(f"  CPU inference time: {cpu_total:.1f}s")
print(f"  GPU inference time: {gpu_total:.1f}s")
print(f"  Expected speedup: {cpu_total/gpu_total:.2f}x")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if gpu_total < cpu_total * 0.5:
    print("\nGPU inference is significantly faster.")
    print("   Keeping models on GPU will speed up rollout collection.")
else:
    print("\nGPU speedup is marginal for small networks.")
    print("   The environment simulation itself may be the bottleneck.")

print("\nNext steps to speed up rollout collection:")
print("   1. Keep models on GPU (already done)")
print("   2. Profile environment.simulator to find hotspots")
print("   3. Vectorize environment operations")
print("   4. Use parallel environments (multiprocessing)")

print("\n" + "=" * 70)
