#!/usr/bin/env python3
"""Find the maximum safe batch size for your GPU without crashing."""
import sys
import os
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import torch
from agents.mappo.agent import MAPPOAgent
from environment.scheduling_env import SchedulingEnvironment
import pandas as pd

print("=" * 70)
print("BATCH SIZE OPTIMIZER FOR H200")
print("=" * 70)

# Check GPU
if not torch.cuda.is_available():
    print("CUDA not available.")
    sys.exit(1)

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load data and create environment (minimal setup)
print("\n" + "=" * 70)
print("LOADING DATA AND CREATING ENVIRONMENT")
print("=" * 70)

df = pd.read_csv("data/cvs_pharmacy/processed/cvs_pharmacy.csv")
print(f"Loaded {len(df)} events")

# Create a small environment for testing
env = SchedulingEnvironment(df, max_agents=8, max_queue_size=10)
print(f"Environment created with {len(env.agents)} agents")

# Create agent with default settings
print("\n" + "=" * 70)
print("CREATING MAPPO AGENT")
print("=" * 70)

agent = MAPPOAgent(
    env=env,
    hidden_size=512,  # Use your actual hidden size
    lr_actor=0.0003,
    lr_critic=0.0003,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_loss_coef=0.5,
    num_epochs=1,  # Just 1 epoch for testing
    batch_size=1024,  # Will override this
    device=torch.device('cuda')
)

print("Agent created")

# Generate some dummy data for testing
print("\n" + "=" * 70)
print("GENERATING DUMMY TRAINING DATA")
print("=" * 70)

n_samples = 62510  # Your actual training size
print(f"Generating {n_samples} dummy samples...")

# Create dummy observations
dummy_observations = []
for i in range(n_samples):
    obs_dict = {}
    for agent_id in range(len(env.agents)):
        obs_dict[agent_id] = {
            'case_id': i,
            'activity': 0,
            'queue_length': 0,
            'time_in_system': 0.0,
            'next_activity': 0,
        }
    dummy_observations.append(obs_dict)

# Create dummy returns and advantages
dummy_returns = torch.randn(n_samples, device='cuda')
dummy_advantages = torch.randn(n_samples, device='cuda')

print("Dummy data generated")

# Function to test a batch size
def test_batch_size(batch_size, verbose=False):
    """Test if a batch size works without OOM."""
    try:
        # Clear GPU cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if verbose:
            print(f"\n  Testing batch_size={batch_size}...")

        # Simulate one training batch
        indices = torch.randperm(min(batch_size * 4, n_samples), device='cuda')[:batch_size]
        batch_indices_cpu = indices.cpu().numpy()

        # Preprocess observations (like in actual training)
        preprocessed_obs_lists = []
        preprocessed_rewards = []
        for i in batch_indices_cpu:
            obs_list = [dummy_observations[i][agent.id] for agent in env.agents]
            preprocessed_obs_lists.append(obs_list)
            preprocessed_rewards.append(0.0)

        # Test critic forward + backward
        batch_returns = dummy_returns[indices]
        value_preds = agent.critic.forward_batch(preprocessed_obs_lists, preprocessed_rewards)
        critic_loss = ((value_preds.squeeze() - batch_returns.squeeze()) ** 2).mean()

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # Test actor forward + backward (for one agent)
        actor = agent.actors[0]
        optimizer = agent.actor_optimizers[0]

        batch_agent_obs = [dummy_observations[i][0] for i in batch_indices_cpu]
        batch_agent_rewards = [0.0] * len(batch_indices_cpu)

        action_probs = actor.forward_batch(batch_agent_obs, batch_agent_rewards)
        actor_loss = -action_probs.mean()  # Dummy loss

        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()

        # Get memory stats
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9

        if verbose:
            print(f"    Success. Peak memory: {peak:.2f} GB")

        return True, peak

    except RuntimeError as e:
        if "out of memory" in str(e):
            if verbose:
                print(f"    OOM at batch_size={batch_size}")
            torch.cuda.empty_cache()
            return False, None
        else:
            raise e

# Binary search for maximum batch size
print("\n" + "=" * 70)
print("FINDING MAXIMUM BATCH SIZE")
print("=" * 70)

min_batch = 1024
max_batch = 65536  # Start with a high upper bound

print(f"\nSearching between {min_batch} and {max_batch}...")

# First, find rough upper bound
test_sizes = [2048, 4096, 8192, 16384, 32768, 65536]
last_working = min_batch
last_peak_memory = 0

print("\nPhase 1: Finding approximate maximum...")
for size in test_sizes:
    success, peak = test_batch_size(size, verbose=True)
    if success:
        last_working = size
        last_peak_memory = peak
    else:
        max_batch = size
        break

# Binary search for exact maximum
print("\nPhase 2: Binary search for exact maximum...")
min_batch = last_working
iterations = 0
max_iterations = 10

while max_batch - min_batch > 512 and iterations < max_iterations:
    mid = (min_batch + max_batch) // 2
    mid = (mid // 512) * 512  # Round to nearest 512

    success, peak = test_batch_size(mid, verbose=True)

    if success:
        min_batch = mid
        last_working = mid
        last_peak_memory = peak
    else:
        max_batch = mid

    iterations += 1

max_safe_batch = last_working

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nMaximum working batch size: {max_safe_batch}")
print(f"  Peak GPU memory usage: {last_peak_memory:.2f} GB")

# Recommend 80% of max for safety
recommended = int(max_safe_batch * 0.8)
recommended = (recommended // 512) * 512  # Round down to nearest 512

print(f"\nRECOMMENDED batch size: {recommended}")
print(f"   (80% of max for safety margin)")

# Calculate expected performance
total_samples = 62510
batches_per_epoch_current = (total_samples + 2048 - 1) // 2048
batches_per_epoch_recommended = (total_samples + recommended - 1) // recommended

print(f"\nIMPACT:")
print(f"   Current (2048): {batches_per_epoch_current} batches/epoch")
print(f"   Recommended ({recommended}): {batches_per_epoch_recommended} batches/epoch")
print(f"   Reduction: {batches_per_epoch_current - batches_per_epoch_recommended} fewer batches")
print(f"   Expected speedup: ~{batches_per_epoch_current/batches_per_epoch_recommended:.1f}x")

print("\n" + "=" * 70)
print("UPDATE YOUR CONFIG")
print("=" * 70)
print(f"\nEdit src/configs/phase_b_standard_rl/mappo_baseline.yaml:")
print(f"  batch_size: {recommended}")
print(f"  buffer_size: {recommended}")

print("\n" + "=" * 70)
print("BATCH SIZE OPTIMIZATION COMPLETE")
print("=" * 70)
