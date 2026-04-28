#!/usr/bin/env python3
"""Profile the environment simulation to find bottlenecks."""
import cProfile
import pstats
import io
import sys
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import pandas as pd
from environment.simulator import AgentOptimizerEnvironment
from agents.mappo.agent import MAPPOAgent

print("=" * 70)
print("PROFILING ENVIRONMENT SIMULATION")
print("=" * 70)

# Load data
print("\nLoading data...")
df = pd.read_csv("data/cvs_pharmacy/processed/cvs_pharmacy.csv")
# Parse timestamps with ISO8601 format
df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], format='ISO8601')
df["end_timestamp"] = pd.to_datetime(df["end_timestamp"], format='ISO8601')
print(f"Loaded {len(df)} events")

# Create environment
print("Creating environment...")
env = AgentOptimizerEnvironment(
    data=df,
    simulation_parameters={"start_timestamp": df["start_timestamp"].min()},
    experiment_dir=None,
    enable_logging=False,
    verbose=False
)
print(f"Environment created with {len(env.agents)} agents")

# Create agent
print("Creating MAPPO agent...")
agent = MAPPOAgent(
    env=env,
    hidden_size=512,
    lr_actor=0.0003,
    lr_critic=0.0003,
    batch_size=200000,
)
print("Agent created")

print("\n" + "=" * 70)
print("RUNNING PROFILED SIMULATION (500 steps)")
print("=" * 70)

# Profile one episode (limited steps)
profiler = cProfile.Profile()
profiler.enable()

observations, _ = env.reset()
done = False
step_count = 0
max_steps = 500  # Profile just 500 steps to save time

while not done and step_count < max_steps:
    actions, action_probs = agent.select_actions(observations)
    observations, rewards, dones, truncated, _ = env.step(actions)
    done = any(dones.values()) or any(truncated.values())
    step_count += 1

profiler.disable()

print(f"Completed {step_count} steps")

# Analyze results
print("\n" + "=" * 70)
print("TOP 20 SLOWEST FUNCTIONS")
print("=" * 70)

s = io.StringIO()
stats = pstats.Stats(profiler, stream=s)
stats.sort_stats('cumulative')
stats.print_stats(20)

output = s.getvalue()
print(output)

# Find specific hotspots
print("\n" + "=" * 70)
print("ENVIRONMENT-SPECIFIC HOTSPOTS")
print("=" * 70)

s = io.StringIO()
stats = pstats.Stats(profiler, stream=s)
stats.sort_stats('cumulative')
stats.print_stats('environment/')

output = s.getvalue()
lines = output.split('\n')
relevant_lines = [l for l in lines if 'environment/' in l or 'ncalls' in l or 'tottime' in l]
print('\n'.join(relevant_lines[:30]))

print("\n" + "=" * 70)
print("OPTIMIZATION RECOMMENDATIONS")
print("=" * 70)

print("\nBased on the profile, focus on optimizing:")
print("  1. Top functions by 'tottime' (own time, not including subcalls)")
print("  2. Functions in environment/* that are called frequently")
print("  3. Any obvious inefficiencies (e.g., repeated calculations)")

print("\nCommon bottlenecks in simulation code:")
print("  • Sorting/searching in Python lists → Use heapq or numpy")
print("  • Dict lookups in tight loops → Cache results")
print("  • Repeated datetime operations → Convert to numeric timestamps")
print("  • Python loops → Vectorize with numpy where possible")

print("\n" + "=" * 70)
