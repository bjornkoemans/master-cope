#!/usr/bin/env python3
"""
Profile the environment step() function to identify bottlenecks.

Usage:
    python profile_env.py --config src/configs/cvs/donkey_a/cvs_mappo_collab.yaml
    python profile_env.py --config src/configs/cvs/donkey_a/cvs_mappo_collab.yaml --steps 2000
    python profile_env.py --config src/configs/cvs/donkey_a/cvs_mappo_collab.yaml --line-profile

Output:
    - cProfile summary (top 40 functions by cumulative time)
    - Per-section timing breakdown of step() internals
    - Optional: line-by-line profile of step() (with --line-profile)
"""
import argparse
import cProfile
import pstats
import sys
import time
import io
import random
import numpy as np
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from configs.config_loader import ExperimentConfig
from preprocessing.data_loader import load_and_preprocess_data
from environment.simulator import AgentOptimizerEnvironment


def create_env(config: ExperimentConfig):
    """Initialize environment from config (same as training/trainer.py)."""
    collaboration_config = config.get("agent.collaboration", None)
    train_data, _ = load_and_preprocess_data(
        data_path=config.get("data.input_file"),
        train_split=config.get("data.train_split", 0.8),
        min_case_length=config.get("data.min_case_length", 3),
        collaboration_config=collaboration_config,
    )

    reward_config = config.get("reward", None)
    work_schedule_enabled = config.get("environment.work_schedule_enabled", False)

    env = AgentOptimizerEnvironment(
        data=train_data,
        simulation_parameters={
            "start_timestamp": train_data["assign_timestamp"].min()
            if "assign_timestamp" in train_data.columns
            else train_data["start_timestamp"].min()
        },
        experiment_dir=None,
        enable_logging=False,  # Disable logging for profiling
        verbose=True,
        max_steps=config.get("environment.max_steps", 200_000),
        max_episodes=config.get("environment.max_episodes", 1000),
        reward_config=reward_config,
        work_schedule_enabled=work_schedule_enabled,
    )
    return env


def random_actions(env):
    """Generate random actions for all agents."""
    return {agent.id: random.randint(0, 1) for agent in env.agents}


def run_profiled_episode(env, max_steps: int):
    """Run one episode with cProfile."""
    obs, _ = env.reset()
    actions = random_actions(env)

    print(f"\nRunning {max_steps} steps with cProfile...")
    print(f"Agents: {len(env.agents)}, Activities: {env.num_activities}")
    print(f"Future cases: {len(env.future_cases)}, Pending: {len(env.pending_cases)}")

    profiler = cProfile.Profile()
    profiler.enable()

    step_count = 0
    done = False
    t0 = time.perf_counter()

    while not done and step_count < max_steps:
        obs, rewards, terminations, truncations, _ = env.step(actions)
        done = any(terminations.values()) or any(truncations.values())
        if not done:
            actions = random_actions(env)
        step_count += 1

    elapsed = time.perf_counter() - t0
    profiler.disable()

    print(f"\nCompleted {step_count} steps in {elapsed:.2f}s ({step_count/elapsed:.0f} steps/s)")

    # Print profile sorted by cumulative time
    print("\n" + "=" * 80)
    print("  cPROFILE — TOP 40 BY CUMULATIVE TIME")
    print("=" * 80)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(40)
    print(stream.getvalue())

    # Also show by tottime (self time, excluding callees)
    print("=" * 80)
    print("  cPROFILE — TOP 40 BY SELF TIME (tottime)")
    print("=" * 80)
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.strip_dirs()
    stats2.sort_stats("tottime")
    stats2.print_stats(40)
    print(stream2.getvalue())

    return profiler


def run_manual_timing(env, max_steps: int):
    """Run one episode with manual section timing to break down step() costs."""
    obs, _ = env.reset()
    actions = random_actions(env)

    print(f"\nRunning {max_steps} steps with manual timing breakdown...")

    # Timing accumulators
    t_step_total = 0.0
    t_action_handling = 0.0
    t_work_case = 0.0
    t_filter_completed = 0.0
    t_reward = 0.0
    t_find_upcoming = 0.0
    t_get_next_time = 0.0
    t_observations = 0.0
    t_random_actions = 0.0

    step_count = 0
    done = False

    # We need to patch step() to instrument it. Instead, we'll time the
    # outer loop and key sub-calls by using the public interface.
    # For a more detailed breakdown, we'll monkey-patch the environment.

    # Save originals
    orig_find_upcoming = env._find_upcoming_case
    orig_filter_completed = env._filter_completed_cases
    orig_get_next_time = env._get_next_time
    orig_get_observations = env._get_observations

    from environment.reward import get_reward as orig_get_reward

    # Timing wrappers
    find_upcoming_time = [0.0]
    filter_completed_time = [0.0]
    get_next_time_time = [0.0]
    get_obs_time = [0.0]
    get_reward_time = [0.0]
    work_case_time = [0.0]

    def timed_find_upcoming():
        t = time.perf_counter()
        result = orig_find_upcoming()
        find_upcoming_time[0] += time.perf_counter() - t
        return result

    def timed_filter_completed():
        t = time.perf_counter()
        result = orig_filter_completed()
        filter_completed_time[0] += time.perf_counter() - t
        return result

    def timed_get_next_time():
        t = time.perf_counter()
        result = orig_get_next_time()
        get_next_time_time[0] += time.perf_counter() - t
        return result

    def timed_get_observations(agent):
        t = time.perf_counter()
        result = orig_get_observations(agent)
        get_obs_time[0] += time.perf_counter() - t
        return result

    # Patch
    env._find_upcoming_case = timed_find_upcoming
    env._filter_completed_cases = timed_filter_completed
    env._get_next_time = timed_get_next_time
    env._get_observations = timed_get_observations

    # For reward and work_case, we need to patch at module level
    import environment.simulator as sim_module
    import environment.reward as reward_module

    orig_get_reward_ref = sim_module.get_reward

    def timed_get_reward(env_ref):
        t = time.perf_counter()
        result = orig_get_reward_ref(env_ref)
        get_reward_time[0] += time.perf_counter() - t
        return result

    sim_module.get_reward = timed_get_reward

    # Patch agent.work_case
    orig_work_cases = {}
    for agent in env.agents:
        orig_work_cases[agent.id] = agent.work_case

        def make_timed_work_case(orig_fn):
            def timed_work_case(current_time):
                t = time.perf_counter()
                result = orig_fn(current_time)
                work_case_time[0] += time.perf_counter() - t
                return result
            return timed_work_case

        agent.work_case = make_timed_work_case(agent.work_case)

    # Run
    t_total_start = time.perf_counter()

    while not done and step_count < max_steps:
        t0 = time.perf_counter()
        obs, rewards, terminations, truncations, _ = env.step(actions)
        t_step_total += time.perf_counter() - t0

        done = any(terminations.values()) or any(truncations.values())
        if not done:
            t_ra = time.perf_counter()
            actions = random_actions(env)
            t_random_actions += time.perf_counter() - t_ra
        step_count += 1

    t_total = time.perf_counter() - t_total_start

    # Restore originals
    env._find_upcoming_case = orig_find_upcoming
    env._filter_completed_cases = orig_filter_completed
    env._get_next_time = orig_get_next_time
    env._get_observations = orig_get_observations
    sim_module.get_reward = orig_get_reward_ref
    for agent in env.agents:
        agent.work_case = orig_work_cases[agent.id]

    # Report
    print(f"\n{'=' * 70}")
    print(f"  MANUAL TIMING BREAKDOWN — {step_count} steps in {t_total:.2f}s")
    print(f"{'=' * 70}")
    print(f"  {'Section':<30} {'Time (s)':>10} {'% of step':>10} {'µs/step':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    sections = [
        ("env.step() total", t_step_total),
        ("  work_case()", work_case_time[0]),
        ("  _filter_completed_cases()", filter_completed_time[0]),
        ("  get_reward()", get_reward_time[0]),
        ("  _find_upcoming_case()", find_upcoming_time[0]),
        ("  _get_next_time()", get_next_time_time[0]),
        ("  _get_observations()", get_obs_time[0]),
        ("random_actions()", t_random_actions),
    ]

    for name, t in sections:
        pct = (t / t_step_total * 100) if t_step_total > 0 else 0
        us = (t / step_count * 1_000_000) if step_count > 0 else 0
        print(f"  {name:<30} {t:>10.3f} {pct:>9.1f}% {us:>9.0f}")

    # Overhead = step_total minus all instrumented sub-sections
    accounted = (
        work_case_time[0]
        + filter_completed_time[0]
        + get_reward_time[0]
        + find_upcoming_time[0]
        + get_next_time_time[0]
        + get_obs_time[0]
    )
    overhead = t_step_total - accounted
    pct = (overhead / t_step_total * 100) if t_step_total > 0 else 0
    us = (overhead / step_count * 1_000_000) if step_count > 0 else 0
    print(f"  {'  action handling + overhead':<30} {overhead:>10.3f} {pct:>9.1f}% {us:>9.0f}")

    print(f"\n  Total wall time: {t_total:.2f}s")
    print(f"  Throughput: {step_count / t_total:.0f} steps/s")


def main():
    parser = argparse.ArgumentParser(description="Profile environment step() function")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=5000,
        help="Number of steps to profile (default: 5000)",
    )
    parser.add_argument(
        "--skip-cprofile",
        action="store_true",
        help="Skip cProfile, only run manual timing",
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # Load config and create environment
    print("Loading config and creating environment...")
    config = ExperimentConfig(args.config)

    env = create_env(config)

    # Run manual timing breakdown
    run_manual_timing(env, args.steps)

    # Run cProfile
    if not args.skip_cprofile:
        env2 = create_env(config)
        run_profiled_episode(env2, args.steps)


if __name__ == "__main__":
    main()
