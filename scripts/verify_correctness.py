#!/usr/bin/env python3
"""
Verify that optimized environment produces identical outputs to the original.

Temporarily swaps in the .bak files (original code), runs N steps, saves all
observations/rewards/terminations, then runs the same with optimized code and
compares bit-for-bit.

Usage:
    python verify_correctness.py --config src/configs/cvs/donkey_a/cvs_mappo_collab.yaml
    python verify_correctness.py --config src/configs/cvs/donkey_a/cvs_mappo_collab.yaml --steps 10000
"""
import argparse
import shutil
import subprocess
import sys
import json
import os
import random
import numpy as np
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

ENV_DIR = Path(__file__).resolve().parent / "src" / "environment"

# Files that were optimized (originals saved as .bak)
OPT_FILES = ["simulator.py", "entities.py", "config.py", "reward.py"]


def swap_to_original():
    """Replace optimized files with originals (.bak)."""
    for f in OPT_FILES:
        bak = ENV_DIR / f"{f}.bak"
        opt = ENV_DIR / f
        if bak.exists():
            # Save optimized version
            shutil.copy2(opt, ENV_DIR / f"{f}.opt")
            # Restore original
            shutil.copy2(bak, opt)


def swap_to_optimized():
    """Replace original files with optimized versions (.opt)."""
    for f in OPT_FILES:
        opt_saved = ENV_DIR / f"{f}.opt"
        target = ENV_DIR / f
        if opt_saved.exists():
            shutil.copy2(opt_saved, target)
            opt_saved.unlink()


def run_episode(config_path: str, max_steps: int, label: str):
    """Run one episode and collect all outputs."""
    # Force reimport of modules
    mods_to_remove = [k for k in sys.modules if k.startswith("environment") or k.startswith("configs") or k.startswith("preprocessing")]
    for m in mods_to_remove:
        del sys.modules[m]

    from configs.config_loader import ExperimentConfig
    from preprocessing.data_loader import load_and_preprocess_data
    from environment.simulator import AgentOptimizerEnvironment

    # Set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    config = ExperimentConfig(config_path)
    collaboration_config = config.get("agent.collaboration", None)
    train_data, _ = load_and_preprocess_data(
        data_path=config.get("data.input_file"),
        train_split=config.get("data.train_split", 0.8),
        min_case_length=config.get("data.min_case_length", 3),
        collaboration_config=collaboration_config,
    )

    env = AgentOptimizerEnvironment(
        data=train_data,
        simulation_parameters={
            "start_timestamp": train_data["assign_timestamp"].min()
            if "assign_timestamp" in train_data.columns
            else train_data["start_timestamp"].min()
        },
        experiment_dir=None,
        enable_logging=False,
        verbose=False,
        max_steps=config.get("environment.max_steps", 200_000),
        max_episodes=config.get("environment.max_episodes", 1000),
        reward_config=config.get("reward", None),
        work_schedule_enabled=config.get("environment.work_schedule_enabled", False),
    )

    # Reset with seed
    random.seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset()

    all_obs = []
    all_rewards = []
    all_terms = []

    # Collect initial observations
    all_obs.append({k: {kk: float(vv) if np.isscalar(vv) else vv.tolist()
                        for kk, vv in v.items()} for k, v in obs.items()})

    step_count = 0
    done = False
    random.seed(seed + 1)  # Separate seed for actions
    np.random.seed(seed + 1)

    while not done and step_count < max_steps:
        actions = {agent.id: random.randint(0, 1) for agent in env.agents}
        obs, rewards, terminations, truncations, _ = env.step(actions)
        done = any(terminations.values()) or any(truncations.values())

        if not done:
            all_obs.append({k: {kk: float(vv) if np.isscalar(vv) else vv.tolist()
                                for kk, vv in v.items()} for k, v in obs.items()})
        all_rewards.append({k: float(v) for k, v in rewards.items()})
        all_terms.append({k: bool(v) for k, v in terminations.items()})
        step_count += 1

    print(f"  [{label}] Completed {step_count} steps")
    return all_obs, all_rewards, all_terms, step_count


def compare_results(orig, opt, name):
    """Compare two lists of dicts for equality."""
    if len(orig) != len(opt):
        print(f"  MISMATCH in {name}: different lengths ({len(orig)} vs {len(opt)})")
        return False

    mismatches = 0
    for i, (o, p) in enumerate(zip(orig, opt)):
        if o != p:
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH in {name} at step {i}:")
                # Find which keys differ
                all_keys = set(list(o.keys()) + list(p.keys()))
                for k in sorted(all_keys):
                    if o.get(k) != p.get(k):
                        print(f"    agent {k}: orig={o.get(k)}, opt={p.get(k)}")

    if mismatches > 0:
        print(f"  TOTAL MISMATCHES in {name}: {mismatches}/{len(orig)} steps")
        return False

    print(f"  {name}: IDENTICAL ({len(orig)} entries)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify optimized env correctness")
    parser.add_argument("--config", "-c", required=True, help="YAML config file")
    parser.add_argument("--steps", "-s", type=int, default=5000, help="Steps to compare")
    args = parser.parse_args()

    print("=" * 60)
    print("  CORRECTNESS VERIFICATION")
    print("=" * 60)

    # Step 1: Run with original code
    print("\n1. Running with ORIGINAL code...")
    swap_to_original()
    try:
        orig_obs, orig_rewards, orig_terms, orig_steps = run_episode(
            args.config, args.steps, "original"
        )
    finally:
        swap_to_optimized()

    # Step 2: Run with optimized code
    print("\n2. Running with OPTIMIZED code...")
    opt_obs, opt_rewards, opt_terms, opt_steps = run_episode(
        args.config, args.steps, "optimized"
    )

    # Step 3: Compare
    print(f"\n3. Comparing results ({orig_steps} vs {opt_steps} steps)...")
    if orig_steps != opt_steps:
        print(f"  MISMATCH: Different step counts ({orig_steps} vs {opt_steps})")
        sys.exit(1)

    obs_ok = compare_results(orig_obs, opt_obs, "observations")
    rew_ok = compare_results(orig_rewards, opt_rewards, "rewards")
    term_ok = compare_results(orig_terms, opt_terms, "terminations")

    print(f"\n{'=' * 60}")
    if obs_ok and rew_ok and term_ok:
        print("  ALL CHECKS PASSED — outputs are identical")
    else:
        print("  SOME CHECKS FAILED — outputs differ!")
        sys.exit(1)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
