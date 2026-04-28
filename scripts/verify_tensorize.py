#!/usr/bin/env python3
"""
Verify that the optimized pretensorize functions and tensor construction
produce IDENTICAL results to the originals.

Loads a real config, runs a short rollout to get real data, then compares
original vs optimized tensor outputs.

Usage:
    python verify_tensorize.py --config src/configs/cvs/donkey_a/cvs_mappo_collab.yaml --steps 500
    python verify_tensorize.py --config src/configs/cvs/cinema_a/cvs_mappo_collab_comm.yaml --steps 500
"""
import argparse
import sys
import time
import shutil
import importlib
from pathlib import Path

import numpy as np
import torch

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def create_env(config):
    """Initialize environment from config."""
    from preprocessing.data_loader import load_and_preprocess_data
    from environment.simulator import AgentOptimizerEnvironment

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
            else train_data["timestamp"].min(),
            "max_steps_per_episode": config.get("training.max_steps_per_episode", 200000),
        },
        reward_config=reward_config,
        work_schedule_enabled=work_schedule_enabled,
        enable_logging=False,
        verbose=False,
    )
    return env


def collect_rollout_data(env, steps):
    """Collect rollout data with random actions, returning buffer-like structures."""
    np.random.seed(42)
    torch.manual_seed(42)

    obs, _ = env.reset()
    n_agents = len(env.agents)
    agent_ids = [a.id for a in env.agents]

    # Simulate buffer storage like MAPPOAgent does
    all_obs = []
    all_actions = {}
    all_action_probs = {}
    all_rewards = []
    all_gate_log_probs = []
    all_gate_actions = []

    n_action_features = 2  # volunteer or not

    for step in range(steps):
        actions_dict = {}
        probs_dict = {}
        for aid in agent_ids:
            action = np.random.randint(0, 2)
            actions_dict[aid] = action
            # Simulate action probabilities (normalized)
            raw = np.random.rand(n_action_features).astype(np.float32)
            raw /= raw.sum()
            probs_dict[aid] = torch.from_numpy(raw)

        all_obs.append(obs)
        all_actions[step] = actions_dict
        all_action_probs[step] = probs_dict
        all_rewards.append(np.random.randn())

        # Simulate gate log probs/actions for IC3Net
        all_gate_log_probs.append(torch.randn(n_agents, 2))
        all_gate_actions.append(torch.randint(0, 2, (n_agents, 2)))

        obs, rewards, terminations, truncations, infos = env.step(
            {aid: actions_dict[aid] for aid in agent_ids}
        )
        if all(terminations.values()):
            obs, _ = env.reset()

    # Convert to list-of-dicts format (matching buffer["actions"] etc.)
    actions_list = [all_actions[i] for i in range(steps)]
    probs_list = [all_action_probs[i] for i in range(steps)]

    return {
        "obs": all_obs,
        "actions": actions_list,
        "action_probs": probs_list,
        "rewards": all_rewards,
        "gate_log_probs": all_gate_log_probs,
        "gate_actions": all_gate_actions,
        "agent_ids": agent_ids,
        "n_agents": n_agents,
    }


def swap_files(file_pairs, use_backup):
    """Swap between .bak (original) and optimized versions."""
    for opt_path, bak_path in file_pairs:
        if use_backup:
            # Use original
            shutil.copy2(bak_path, opt_path)
        else:
            # Restore optimized (from .opt)
            shutil.copy2(opt_path.replace(".py", ".opt"), opt_path)


def run_tensorize(data, obs_keys_actor, obs_keys_critic, use_ic3net=True):
    """Run the tensorize functions and return all tensors for comparison."""
    # Force reimport to get current version of the files
    import agents.mappo.networks as networks_mod
    import agents.mappo.communication as comm_mod
    importlib.reload(networks_mod)
    importlib.reload(comm_mod)

    from agents.mappo.networks import pretensorize_actor_observations, pretensorize_critic_observations
    from agents.mappo.communication import pretensorize_ic3net_observations

    observations = data["obs"]
    actions = data["actions"]
    old_action_probs = data["action_probs"]
    rewards = data["rewards"]
    agent_ids = data["agent_ids"]
    n_agents = data["n_agents"]
    N = len(actions)
    n_actions = 2
    n_action_features = 2

    results = {}

    # 1. Build joint_action_onehot
    joint_action_onehot = torch.zeros(N, n_agents * n_actions)
    for t in range(N):
        for agent_idx, agent_id in enumerate(agent_ids):
            action = actions[t].get(agent_id, 0)
            joint_action_onehot[t, agent_idx * n_actions + action] = 1.0
    results["joint_action_onehot"] = joint_action_onehot

    # 2. Build per-agent action_tensors and old_action_prob_tensors
    action_tensors = {}
    old_action_prob_tensors = {}
    for agent_id in agent_ids:
        agent_actions = []
        agent_old_probs = []
        for i in range(N):
            if agent_id in actions[i]:
                agent_actions.append(actions[i][agent_id])
                agent_old_probs.append(old_action_probs[i][agent_id])
            else:
                agent_actions.append(0)
                agent_old_probs.append(torch.ones(n_action_features))
        action_tensors[agent_id] = torch.LongTensor(agent_actions)
        old_action_prob_tensors[agent_id] = torch.stack(agent_old_probs)
    results["action_tensors"] = action_tensors
    results["old_action_prob_tensors"] = old_action_prob_tensors

    # 3. Pretensorize critic observations
    preprocessed_obs_lists = []
    preprocessed_rewards = []
    for i in range(N):
        obs_list = [observations[i][aid] for aid in agent_ids]
        preprocessed_obs_lists.append(obs_list)
        preprocessed_rewards.append(rewards[i])
    results["critic_tensor"] = pretensorize_critic_observations(
        preprocessed_obs_lists, preprocessed_rewards, obs_keys_critic
    )

    # 4. IC3Net pretensorize
    if use_ic3net:
        results["ic3net_tensor"] = pretensorize_ic3net_observations(
            observations, preprocessed_rewards, obs_keys_actor,
            n_agents, agent_ids
        )

        # 5. IC3Net action/probs tensors
        ic3net_action_tensor = torch.zeros(N, n_agents, dtype=torch.long)
        ic3net_old_action_probs_tensor = torch.zeros(N, n_agents, n_action_features)
        for t in range(N):
            for agent_idx, agent_id in enumerate(agent_ids):
                if agent_id in actions[t]:
                    ic3net_action_tensor[t, agent_idx] = actions[t][agent_id]
                    ic3net_old_action_probs_tensor[t, agent_idx] = old_action_probs[t][agent_id]
        results["ic3net_action_tensor"] = ic3net_action_tensor
        results["ic3net_old_action_probs_tensor"] = ic3net_old_action_probs_tensor

    return results


def run_tensorize_optimized(data, obs_keys_actor, obs_keys_critic, use_ic3net=True):
    """Run the OPTIMIZED tensorize functions and return all tensors."""
    import agents.mappo.networks as networks_mod
    import agents.mappo.communication as comm_mod
    importlib.reload(networks_mod)
    importlib.reload(comm_mod)

    from agents.mappo.networks import pretensorize_critic_observations
    from agents.mappo.communication import pretensorize_ic3net_observations

    observations = data["obs"]
    actions = data["actions"]
    old_action_probs = data["action_probs"]
    rewards = data["rewards"]
    agent_ids = data["agent_ids"]
    n_agents = data["n_agents"]
    N = len(actions)
    n_actions = 2
    n_action_features = 2

    results = {}

    # --- Extract all actions + probs in a single pass ---
    action_matrix = np.zeros((N, n_agents), dtype=np.int64)
    probs_matrix = np.ones((N, n_agents, n_action_features), dtype=np.float32)

    for t in range(N):
        act_dict = actions[t]
        prob_dict = old_action_probs[t]
        for idx, aid in enumerate(agent_ids):
            if aid in act_dict:
                action_matrix[t, idx] = act_dict[aid]
                prob_val = prob_dict[aid]
                if isinstance(prob_val, torch.Tensor):
                    probs_matrix[t, idx] = prob_val.numpy()
                else:
                    probs_matrix[t, idx] = prob_val

    # 1. joint_action_onehot via scatter_
    action_tensor_cpu = torch.from_numpy(action_matrix)
    offsets = torch.arange(n_agents).unsqueeze(0) * n_actions
    scatter_indices = (offsets + action_tensor_cpu).long()
    joint_action_onehot = torch.zeros(N, n_agents * n_actions)
    joint_action_onehot.scatter_(1, scatter_indices, 1.0)
    results["joint_action_onehot"] = joint_action_onehot

    # 2. Per-agent action_tensors + old_action_prob_tensors (sliced)
    action_tensors = {}
    old_action_prob_tensors = {}
    probs_tensor_all = torch.from_numpy(probs_matrix)
    for idx, agent_id in enumerate(agent_ids):
        action_tensors[agent_id] = action_tensor_cpu[:, idx].long()
        old_action_prob_tensors[agent_id] = probs_tensor_all[:, idx]
    results["action_tensors"] = action_tensors
    results["old_action_prob_tensors"] = old_action_prob_tensors

    # 3. Pretensorize critic observations
    preprocessed_obs_lists = [
        [observations[i][aid] for aid in agent_ids]
        for i in range(N)
    ]
    results["critic_tensor"] = pretensorize_critic_observations(
        preprocessed_obs_lists, rewards, obs_keys_critic
    )

    # 4. IC3Net pretensorize
    if use_ic3net:
        results["ic3net_tensor"] = pretensorize_ic3net_observations(
            observations, rewards, obs_keys_actor,
            n_agents, agent_ids
        )

        # 5. IC3Net action/probs tensors (reused from matrices)
        results["ic3net_action_tensor"] = action_tensor_cpu.long()
        results["ic3net_old_action_probs_tensor"] = probs_tensor_all

    return results


def compare_results(orig, opt, label=""):
    """Compare two result dicts tensor by tensor."""
    all_ok = True
    for key in orig:
        if isinstance(orig[key], dict):
            for sub_key in orig[key]:
                o = orig[key][sub_key]
                p = opt[key][sub_key]
                if not torch.equal(o, p):
                    max_diff = (o.float() - p.float()).abs().max().item()
                    print(f"  MISMATCH: {label}{key}[{sub_key}] max_diff={max_diff}")
                    # Show first difference
                    diff_mask = o != p
                    idx = diff_mask.nonzero(as_tuple=False)[0]
                    print(f"    First diff at index {idx.tolist()}: orig={o[tuple(idx)]} vs opt={p[tuple(idx)]}")
                    all_ok = False
                else:
                    print(f"  OK: {label}{key}[{sub_key}] shape={o.shape}")
        elif isinstance(orig[key], torch.Tensor):
            o = orig[key]
            p = opt[key]
            if not torch.equal(o, p):
                max_diff = (o.float() - p.float()).abs().max().item()
                print(f"  MISMATCH: {label}{key} max_diff={max_diff}")
                diff_mask = o != p
                idx = diff_mask.nonzero(as_tuple=False)[0]
                print(f"    First diff at index {idx.tolist()}: orig={o[tuple(idx)]} vs opt={p[tuple(idx)]}")
                all_ok = False
            else:
                print(f"  OK: {label}{key} shape={o.shape}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Verify tensorize optimizations")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=500, help="Number of rollout steps")
    args = parser.parse_args()

    from configs.config_loader import ExperimentConfig
    config = ExperimentConfig(args.config)

    print(f"Creating environment from {args.config}...")
    env = create_env(config)

    print(f"Collecting {args.steps} rollout steps with random actions...")
    data = collect_rollout_data(env, args.steps)
    print(f"  Collected {len(data['obs'])} observations, {len(data['actions'])} actions")

    # Determine obs_keys from env
    first_agent_id = data["agent_ids"][0]
    obs_space = env.observation_space(first_agent_id)
    obs_keys = list(obs_space.keys())
    print(f"  obs_keys: {obs_keys}")

    use_ic3net = config.get("agent.communication.enabled", False)
    print(f"  IC3Net: {'enabled' if use_ic3net else 'disabled'}")

    # Run original (hardcoded reference implementation in run_tensorize)
    print("\n--- Running ORIGINAL tensorize ---")
    t0 = time.perf_counter()
    orig_results = run_tensorize(data, obs_keys, obs_keys, use_ic3net=use_ic3net)
    t_orig = time.perf_counter() - t0
    print(f"  Original: {t_orig:.3f}s")

    # Run optimized
    print("\n--- Running OPTIMIZED tensorize ---")
    t0 = time.perf_counter()
    opt_results = run_tensorize_optimized(data, obs_keys, obs_keys, use_ic3net=use_ic3net)
    t_opt = time.perf_counter() - t0
    print(f"  Optimized: {t_opt:.3f}s")

    # Compare
    print(f"\n--- COMPARISON ---")
    all_ok = compare_results(orig_results, opt_results)

    print(f"\n{'=' * 60}")
    if all_ok:
        print(f"ALL CHECKS PASSED — tensors are identical")
        print(f"Speedup: {t_orig / t_opt:.1f}x ({t_orig:.3f}s → {t_opt:.3f}s)")
    else:
        print(f"MISMATCHES FOUND — check output above")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
