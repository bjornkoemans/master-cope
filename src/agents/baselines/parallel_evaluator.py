"""
Parallel baseline evaluation - uses all CPU cores for faster evaluation.
"""
import numpy as np
import multiprocessing as mp
from functools import partial
from typing import Dict, Any
import pandas as pd

from environment.display import print_colored


def run_single_episode(args):
    """
    Run a single episode (designed to be run in parallel).

    Args:
        args: Tuple of (episode_num, env_data, agent_type, seed)

    Returns:
        Dict with episode results
    """
    episode_num, env_config, agent_type, seed = args

    # Import here to avoid pickling issues
    from environment.simulator import AgentOptimizerEnvironment
    from agents.baselines.baselines import RandomAgent, BestMedianAgent, ShortestQueueAgent, GroundTruthAssignmentAgent

    # Create environment for this worker
    env = AgentOptimizerEnvironment(
        data=env_config['data'],
        simulation_parameters=env_config['simulation_parameters'],
        experiment_dir=None,
        enable_logging=False,  # Disable logging for parallel runs
        max_steps=env_config.get('max_steps', 100_000),
        max_episodes=env_config.get('max_episodes', 1000),
        work_schedule_enabled=env_config.get('work_schedule_enabled', False),
        work_start_hour=env_config.get('work_start_hour', 8),
        work_end_hour=env_config.get('work_end_hour', 20),
        parallel_task_groups=env_config.get('parallel_task_groups'),
    )

    # Create agent
    if agent_type == 'random':
        agent = RandomAgent(env, seed=seed + episode_num)
    elif agent_type == 'best_median':
        agent = BestMedianAgent(env, seed=seed + episode_num)
    elif agent_type == 'shortest_queue':
        agent = ShortestQueueAgent(env, seed=seed + episode_num)
    elif agent_type == 'ground_truth':
        agent = GroundTruthAssignmentAgent(env)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        observations = reset_result[0]
    else:
        observations = reset_result

    if hasattr(agent, "reset_history"):
        agent.reset_history()

    episode_reward = 0
    episode_length = 0
    episode_done = False

    # Run episode
    while not episode_done:
        # Get actions
        actions, _ = agent.select_actions(observations, deterministic=True)

        # Step environment
        step_result = env.step(actions)

        if len(step_result) == 5:
            next_observations, rewards, terminations, truncations, infos = step_result
            dones = {
                agent_id: terminations.get(agent_id, False) or truncations.get(agent_id, False)
                for agent_id in rewards.keys()
            }
        elif len(step_result) == 4:
            next_observations, rewards, dones, infos = step_result
        else:
            raise ValueError(f"Unexpected step return format")

        # Accumulate rewards
        episode_reward += sum(rewards.values())
        episode_length += 1

        # Check if done
        episode_done = any(dones.values()) or episode_length >= 1000

        observations = next_observations

    return {
        'episode': episode_num,
        'reward': episode_reward,
        'length': episode_length
    }


class ParallelBaselineEvaluator:
    """Evaluates baseline agents using all CPU cores."""

    def __init__(self, env, n_workers=None):
        """
        Initialize parallel evaluator.

        Args:
            env: Environment instance (used to get config)
            n_workers: Number of parallel workers (default: number of CPU cores)
        """
        self.env = env
        self.n_workers = n_workers or mp.cpu_count()

        # Store environment config for workers
        self.env_config = {
            'data': env.data,
            'simulation_parameters': {
                'start_timestamp': env.data['assign_timestamp'].min()
                if 'assign_timestamp' in env.data.columns
                else env.data['start_timestamp'].min()
            },
            'work_schedule_enabled': env.work_schedule_enabled,
            'work_start_hour': env.work_start_hour,
            'work_end_hour': env.work_end_hour,
            'parallel_task_groups': env._parallel_task_groups_config,
        }

        print_colored(
            f"Initialized ParallelBaselineEvaluator with {self.n_workers} workers",
            "green"
        )

    def evaluate_agent(
        self,
        agent_type: str,
        agent_name: str,
        num_episodes: int = 100,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Evaluate agent in parallel across all CPU cores.

        Args:
            agent_type: Type of baseline ('random', 'best_median', 'shortest_queue', 'ground_truth')
            agent_name: Name for logging
            num_episodes: Number of episodes to evaluate
            seed: Random seed

        Returns:
            Dict with evaluation metrics
        """
        print_colored(
            f"\nEvaluating {agent_name} for {num_episodes} episodes "
            f"(using {self.n_workers} CPU cores in parallel)...",
            "yellow"
        )

        # Prepare arguments for parallel execution
        args_list = [
            (i, self.env_config, agent_type, seed)
            for i in range(num_episodes)
        ]

        # Run episodes in parallel
        with mp.Pool(processes=self.n_workers) as pool:
            results = pool.map(run_single_episode, args_list)

        # Aggregate results
        episode_rewards = [r['reward'] for r in results]
        episode_lengths = [r['length'] for r in results]

        # Compute statistics
        evaluation_results = {
            'agent_name': agent_name,
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'total_steps': sum(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
        }

        # Print summary
        print_colored(f"\n{agent_name} Evaluation Results:", "green")
        print_colored(
            f"  Mean Reward: {evaluation_results['mean_reward']:.2f} ± "
            f"{evaluation_results['std_reward']:.2f}",
            "white",
        )
        print_colored(
            f"  Median Reward: {evaluation_results['median_reward']:.2f}",
            "white"
        )
        print_colored(
            f"  Reward Range: [{evaluation_results['min_reward']:.2f}, "
            f"{evaluation_results['max_reward']:.2f}]",
            "white",
        )
        print_colored(
            f"  Mean Episode Length: {evaluation_results['mean_episode_length']:.1f}",
            "white"
        )
        print_colored(
            f"  Total Steps: {evaluation_results['total_steps']}",
            "white"
        )

        return evaluation_results

    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to JSON."""
        import json

        # Convert numpy types to Python types
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print_colored(f"Results saved to {filepath}", "green")


def run_baseline_parallel(
    env,
    baseline_type: str,
    experiment_dir: str,
    n_episodes: int = 100,
    n_workers: int = None
):
    """
    Run baseline agent evaluation using parallel processing.

    Args:
        env: Environment instance
        baseline_type: Type of baseline ('random', 'best_median', 'ground_truth')
        experiment_dir: Directory to save results
        n_episodes: Number of episodes to evaluate
        n_workers: Number of parallel workers (default: CPU count)
    """
    from pathlib import Path

    # Map baseline types to names
    agent_names = {
        'random': 'Random Baseline',
        'best_median': 'Best Median Baseline',
        'shortest_queue': 'Shortest Queue Baseline',
        'ground_truth': 'Ground Truth Baseline'
    }

    agent_name = agent_names.get(baseline_type, baseline_type)

    print_colored(f"\nRunning {agent_name} (PARALLEL)", "yellow")
    print_colored("=" * 60, "yellow")

    # Create parallel evaluator
    evaluator = ParallelBaselineEvaluator(env, n_workers=n_workers)

    # Run evaluation
    results = evaluator.evaluate_agent(
        agent_type=baseline_type,
        agent_name=agent_name,
        num_episodes=n_episodes,
        seed=42
    )

    # Save results
    output_dir = Path(experiment_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{baseline_type}_results.json"
    evaluator.save_results(results, str(results_file))

    print_colored(f"\nParallel baseline evaluation completed.", "green")
    print_colored(f"Results saved to: {results_file}", "green")
    print_colored(
        f"Speed: Used {evaluator.n_workers} CPU cores in parallel",
        "cyan"
    )

    return results
