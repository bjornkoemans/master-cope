"""
Baseline agents for comparison with MAPPO.
These agents don't require training and are used for evaluation only.
"""

import numpy as np
import torch
from typing import Any, Dict, List
from collections import defaultdict

from environment.display import print_colored


class RandomAgent:
    """
    Baseline agent that selects actions randomly.
    """

    def __init__(self, env, seed=None):
        self.env = env
        self.agents = env.agents
        self.rng = np.random.RandomState(seed)

    def select_actions(self, observations, deterministic=False):
        """Select random actions for all agents."""
        actions = {}
        action_probs = {}

        for agent_id, obs in observations.items():
            if agent_id in [agent.id for agent in self.agents]:
                # Get action space for this agent
                action_space = self.env.action_space(agent_id)
                # Sample random action
                action = self.rng.randint(0, action_space.n)
                actions[agent_id] = action

                # Create uniform probability distribution
                n_actions = action_space.n
                probs = torch.ones(n_actions) / n_actions
                action_probs[agent_id] = probs

        return actions, action_probs

    def compute_values(self, observations):
        """Return zero value estimate (not used for evaluation)."""
        return 0.0

    def reset_history(self):
        """No history to reset for random agent."""
        pass


class BestMedianAgent:
    """
    Baseline agent: at each decision step, only the capable agent with the
    lowest historical median duration for the offered task volunteers.
    Ties are broken randomly.  This deterministic heuristic exploits
    historical performance data but ignores current workload.
    """

    def __init__(self, env, performance_data=None, seed=None):
        self.env = env
        self.agents = env.agents
        self.rng = np.random.RandomState(seed)
        print_colored(
            "BestMedianAgent initialized - fastest capable agent volunteers", "blue"
        )

    def _get_task_median(self, agent, task_id):
        """Get the agent's historical median duration for a task.
        Uses agent.stats_dict which maps activity_name -> {median, ...}.
        Returns None if the agent cannot perform the task."""
        if not agent.can_perform_task(task_id):
            return None
        # stats_dict is keyed by activity name, but we have task_id.
        # Use the environment's reverse mapping.
        task_name = None
        for name, tid in self.env.task_dict.items():
            if tid == task_id:
                task_name = name
                break
        if task_name is None:
            return None
        stats = getattr(agent, "stats_dict", None)
        if stats and task_name in stats and stats[task_name] is not None:
            return stats[task_name].get("median")
        return None

    def select_actions(self, observations, deterministic=True):
        """
        Only the capable agent with the lowest historical median duration
        for the current task volunteers.  All others decline.
        """
        actions = {}
        action_probs = {}

        # Determine the upcoming task
        upcoming = getattr(self.env, "upcoming_case", None)
        task_id = None
        if upcoming and upcoming.current_task:
            task_id = upcoming.current_task.id

        # Find the capable agent with the lowest median for this task
        best_agent_id = None
        if task_id is not None:
            candidates = []
            for agent in self.agents:
                median = self._get_task_median(agent, task_id)
                if median is not None:
                    candidates.append((agent.id, median))
            if candidates:
                min_median = min(m for _, m in candidates)
                tied = [aid for aid, m in candidates if m == min_median]
                best_agent_id = self.rng.choice(tied)

        for agent_id, obs in observations.items():
            if agent_id in [agent.id for agent in self.agents]:
                action_space = self.env.action_space(agent_id)
                n_actions = action_space.n

                if agent_id == best_agent_id:
                    action = 1 if n_actions > 1 else 0
                    probs = torch.zeros(n_actions)
                    probs[action] = 0.95
                    if n_actions > 1:
                        probs[1 - action] = 0.05
                else:
                    action = 0
                    probs = torch.zeros(n_actions)
                    probs[0] = 0.95
                    if n_actions > 1:
                        probs[1] = 0.05

                actions[agent_id] = action
                action_probs[agent_id] = probs

        return actions, action_probs

    def compute_values(self, observations):
        """Return zero value estimate (not used for evaluation)."""
        return 0.0

    def reset_history(self):
        """No history to reset for this agent."""
        pass


class ShortestQueueAgent:
    """
    Baseline agent: at each decision step, only the capable agent with the
    fewest pending tasks in its queue volunteers.  Ties are broken randomly.
    This workload-aware heuristic provides a strong baseline for load balancing.
    """

    def __init__(self, env, seed=None):
        self.env = env
        self.agents = env.agents
        self.rng = np.random.RandomState(seed)
        print_colored(
            "ShortestQueueAgent initialized - shortest-queue capable agent volunteers",
            "blue",
        )

    def select_actions(self, observations, deterministic=True):
        """
        Only the capable agent with the shortest queue volunteers.
        """
        actions = {}
        action_probs = {}

        # Determine the upcoming task
        upcoming = getattr(self.env, "upcoming_case", None)
        task_id = None
        if upcoming and upcoming.current_task:
            task_id = upcoming.current_task.id

        # Compute queue length only for capable agents
        queue_lengths = {}
        for agent in self.agents:
            if task_id is not None and not agent.can_perform_task(task_id):
                continue  # skip agents that cannot do this task
            q_len = len(agent.case_queue)
            if agent.current_case is not None:
                q_len += 1
            queue_lengths[agent.id] = q_len

        shortest_agent_id = None
        if queue_lengths:
            min_q = min(queue_lengths.values())
            candidates = [aid for aid, ql in queue_lengths.items() if ql == min_q]
            shortest_agent_id = self.rng.choice(candidates)

        for agent_id, obs in observations.items():
            if agent_id in [agent.id for agent in self.agents]:
                action_space = self.env.action_space(agent_id)
                n_actions = action_space.n

                if agent_id == shortest_agent_id:
                    action = 1 if n_actions > 1 else 0
                    probs = torch.zeros(n_actions)
                    probs[action] = 0.95
                    if n_actions > 1:
                        probs[1 - action] = 0.05
                else:
                    action = 0
                    probs = torch.zeros(n_actions)
                    probs[0] = 0.95
                    if n_actions > 1:
                        probs[1] = 0.05

                actions[agent_id] = action
                action_probs[agent_id] = probs

        return actions, action_probs

    def compute_values(self, observations):
        """Return zero value estimate (not used for evaluation)."""
        return 0.0

    def reset_history(self):
        """No history to reset for this agent."""
        pass


class GroundTruthAssignmentAgent:
    """
    Baseline agent that, at each timestep, assigns the agent that was assigned to the task in the actual data.
    The environment must provide the assigned agent id in the observation or info dict for each agent.
    """

    def __init__(self, env, assigned_agent_key="assigned_agent_id"):
        self.env = env
        self.agents = env.agents
        self.assigned_agent_key = assigned_agent_key
        print_colored(
            "GroundTruthAssignmentAgent initialized - uses ground truth assignments",
            "blue",
        )

    def select_actions(self, observations, deterministic=True):
        """
        Select actions so that only the ground truth assigned agent volunteers.
        Uses the ground_truth_resource stored on the upcoming task at case creation.
        """
        actions = {}
        action_probs = {}

        # Determine the ground truth agent for the current upcoming task
        assigned_agent_id = None
        upcoming = getattr(self.env, "upcoming_case", None)
        if upcoming and upcoming.current_task:
            gt_resource = upcoming.current_task.ground_truth_resource
            if gt_resource:
                try:
                    assigned_agent_id = self.env.resource_name_to_id(gt_resource)
                except ValueError:
                    assigned_agent_id = None

        for agent_id, obs in observations.items():
            if agent_id in [agent.id for agent in self.agents]:
                action_space = self.env.action_space(agent_id)
                n_actions = action_space.n

                if agent_id == assigned_agent_id:
                    action = 1 if n_actions > 1 else 0
                    probs = torch.zeros(n_actions)
                    probs[action] = 1.0
                else:
                    action = 0
                    probs = torch.zeros(n_actions)
                    probs[0] = 1.0

                actions[agent_id] = action
                action_probs[agent_id] = probs

        return actions, action_probs

    def compute_values(self, observations):
        """Return zero value estimate (not used for evaluation)."""
        return 0.0

    def reset_history(self):
        """No history to reset for this agent."""
        pass


class BaselineEvaluator:
    """
    Utility class to evaluate baseline agents alongside trained agents.
    """

    def __init__(self, env, phase=None):
        self.env = env
        self._phase = phase
        self.results = {}

    def evaluate_agent(
        self,
        agent,
        agent_name: str,
        num_episodes: int = 100,
        deterministic: bool = True,
    ):
        """
        Evaluate an agent for a specified number of episodes.

        Args:
            agent: The agent to evaluate (can be MAPPO, Random, or BestMedian)
            agent_name: Name for logging/results
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic action selection

        Returns:
            Dict with evaluation metrics
        """
        print(f"\nEvaluating {agent_name} for {num_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []
        task_success_rates = defaultdict(list)
        total_steps = 0

        for episode in range(num_episodes):
            # Reset environment and agent
            phase = f"eval_ep{episode}" if self._phase is None else f"{self._phase}_ep{episode}"
            reset_result = self.env.reset(options={"phase": phase})
            # Handle both single dict and tuple returns from reset()
            if isinstance(reset_result, tuple):
                observations = reset_result[0]  # observations, info = env.reset()
            else:
                observations = reset_result  # observations = env.reset()

            if hasattr(agent, "reset_history"):
                agent.reset_history()

            episode_reward = 0
            episode_length = 0
            episode_done = False
            infos = None  # Initialize infos variable

            while not episode_done:
                # Get actions from agent
                actions, _ = agent.select_actions(
                    observations, deterministic=deterministic
                )

                # Step environment
                step_result = self.env.step(actions)

                # Handle different step return formats
                if len(step_result) == 5:
                    # observations, rewards, terminations, truncations, infos
                    next_observations, rewards, terminations, truncations, infos = (
                        step_result
                    )
                    # Combine terminations and truncations into dones
                    dones = {
                        agent_id: terminations.get(agent_id, False)
                        or truncations.get(agent_id, False)
                        for agent_id in rewards.keys()
                    }
                elif len(step_result) == 4:
                    # observations, rewards, dones, infos
                    next_observations, rewards, dones, infos = step_result
                else:
                    raise ValueError(
                        f"Unexpected step return format with {len(step_result)} values"
                    )

                # Accumulate rewards
                episode_reward += sum(rewards.values())
                episode_length += 1
                total_steps += 1

                # Check if episode is done (environment signals termination/truncation)
                episode_done = any(dones.values())

                observations = next_observations

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Extract task-specific metrics if available in info
            if "infos" in locals() and infos:
                for agent_id, info in infos.items():
                    if isinstance(info, dict) and "task_success" in info:
                        task_success_rates[agent_id].append(info["task_success"])

            # Progress logging
            if (episode + 1) % max(1, num_episodes // 10) == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"  Episode {episode + 1}/{num_episodes} | Reward (avg last 10): {avg_reward:.2f}")

        # Compute final metrics
        results = {
            "agent_name": agent_name,
            "num_episodes": num_episodes,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "median_reward": np.median(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "total_steps": total_steps,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        # Add task success rates if available
        if task_success_rates:
            for agent_id, successes in task_success_rates.items():
                results[f"task_success_rate_{agent_id}"] = np.mean(successes)

        # Store results
        self.results[agent_name] = results

        # Print summary
        print(f"\n{agent_name} - Results Summary:")
        print("-" * 70)
        print(f"  Mean Reward:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Median Reward:   {results['median_reward']:.2f}")
        print(f"  Reward Range:    [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print(f"  Episode Length:  {results['mean_episode_length']:.1f} (mean)")
        print(f"  Total Steps:     {results['total_steps']}")

        if task_success_rates:
            for agent_id in task_success_rates:
                success_rate = results[f"task_success_rate_{agent_id}"]
                print(f"  Task Success Rate ({agent_id}): {success_rate:.2%}")

        return results

    def compare_agents(self, agent_configs: List[tuple], num_episodes: int = 100):
        """
        Compare multiple agents.

        Args:
            agent_configs: List of (agent, name) tuples
            num_episodes: Number of episodes to evaluate each agent

        Returns:
            Dict with comparison results
        """
        print_colored("\n" + "=" * 60, "yellow")
        print_colored("BASELINE COMPARISON EVALUATION", "yellow")
        print_colored("=" * 60, "yellow")

        # Evaluate each agent
        for agent, name in agent_configs:
            self.evaluate_agent(agent, name, num_episodes)

        # Print comparison summary
        print_colored("\n" + "=" * 60, "yellow")
        print_colored("COMPARISON SUMMARY", "yellow")
        print_colored("=" * 60, "yellow")

        # Sort by mean reward
        sorted_results = sorted(
            self.results.items(), key=lambda x: x[1]["mean_reward"], reverse=True
        )

        print_colored("\nRanking by Mean Reward:", "green")
        for i, (name, results) in enumerate(sorted_results, 1):
            print_colored(
                f"  {i}. {name}: \t{results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
                "white",
            )

        # Statistical comparison
        if len(self.results) >= 2:
            print_colored("\nStatistical Comparison (Mean Reward):", "green")
            baseline_names = list(self.results.keys())
            for i in range(len(baseline_names)):
                for j in range(i + 1, len(baseline_names)):
                    name1, name2 = baseline_names[i], baseline_names[j]
                    rewards1 = self.results[name1]["episode_rewards"]
                    rewards2 = self.results[name2]["episode_rewards"]

                    # Simple t-test approximation
                    diff = np.mean(rewards1) - np.mean(rewards2)
                    pooled_std = np.sqrt((np.var(rewards1) + np.var(rewards2)) / 2)

                    if pooled_std > 0:
                        effect_size = diff / pooled_std
                        print_colored(
                            f"  {name1} vs {name2}: \tΔ={diff:.2f}, Effect Size={effect_size:.2f}",
                            "white",
                        )

        return self.results

    def save_results(self, filepath: str):
        """Save evaluation results to file."""
        import json

        # Convert numpy arrays to lists for JSON serialization
        serializable_results: Dict[str, Dict[str, Any]] = {}
        for name, results in self.results.items():
            serializable_results[name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[name][key] = value.tolist()
                elif isinstance(value, (int, float)) or hasattr(value, "item"):
                    # Handle numpy scalars and regular scalars
                    serializable_results[name][key] = (
                        float(value) if hasattr(value, "item") else value
                    )
                else:
                    serializable_results[name][key] = value

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)


def create_baseline_agents(env, performance_data=None, seed=42):
    """
    Factory function to create baseline agents.

    Args:
        env: The environment
        performance_data: Unused (kept for backward compat)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (random_agent, best_median_agent, shortest_queue_agent, ground_truth_agent)
    """
    random_agent = RandomAgent(env, seed=seed)
    best_median_agent = BestMedianAgent(env, seed=seed)
    shortest_queue_agent = ShortestQueueAgent(env, seed=seed)
    ground_truth_agent = GroundTruthAssignmentAgent(env)

    return random_agent, best_median_agent, shortest_queue_agent, ground_truth_agent


def run_baseline(
    env,
    baseline_type: str,
    experiment_dir: str,
    n_episodes: int = 100,
    test_data=None,
    fitted_distributions=None,
):
    """
    Run a baseline agent evaluation on train data and (optionally) test data.

    Args:
        env: The environment (initialised with train_data)
        baseline_type: Type of baseline ('random', 'best_median', 'shortest_queue', 'ground_truth')
        experiment_dir: Directory to save results
        n_episodes: Number of episodes to evaluate
        test_data: Optional test DataFrame for a separate eval run
        fitted_distributions: Pre-fitted distributions for the eval environment
    """
    from pathlib import Path

    print("\n" + "=" * 70)
    print(f"BASELINE EVALUATION: {baseline_type.upper()}")
    print("=" * 70)

    # Create baseline agent
    seed = 42
    if baseline_type == 'random':
        agent = RandomAgent(env, seed=seed)
        agent_name = "Random Baseline"
    elif baseline_type == 'best_median':
        agent = BestMedianAgent(env, seed=seed)
        agent_name = "Best Median Baseline"
    elif baseline_type == 'shortest_queue':
        agent = ShortestQueueAgent(env, seed=seed)
        agent_name = "Shortest Queue Baseline"
    elif baseline_type == 'ground_truth':
        env.ground_truth_replay = True
        agent = RandomAgent(env, seed=seed)  # Actions are ignored in replay mode
        agent_name = "Ground Truth Baseline"
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    # --- Run on train data ---
    print("\n--- Train data ---")
    evaluator = BaselineEvaluator(env, phase="train")
    results = evaluator.evaluate_agent(
        agent=agent,
        agent_name=agent_name,
        num_episodes=n_episodes,
        deterministic=True,
    )

    # Save train results
    output_dir = Path(experiment_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{baseline_type}_train_results.json"
    evaluator.save_results(str(results_file))
    print(f"Train results saved to: {results_file}")

    # --- Run on test data ---
    if test_data is not None and fitted_distributions is not None:
        from environment.simulator import AgentOptimizerEnvironment

        print("\n--- Test data (eval) ---")
        eval_env = AgentOptimizerEnvironment(
            data=test_data,
            simulation_parameters={
                "start_timestamp": test_data["assign_timestamp"].min()
                if "assign_timestamp" in test_data.columns
                else test_data["start_timestamp"].min()
            },
            experiment_dir=experiment_dir,
            enable_logging=True,
            verbose=True,
            pre_fitted_distributions=fitted_distributions,
            max_steps=env.max_steps,
            max_episodes=env.max_episodes,
            work_schedule_enabled=env.work_schedule_enabled,
            work_start_hour=env.work_start_hour,
            work_end_hour=env.work_end_hour,
            parallel_task_groups=env._parallel_task_groups_config,
        )

        # Re-create agent for eval env so it sees the correct agents/tasks
        if baseline_type == 'random':
            eval_agent = RandomAgent(eval_env, seed=seed)
        elif baseline_type == 'best_median':
            eval_agent = BestMedianAgent(eval_env, seed=seed)
        elif baseline_type == 'shortest_queue':
            eval_agent = ShortestQueueAgent(eval_env, seed=seed)
        elif baseline_type == 'ground_truth':
            eval_env.ground_truth_replay = True
            eval_agent = RandomAgent(eval_env, seed=seed)  # Actions ignored in replay mode

        eval_evaluator = BaselineEvaluator(eval_env, phase="final_eval")
        eval_results = eval_evaluator.evaluate_agent(
            agent=eval_agent,
            agent_name=f"{agent_name} (test)",
            num_episodes=n_episodes,
            deterministic=True,
        )

        eval_results_file = output_dir / f"{baseline_type}_eval_results.json"
        eval_evaluator.save_results(str(eval_results_file))
        print(f"Eval results saved to: {eval_results_file}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    return results
