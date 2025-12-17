"""
Baseline agents for comparison with MAPPO.
These agents don't require training and are used for evaluation only.
"""

import numpy as np
import torch
from typing import Any, Dict, List
from collections import defaultdict

from ...core.display import print_colored


class RandomAgent:
    """
    Baseline agent that selects actions randomly.
    """

    def __init__(self, env, seed=None):
        self.env = env
        self.agents = env.agents
        self.rng = np.random.RandomState(seed)
        print_colored("RandomAgent initialized - selects actions randomly", "blue")

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
    Baseline agent where only the agent with the best median performance
    for upcoming tasks raises their hand (takes action 1), others don't raise (action 0).
    """

    def __init__(self, env, performance_data=None, seed=None):
        self.env = env
        self.agents = env.agents
        self.rng = np.random.RandomState(seed)
        self.performance_data = performance_data or {}
        self.agent_medians = {}
        self._compute_agent_medians()
        print_colored(
            "BestMedianAgent initialized - only best performer raises hand", "blue"
        )

    def _compute_agent_medians(self):
        """Compute median performance for each agent."""
        if not self.performance_data:
            # If no performance data provided, use random medians for demo
            print_colored(
                "No performance data provided, using random medians for demo", "yellow"
            )
            for agent in self.agents:
                self.agent_medians[agent.id] = self.rng.uniform(0.3, 0.9)
        else:
            # Compute actual medians from performance data
            for agent_id, performances in self.performance_data.items():
                if performances:
                    self.agent_medians[agent_id] = np.median(performances)
                else:
                    self.agent_medians[agent_id] = 0.0

        print_colored(f"Agent median performances: {self.agent_medians}", "cyan")

    def set_performance_data(self, performance_data: Dict[str, List[float]]):
        """Update performance data and recompute medians."""
        self.performance_data = performance_data
        self._compute_agent_medians()

    def select_actions(self, observations, deterministic=True):
        """
        Select actions where only the best median performer raises hand.
        Action 0 = don't raise hand, Action 1 = raise hand
        """
        actions = {}
        action_probs = {}

        # Find the agent with the best (highest) median performance
        if self.agent_medians:
            best_agent_id = max(
                self.agent_medians.keys(), key=lambda x: self.agent_medians[x]
            )
        else:
            # Fallback: pick first agent if no performance data
            best_agent_id = self.agents[0].id if self.agents else None

        for agent_id, obs in observations.items():
            if agent_id in [agent.id for agent in self.agents]:
                action_space = self.env.action_space(agent_id)
                n_actions = action_space.n

                if agent_id == best_agent_id:
                    # Best performer raises hand (action 1)
                    action = 1 if n_actions > 1 else 0
                    # High probability for chosen action
                    probs = torch.zeros(n_actions)
                    probs[action] = 0.95
                    if n_actions > 1:
                        probs[1 - action] = 0.05
                else:
                    # Other agents don't raise hand (action 0)
                    action = 0
                    # High probability for not raising hand
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
        Select actions so that only the ground truth assigned agent acts (e.g., raises hand).
        Assumes the observation for each agent contains the assigned agent id under self.assigned_agent_key.
        """
        actions = {}
        action_probs = {}

        # Find the assigned agent id from any agent's observation (all should agree)
        assigned_agent_id = None
        # Group the self.env.data by case_id, sort by start_timestamp and get the corresponding task and its assigned agent
        # env.data is a pandas DataFrame with columns: case_id, resource, activity_name, start_timestamp, end_timestamp
        # Get the current timestep from the environment
        current_timestep = getattr(self.env, "current_timestep", 0)

        # Find the assigned agent for the current task
        if hasattr(self.env, "data") and self.env.data is not None:
            # Get current case_id from the environment
            current_case_id = getattr(self.env, "current_case_id", None)

            if current_case_id is not None:
                # Filter data for current case and sort by start_timestamp
                case_data = self.env.data[
                    self.env.data["case_id"] == current_case_id
                ].sort_values("start_timestamp")

                # Get the task at current timestep
                if current_timestep < len(case_data):
                    current_task = case_data.iloc[current_timestep]
                    assigned_agent_id = current_task["resource"]
                else:
                    # Fallback if timestep is out of bounds
                    assigned_agent_id = self.agents[0].id if self.agents else None
            else:
                # Fallback if no current case
                assigned_agent_id = self.agents[0].id if self.agents else None
        else:
            # Fallback if no data available
            assigned_agent_id = self.agents[0].id if self.agents else None

        assigned_agent_id = (
            self.env.resource_name_to_id(assigned_agent_id)
            if assigned_agent_id
            else None
        )

        for agent_id, obs in observations.items():
            if agent_id in [agent.id for agent in self.agents]:
                action_space = self.env.action_space(agent_id)
                n_actions = action_space.n

                if agent_id == assigned_agent_id:
                    # Assigned agent acts (e.g., raises hand)
                    action = 1 if n_actions > 1 else 0
                    probs = torch.zeros(n_actions)
                    probs[action] = 1.0
                else:
                    # Other agents do not act
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

    def __init__(self, env):
        self.env = env
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
        print_colored(
            f"\nEvaluating {agent_name} for {num_episodes} episodes...", "yellow"
        )

        episode_rewards = []
        episode_lengths = []
        task_success_rates = defaultdict(list)
        total_steps = 0

        for episode in range(num_episodes):
            # Reset environment and agent
            reset_result = self.env.reset()
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

                # Check if episode is done
                episode_done = (
                    any(dones.values()) or episode_length >= 1000
                )  # Max episode length

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
                print_colored(
                    f"  Episode {episode + 1}/{num_episodes} - Avg Reward (last 10): {avg_reward:.2f}",
                    "cyan",
                )

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
        print_colored(f"\n{agent_name} Evaluation Results:", "green")
        print_colored(
            f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
            "white",
        )
        print_colored(f"  Median Reward: {results['median_reward']:.2f}", "white")
        print_colored(
            f"  Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]",
            "white",
        )
        print_colored(
            f"  Mean Episode Length: {results['mean_episode_length']:.1f}", "white"
        )
        print_colored(f"  Total Steps: {results['total_steps']}", "white")

        if task_success_rates:
            for agent_id in task_success_rates:
                success_rate = results[f"task_success_rate_{agent_id}"]
                print_colored(
                    f"  Task Success Rate ({agent_id}): {success_rate:.2%}", "white"
                )

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

        print_colored(f"Results saved to {filepath}", "green")


def create_baseline_agents(env, performance_data=None, seed=42):
    """
    Factory function to create baseline agents.

    Args:
        env: The environment
        performance_data: Dict mapping agent_id to list of performance scores
        seed: Random seed for reproducibility
        include_ground_truth: If True, also return the GroundTruthAssignmentAgent

    Returns:
        Tuple of (random_agent, best_median_agent[, ground_truth_agent])
    """
    random_agent = RandomAgent(env, seed=seed)
    best_median_agent = BestMedianAgent(
        env, performance_data=performance_data, seed=seed
    )
    ground_truth_agent = GroundTruthAssignmentAgent(env)

    return random_agent, best_median_agent, ground_truth_agent
