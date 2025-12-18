#!/usr/bin/env python3
"""
Consolidated baseline evaluation script for multiple datasets.
Evaluates all baseline agents and trained models (MAPPO, QMIX) across different datasets.

Baseline agents include:
1. Random Agent - selects actions randomly
2. Best Median Agent - only the agent with the best median performance raises hand
3. Ground Truth Agent - follows the actual agent assignments from the data

Trained agents include:
1. MAPPO Agent - Multi-Agent Proximal Policy Optimization
2. QMIX Agent - Q-Mixing Networks

The script creates experiment directories with dataset names in the filename for easy identification.

Usage:
    python run_multi_dataset_baseline_evaluation.py [options]

Options:
    --datasets LIST         Datasets to evaluate (default: all available)
    --episodes N           Number of evaluation episodes per dataset (default: 20)
    --seed N              Random seed for reproducibility (default: 42)
    --use-test-data       Use test data split instead of training data
    --include-trained     Include trained agents (MAPPO, QMIX) in comparison
    --mappo-model-path    Path to trained MAPPO model directory
    --qmix-model-path     Path to trained QMIX model directory
    --output-dir          Base directory for experiment outputs (default: ./experiments)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import json
from typing import Dict, List, Tuple, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import custom modules
from src.CustomEnvironment.custom_environment import AgentOptimizerEnvironment
from src.CustomEnvironment.custom_environment.env.custom_environment import (
    SimulationParameters,
)
from src.MAPPO.agent import MAPPOAgent
from src.QMIX.agent import QMIXAgent
from src.config import config
from src.display import print_colored
from src.preprocessing.load_data import load_data, split_data
from src.preprocessing.preprocessing import remove_short_cases


# Dataset configurations
DATASET_CONFIGS = {
    "BPI12W": {
        "filename": "BPI12W.csv",
        "case_id_col": "case_id",
        "resource_id_col": "resource",
        "activity_col": "activity_name",
        "start_timestamp_col": "start_timestamp",
        "end_timestamp_col": "end_timestamp",
    },
    "BPI12W_1": {
        "filename": "BPI12W_1.csv",
        "case_id_col": "case_id",
        "resource_id_col": "resource",
        "activity_col": "activity_name",
        "start_timestamp_col": "start_timestamp",
        "end_timestamp_col": "end_timestamp",
    },
    "BPI12W_2": {
        "filename": "BPI12W_2.csv",
        "case_id_col": "case_id",
        "resource_id_col": "resource",
        "activity_col": "activity_name",
        "start_timestamp_col": "start_timestamp",
        "end_timestamp_col": "end_timestamp",
    },
    "LoanApp": {
        "filename": "LoanApp.csv",
        "case_id_col": "case_id",
        "resource_id_col": "resource",
        "activity_col": "activity_name",
        "start_timestamp_col": "start_timestamp",
        "end_timestamp_col": "end_timestamp",
    },
    "CVS_Pharmacy": {
        "filename": "cvs_pharmacy.csv",
        "case_id_col": "case_id",
        "resource_id_col": "resource",
        "activity_col": "activity_name",
        "start_timestamp_col": "start_timestamp",
        "end_timestamp_col": "end_timestamp",
    },
    "Train_Preprocessed": {
        "filename": "train_preprocessed.csv",
        "case_id_col": "case_id",
        "resource_id_col": "resource",
        "activity_col": "activity_name",
        "start_timestamp_col": "start_timestamp",
        "end_timestamp_col": "end_timestamp",
    },
    "BPI12W_Processed": {
        "filename": "BPI12w_processed.csv",
        "case_id_col": "case_id",
        "resource_id_col": "resource",
        "activity_col": "activity_name",
        "start_timestamp_col": "start_timestamp",
        "end_timestamp_col": "end_timestamp",
    },
}


class RandomAgent:
    """Baseline agent that selects actions randomly."""

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
                action_space = self.env.action_space(agent_id)
                action = self.rng.randint(0, action_space.n)
                actions[agent_id] = action

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
    """Agent where only the best median performer raises hand."""

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
            print_colored(
                "No performance data provided, using random medians for demo", "yellow"
            )
            for agent in self.agents:
                self.agent_medians[agent.id] = self.rng.uniform(0.3, 0.9)
        else:
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
        """Select actions where only the best median performer raises hand."""
        actions = {}
        action_probs = {}

        if self.agent_medians:
            best_agent_id = max(
                self.agent_medians.keys(), key=lambda x: self.agent_medians[x]
            )
        else:
            best_agent_id = self.agents[0].id if self.agents else None

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


class GroundTruthAssignmentAgent:
    """Agent that follows actual assignments from the data."""

    def __init__(self, env, assigned_agent_key="assigned_agent_id"):
        self.env = env
        self.agents = env.agents
        self.assigned_agent_key = assigned_agent_key
        print_colored(
            "GroundTruthAssignmentAgent initialized - uses ground truth assignments",
            "blue",
        )

    def select_actions(self, observations, deterministic=True):
        """Select actions so that only the ground truth assigned agent acts."""
        actions = {}
        action_probs = {}

        current_timestep = getattr(self.env, "current_timestep", 0)
        assigned_agent_id = None

        if hasattr(self.env, "data") and self.env.data is not None:
            current_case_id = getattr(self.env, "current_case_id", None)

            if current_case_id is not None:
                case_data = self.env.data[
                    self.env.data["case_id"] == current_case_id
                ].sort_values("start_timestamp")

                if current_timestep < len(case_data):
                    current_task = case_data.iloc[current_timestep]
                    assigned_agent_id = current_task["resource"]
                else:
                    assigned_agent_id = self.agents[0].id if self.agents else None
            else:
                assigned_agent_id = self.agents[0].id if self.agents else None
        else:
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
    """Utility class to evaluate baseline and trained agents."""

    def __init__(self, env):
        self.env = env
        self.results = {}

    def evaluate_agent(
        self, agent, agent_name: str, num_episodes: int = 20
    ) -> Dict[str, Any]:
        """Evaluate a single agent."""
        print_colored(f"\nEvaluating {agent_name}...", "yellow")

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            print_colored(f"Episode {episode + 1}/{num_episodes}", "cyan")

            # Reset environment
            observations, infos = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Agent selects actions
                if hasattr(agent, "select_actions"):
                    actions, _ = agent.select_actions(observations)
                else:
                    # For trained agents like MAPPO/QMIX that might have different interfaces
                    actions = agent.get_actions(observations)

                # Environment step
                next_observations, rewards, terminations, truncations, infos = (
                    self.env.step(actions)
                )

                # Accumulate rewards (sum across all agents)
                step_reward = (
                    sum(rewards.values()) if isinstance(rewards, dict) else rewards
                )
                episode_reward += step_reward
                episode_length += 1

                # Check if episode is done
                done = any(terminations.values()) or any(truncations.values())
                observations = next_observations

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print_colored(
                f"Episode {episode + 1} - Reward: {episode_reward:.2f}, Length: {episode_length}",
                "green",
            )

        # Compute statistics
        results = {
            "agent_name": agent_name,
            "num_episodes": num_episodes,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "median_reward": np.median(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "total_steps": sum(episode_lengths),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        print_colored(f"{agent_name} Results:", "blue")
        print_colored(
            f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
            "white",
        )
        print_colored(f"  Median Reward: {results['median_reward']:.2f}", "white")
        print_colored(
            f"  Mean Episode Length: {results['mean_episode_length']:.2f}", "white"
        )

        return results

    def compare_agents(
        self, agent_configs: List[Tuple], num_episodes: int = 20
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple agents."""
        results = {}

        for agent, agent_name in agent_configs:
            results[agent_name] = self.evaluate_agent(agent, agent_name, num_episodes)

        return results

    def save_results(self, results: Dict[str, Dict[str, Any]], filepath: str):
        """Save results to JSON file."""

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Deep convert all numpy types
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(item) for item in data]
            else:
                return convert_numpy(data)

        converted_results = deep_convert(results)

        with open(filepath, "w") as f:
            json.dump(converted_results, f, indent=2)

        print_colored(f"Results saved to: {filepath}", "green")


def get_device():
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_trained_mappo_agent(env, model_path=None):
    """Load a trained MAPPO agent."""
    device = get_device()
    print_colored(f"Using device: {device}", "yellow")

    if model_path and os.path.exists(model_path):
        print_colored(f"Loading trained MAPPO model from {model_path}", "green")
        mappo_agent = MAPPOAgent(
            env=env,
            hidden_size=64,
            lr_actor=0.0003,
            lr_critic=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
            clip_param=0.2,
            batch_size=1028,
            num_epochs=5,
            device=device,
        )
        mappo_agent.load_models(model_path)
        return mappo_agent
    else:
        print_colored(
            "No trained MAPPO model found, skipping MAPPO evaluation", "yellow"
        )
        return None


def load_trained_qmix_agent(env, model_path=None):
    """Load a trained QMIX agent."""
    device = get_device()

    if model_path and os.path.exists(model_path):
        print_colored(f"Loading trained QMIX model from {model_path}", "green")
        # Create QMIX agent - you may need to adjust parameters based on your implementation
        qmix_agent = QMIXAgent(
            env=env,
            device=device,
        )
        qmix_agent.load_models(model_path)
        return qmix_agent
    else:
        print_colored("No trained QMIX model found, skipping QMIX evaluation", "yellow")
        return None


def create_performance_data(env):
    """Create performance data for BestMedianAgent based on agent capabilities."""
    performance_data = {}

    for agent in env.agents:
        base_performance = np.random.uniform(0.4, 0.8)
        performance_variations = np.random.normal(0, 0.1, 10)
        performance_data[agent.id] = np.clip(
            base_performance + performance_variations, 0.0, 1.0
        ).tolist()

        try:
            if hasattr(agent, "capabilities") and hasattr(agent, "stats_dict"):
                capable_tasks = sum(
                    1 for v in agent.capabilities.values() if v is not None
                )
                capability_ratio = (
                    capable_tasks / len(agent.capabilities)
                    if agent.capabilities
                    else 0.5
                )

                adjusted_base = 0.3 + (0.6 * capability_ratio)
                variations = np.random.normal(0, 0.15, 10)
                performance_data[agent.id] = np.clip(
                    adjusted_base + variations, 0.0, 1.0
                ).tolist()
        except Exception:
            pass

    return performance_data


def load_dataset(dataset_name: str, data_dir: str = "./data/input") -> pd.DataFrame:
    """Load and preprocess a specific dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not found in configurations")

    dataset_config = DATASET_CONFIGS[dataset_name]
    filepath = os.path.join(data_dir, dataset_config["filename"])

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    print_colored(f"Loading dataset: {dataset_name} from {filepath}", "blue")

    # Update global config with dataset-specific config
    original_config = config.copy()
    config.update(
        {
            "input_filename": dataset_config["filename"],
            "case_id_col": dataset_config["case_id_col"],
            "resource_id_col": dataset_config["resource_id_col"],
            "activity_col": dataset_config["activity_col"],
            "start_timestamp_col": dataset_config["start_timestamp_col"],
            "end_timestamp_col": dataset_config["end_timestamp_col"],
        }
    )

    try:
        data = load_data(config)
        data = remove_short_cases(data)
        print_colored(
            f"Dataset {dataset_name} loaded successfully: {len(data)} records", "green"
        )
        return data
    finally:
        # Restore original config
        config.clear()
        config.update(original_config)


def run_dataset_evaluation(dataset_name: str, args) -> Dict[str, Any]:
    """Run baseline evaluation for a single dataset."""
    print_colored("=" * 80, "yellow")
    print_colored(f"BASELINE EVALUATION - DATASET: {dataset_name}", "yellow")
    print_colored("=" * 80, "yellow")

    # Load dataset
    try:
        data = load_dataset(dataset_name)
    except (FileNotFoundError, ValueError) as e:
        print_colored(f"Error loading dataset {dataset_name}: {e}", "red")
        return {}

    # Split data
    train, test = split_data(data)
    evaluation_data = test if args.use_test_data else train

    # Create experiment directory with dataset name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(
        args.output_dir, f"baseline_evaluation_{dataset_name}_{timestamp}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    print_colored(f"Experiment directory: {experiment_dir}", "green")

    # Create environment
    simulation_parameters = SimulationParameters(
        {"start_timestamp": evaluation_data["start_timestamp"].min()}
    )

    env = AgentOptimizerEnvironment(
        evaluation_data,
        simulation_parameters,
        experiment_dir=experiment_dir,
    )

    # Generate performance data for BestMedianAgent
    performance_data = create_performance_data(env)

    # Create baseline agents
    random_agent = RandomAgent(env, seed=args.seed)
    best_median_agent = BestMedianAgent(
        env, performance_data=performance_data, seed=args.seed
    )
    ground_truth_agent = GroundTruthAssignmentAgent(env)

    # Create evaluator
    evaluator = BaselineEvaluator(env)

    # Define agents to compare
    agent_configs = [
        (random_agent, "Random Baseline"),
        (best_median_agent, "Best Median Baseline"),
        (ground_truth_agent, "Ground Truth Baseline"),
    ]

    # Load trained agents if requested
    if args.include_trained:
        mappo_agent = load_trained_mappo_agent(env, args.mappo_model_path)
        if mappo_agent:
            agent_configs.append((mappo_agent, "MAPPO Agent (Trained)"))

        qmix_agent = load_trained_qmix_agent(env, args.qmix_model_path)
        if qmix_agent:
            agent_configs.append((qmix_agent, "QMIX Agent (Trained)"))

    # Run comparison
    results = evaluator.compare_agents(agent_configs, num_episodes=args.episodes)

    # Save results with dataset name in filename
    results_file = os.path.join(
        experiment_dir, f"baseline_comparison_results_{dataset_name}.json"
    )
    evaluator.save_results(results, results_file)

    # Close environment
    env.close()

    print_colored(f"\nBaseline evaluation for {dataset_name} completed!", "green")
    print_colored(f"Results saved to: {results_file}", "green")

    return {
        "dataset": dataset_name,
        "results": results,
        "experiment_dir": experiment_dir,
        "results_file": results_file,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-dataset baseline evaluations for Agent Optimizer"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Datasets to evaluate (default: all available)",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per dataset",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--use-test-data",
        action="store_true",
        help="Use test data split instead of training data",
    )

    parser.add_argument(
        "--include-trained",
        action="store_true",
        help="Include trained agents (MAPPO, QMIX) in comparison",
    )

    parser.add_argument(
        "--mappo-model-path",
        type=str,
        default="./models/mappo_final",
        help="Path to trained MAPPO model directory",
    )

    parser.add_argument(
        "--qmix-model-path",
        type=str,
        default="./models/qmix_final",
        help="Path to trained QMIX model directory",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments",
        help="Base directory for experiment outputs",
    )

    args = parser.parse_args()

    # Determine which datasets to run
    if "all" in args.datasets:
        datasets_to_run = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_run = args.datasets

    # Filter datasets to only include those that exist
    available_datasets = []
    for dataset in datasets_to_run:
        dataset_file = os.path.join(
            "./data/input", DATASET_CONFIGS[dataset]["filename"]
        )
        if os.path.exists(dataset_file):
            available_datasets.append(dataset)
        else:
            print_colored(
                f"Dataset file not found, skipping {dataset}: {dataset_file}", "yellow"
            )

    if not available_datasets:
        print_colored("No valid datasets found!", "red")
        return

    print_colored(
        f"Running evaluation on {len(available_datasets)} datasets: {available_datasets}",
        "cyan",
    )

    # Run evaluation for each dataset
    all_results = {}
    for dataset in available_datasets:
        dataset_result = run_dataset_evaluation(dataset, args)
        if dataset_result:
            all_results[dataset] = dataset_result

    # Save summary of all results
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(
            args.output_dir, f"multi_dataset_summary_{timestamp}.json"
        )

        with open(summary_file, "w") as f:
            # Only save metadata and summary stats, not full results
            summary = {}
            for dataset, result in all_results.items():
                summary[dataset] = {
                    "experiment_dir": result["experiment_dir"],
                    "results_file": result["results_file"],
                    "agents_evaluated": list(result["results"].keys()),
                    "summary_stats": {
                        agent_name: {
                            "mean_reward": agent_results["mean_reward"],
                            "median_reward": agent_results["median_reward"],
                            "num_episodes": agent_results["num_episodes"],
                        }
                        for agent_name, agent_results in result["results"].items()
                    },
                }
            json.dump(summary, f, indent=2)

        print_colored(
            f"\nMulti-dataset evaluation summary saved to: {summary_file}", "green"
        )
        print_colored("=" * 80, "yellow")
        print_colored("EVALUATION COMPLETE", "yellow")
        print_colored("=" * 80, "yellow")


if __name__ == "__main__":
    main()
