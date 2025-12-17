import time
import argparse
import numpy as np
import os
from datetime import datetime
import torch

# Set number of threads for CPU operations
# Use all available CPU cores for PyTorch and NumPy operations
num_cpu_cores = os.cpu_count() or 4
torch.set_num_threads(num_cpu_cores)
torch.set_num_interop_threads(num_cpu_cores)
os.environ['OMP_NUM_THREADS'] = str(num_cpu_cores)
os.environ['MKL_NUM_THREADS'] = str(num_cpu_cores)
print(f"CPU Threading: Using {num_cpu_cores} threads for PyTorch/NumPy operations")

# GPU Optimizations
if torch.cuda.is_available():
    # Enable cuDNN benchmark for faster convolutions (finds optimal algorithms)
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for faster matrix multiplications on Ampere+ GPUs (optional, but safe)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU Optimizations: cuDNN benchmark enabled, TF32 enabled")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import sys
from pathlib import Path
# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.custom_environment import AgentOptimizerEnvironment, SimulationParameters
from src.algorithms.mappo.agent import MAPPOAgent
from src.algorithms.mappo.trainer import MAPPOTrainer
from src.algorithms.baselines.baselines import create_baseline_agents, BaselineEvaluator

from src.core.config import config
from src.core.display import print_colored
from src.core import hyperparameters as hp
from src.preprocessing.load_data import load_data, split_data
from src.preprocessing.preprocessing import remove_short_cases
from src.utils.duration_fitting import (
    fit_duration_distributions_on_training_data,
    save_fitted_distributions,
    load_fitted_distributions,
    print_distribution_summary,
)
from src.algorithms.qmix.agent import QMIXAgent
from src.algorithms.qmix.trainer import QMIXTrainer

from src.core import env_config
from analysis.plotting.plot_distributions import plot_log_metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_gpu_memory():
    """Print GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_colored(
            f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
            f"{max_allocated:.2f}GB peak, {total:.2f}GB total",
            "cyan"
        )


def train_mappo(
    env, experiment_dir: str, total_training_episodes=50, policy_update_epochs=2, online_training=False
):
    """
    Train a MAPPO agent on the environment.

    Args:
        env: Training environment
        experiment_dir: Directory to save training results
        total_training_episodes: Total number of training episodes to run
        policy_update_epochs: Number of epochs for each policy update (PPO inner loop)
        online_training: If True, use online training (update at every timestep, CPU only)
    """
    device = get_device()
    
    # Force CPU for online training since we can't parallelize
    if online_training:
        device = torch.device("cpu")
        print_colored("Online training mode: forcing CPU device (no parallelization)", "yellow")
    
    print_colored(f"Using device: {device}", "yellow")
    if online_training:
        print_colored(
            f"Online training for {total_training_episodes} episodes (updating policy at every timestep)",
            "blue",
        )
    else:
        print_colored(
            f"Training for {total_training_episodes} episodes with {policy_update_epochs} policy update epochs per episode",
            "blue",
        )

    # Initialize MAPPO agent with hyperparameters from config
    mappo_agent = MAPPOAgent(
        env,
        hidden_size=hp.ACTOR_HIDDEN_SIZE,
        lr_actor=hp.LEARNING_RATE_ACTOR,
        lr_critic=hp.LEARNING_RATE_CRITIC,
        gamma=hp.GAMMA,
        num_epochs=policy_update_epochs,  # This is the PPO policy update epochs
        gae_lambda=hp.GAE_LAMBDA,
        clip_param=hp.PPO_CLIP_PARAM,
        batch_size=hp.BATCH_SIZE,
        mini_batch_size=hp.MINI_BATCH_SIZE,  # GPU-optimized mini-batch size
        device=device,
        use_amp=True,  # Enable automatic mixed precision for faster training
    )

    # Print GPU memory usage after agent initialization
    print_gpu_memory()

    # Use different trainer based on training mode
    if online_training:
        from src.algorithms.mappo.online_trainer import MAPPOOnlineTrainer
        trainer = MAPPOOnlineTrainer(
            env,
            mappo_agent,
            total_training_episodes=total_training_episodes,
            should_eval=True,
            eval_episodes=3,
            experiment_dir=experiment_dir,
        )
    else:
        trainer = MAPPOTrainer(
            env,
            mappo_agent,
            total_training_episodes=total_training_episodes,  # Use the parameter
            should_eval=True,  # Enable evaluation during training
            eval_episodes=3,  # Number of evaluation episodes
            experiment_dir=experiment_dir,
        )

    # Start training
    training_mode = "online" if online_training else "batch"
    print_colored(f"Starting MAPPO training ({training_mode} mode)...", "green")
    episode_rewards = trainer.train()

    return mappo_agent, episode_rewards


def evaluate_agent(env, agent, episodes=10, output_dir=None):
    """Evaluate a trained agent on the environment."""
    total_rewards = []

    for ep in range(episodes):
        observations, _ = env.reset()
        episode_reward = 0
        done = False

        print("Starting evaluation")
        print_colored(f"Episode {ep+1}/{episodes}", "blue")
        while not done:
            # Select actions using the policy
            actions, _ = agent.select_actions(observations, deterministic=True)

            # Execute actions
            observations, rewards, terminations, truncations, _ = env.step(actions)

            # Sum rewards across all agents
            step_reward = sum(rewards.values())
            episode_reward += step_reward

            # Check if episode is done
            done = any(list(terminations.values()) + list(truncations.values()))

            if env_config.DEBUG:
                env.render()

        env.reset()

        total_rewards.append(episode_reward)
        print_colored(f"Episode {ep+1} reward: {episode_reward:.2f}", "green")

    avg_reward = np.mean(total_rewards)
    print_colored(f"Average reward over {episodes} episodes: {avg_reward:.2f}", "green")

    if output_dir:
        # Save per-episode rewards
        rewards_file = os.path.join(output_dir, "rewards.csv")
        with open(rewards_file, "a") as f:
            f.write(f"{datetime.now()},{avg_reward:.2f}\n")

    return avg_reward


def evaluate_baselines(env, model_path=None, num_episodes=100):
    """
    Evaluate baseline agents against trained MAPPO agent.

    Args:
        env: The environment to evaluate on
        model_path: Path to trained MAPPO model (optional)
        num_episodes: Number of episodes to evaluate each agent

    Returns:
        Dict with evaluation results
    """
    print_colored("\n" + "=" * 70, "yellow")
    print_colored("BASELINE EVALUATION COMPARISON", "yellow")
    print_colored("=" * 70, "yellow")

    # Create example performance data for BestMedianAgent
    # In practice, this would come from historical data or domain knowledge
    performance_data = {}
    for agent in env.agents:
        # Example: Some agents perform better than others
        # You can replace this with actual historical performance data
        base_performance = np.random.uniform(0.4, 0.8)
        noise = np.random.normal(0, 0.1, 10)
        performance_data[agent.id] = np.clip(
            base_performance + noise, 0.0, 1.0
        ).tolist()

    # Create baseline agents
    random_agent, best_median_agent, ground_truth_agent = create_baseline_agents(
        env, performance_data=performance_data, seed=42
    )

    # Load or create MAPPO agent
    if model_path and os.path.exists(model_path):
        print_colored(f"Loading trained MAPPO model from {model_path}", "green")
        mappo_agent = MAPPOAgent(
            env=env,
            hidden_size=hp.ACTOR_HIDDEN_SIZE,
            lr_actor=hp.LEARNING_RATE_ACTOR,
            lr_critic=hp.LEARNING_RATE_CRITIC,
            gamma=hp.GAMMA,
            gae_lambda=hp.GAE_LAMBDA,
            clip_param=hp.PPO_CLIP_PARAM,
            batch_size=hp.BATCH_SIZE,
            mini_batch_size=hp.MINI_BATCH_SIZE,
            num_epochs=hp.NUM_EPOCHS,
            device=get_device(),
            use_amp=True,
        )
        mappo_agent.load_models(model_path)
    else:
        print_colored(
            "No trained model found, will create untrained MAPPO for comparison",
            "yellow",
        )
        mappo_agent = MAPPOAgent(
            env=env,
            hidden_size=hp.ACTOR_HIDDEN_SIZE,
            lr_actor=hp.LEARNING_RATE_ACTOR,
            lr_critic=hp.LEARNING_RATE_CRITIC,
            gamma=hp.GAMMA,
            gae_lambda=hp.GAE_LAMBDA,
            clip_param=hp.PPO_CLIP_PARAM,
            batch_size=hp.BATCH_SIZE,
            mini_batch_size=hp.MINI_BATCH_SIZE,
            num_epochs=hp.NUM_EPOCHS,
            device=get_device(),
            use_amp=True,
        )

    # Create evaluator
    evaluator = BaselineEvaluator(env)

    # Define agents to compare
    agent_configs = [
        (random_agent, "Random Baseline"),
        (best_median_agent, "Best Median Baseline"),
        (ground_truth_agent, "Ground Truth Baseline"),
        (
            mappo_agent,
            "MAPPO Agent"
            + (
                " (Trained)"
                if model_path and os.path.exists(model_path)
                else " (Untrained)"
            ),
        ),
    ]

    # Run comparison
    results = evaluator.compare_agents(agent_configs, num_episodes=num_episodes)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/output/baseline_comparison_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    evaluator.save_results(results_file)

    return results


def get_model_architecture_summary(agent) -> str:
    """Get a detailed summary of the model architecture."""
    summary = []
    summary.append("Model Architecture")
    summary.append("=" * 18)

    # Get basic configuration
    summary.append(f"Number of agents: {agent.n_agents}")
    summary.append(f"Device: {agent.device}")
    summary.append(f"Gamma (discount factor): {agent.gamma}")
    summary.append(f"GAE Lambda: {agent.gae_lambda}")
    summary.append(f"Clip parameter: {agent.clip_param}")
    summary.append(f"Batch size: {agent.batch_size}")
    summary.append(f"Number of epochs: {agent.num_epochs}")
    summary.append("")

    # Actor networks
    summary.append("Actor Networks:")
    first_actor = next(iter(agent.actors.values()))
    total_actor_params = 0

    for agent_id, actor in agent.actors.items():
        actor_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
        total_actor_params += actor_params
        summary.append(f"  Agent {agent_id}: {actor_params:,} trainable parameters")

    summary.append(f"  Total actor parameters: {total_actor_params:,}")
    summary.append("")

    # Actor architecture details
    summary.append("Actor Architecture:")
    actor_layers = []
    for name, module in first_actor.named_modules():
        if isinstance(module, torch.nn.Linear):
            actor_layers.append(
                f"  {name}: Linear({module.in_features} -> {module.out_features})"
            )
        elif isinstance(module, torch.nn.Dropout):
            actor_layers.append(f"  {name}: Dropout(p={module.p})")

    if actor_layers:
        summary.extend(actor_layers)
    summary.append("")

    # Critic network
    summary.append("Critic Network:")
    critic_params = sum(p.numel() for p in agent.critic.parameters() if p.requires_grad)
    summary.append(f"  Trainable parameters: {critic_params:,}")
    summary.append("")

    # Critic architecture details
    summary.append("Critic Architecture:")
    critic_layers = []
    for name, module in agent.critic.named_modules():
        if isinstance(module, torch.nn.Linear):
            critic_layers.append(
                f"  {name}: Linear({module.in_features} -> {module.out_features})"
            )
        elif isinstance(module, torch.nn.Dropout):
            critic_layers.append(f"  {name}: Dropout(p={module.p})")

    if critic_layers:
        summary.extend(critic_layers)
    summary.append("")

    # Total parameters
    total_params = total_actor_params + critic_params
    summary.append(f"Total trainable parameters: {total_params:,}")
    summary.append("")

    # Optimizer information
    summary.append("Optimizers:")
    first_actor_optimizer = next(iter(agent.actor_optimizers.values()))
    summary.append(
        f"  Actor learning rate: {first_actor_optimizer.param_groups[0]['lr']}"
    )
    summary.append(
        f"  Critic learning rate: {agent.critic_optimizer.param_groups[0]['lr']}"
    )
    summary.append("")

    return "\n".join(summary)


def train_qmix(env, experiment_dir: str, total_training_episodes=50, batch_size=32):
    """
    Train a QMIX agent on the environment.

    Args:
        env: Training environment
        experiment_dir: Directory to save training results
        total_training_episodes: Total number of training episodes to run
        batch_size: Batch size for training
    """

    device = get_device()
    print_colored(f"Using device: {device}", "yellow")
    print_colored(
        f"Training QMIX for {total_training_episodes} episodes with batch size {batch_size}",
        "blue",
    )

    # Initialize QMIX agent with hyperparameters from config
    qmix_agent = QMIXAgent(
        env,
        device=device,
        lr=hp.QMIX_LEARNING_RATE,
        gamma=hp.GAMMA,
        epsilon=hp.QMIX_EPSILON,
    )

    # Initialize trainer
    trainer = QMIXTrainer(
        env,
        qmix_agent,
        total_training_episodes=total_training_episodes,
        batch_size=batch_size,
        buffer_size=hp.QMIX_BUFFER_SIZE,
        target_update_interval=hp.QMIX_TARGET_UPDATE_INTERVAL,
    )

    # Start training
    print_colored("Starting QMIX training...", "green")
    episode_rewards = trainer.train()

    return qmix_agent, episode_rewards


def main(args):
    """Main function to run the environment with proper training and evaluation sequence."""

    # Handle QMIX mode
    if args.mode == "qmix":
        print_colored("=" * 60, "yellow")
        print_colored("STARTING QMIX TRAINING PIPELINE", "yellow")
        print_colored("=" * 60, "yellow")

        # Load and preprocess data
        data = load_data(config)
        data = remove_short_cases(data)
        train, test = split_data(data)
        simulation_parameters = SimulationParameters(
            {"start_timestamp": data["start_timestamp"].min()}
        )

        # Fit duration distributions on training data
        print_colored("Fitting duration distributions on training data...", "blue")
        fitted_distributions = fit_duration_distributions_on_training_data(train)

        # Create timestamped directory for this experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f"./experiments/qmix_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
        print_colored(f"Experiment directory: {experiment_dir}", "green")

        # Save fitted distributions for reproducibility
        distributions_path = os.path.join(experiment_dir, "fitted_distributions.pkl")
        save_fitted_distributions(*fitted_distributions, distributions_path)

        # Training environment
        train_env = AgentOptimizerEnvironment(
            train,
            simulation_parameters,
            experiment_dir=experiment_dir,
            pre_fitted_distributions=fitted_distributions,
        )

        # Train QMIX agent
        agent, episode_rewards = train_qmix(
            train_env,
            experiment_dir,
            total_training_episodes=args.training_episodes,
            batch_size=hp.QMIX_BATCH_SIZE,
        )

        # Save the trained agent
        model_path = os.path.join(experiment_dir, "qmix_agent.pth")
        agent.save_models(model_path)
        print_colored(f"Trained QMIX model saved to: {model_path}", "green")
        train_env.close()

        # Final evaluation on test set
        print_colored("\nEvaluating trained QMIX agent on test set...", "blue")
        eval_env = AgentOptimizerEnvironment(
            test,
            simulation_parameters,
            experiment_dir=os.path.join(experiment_dir, "final_evaluation"),
            pre_fitted_distributions=fitted_distributions,
        )
        avg_reward = evaluate_agent(
            eval_env,
            agent,
            episodes=10,
            output_dir=os.path.join(experiment_dir, "final_evaluation"),
        )

        # Save final summary
        with open(os.path.join(experiment_dir, "training_summary.txt"), "w") as f:
            f.write("QMIX Training Summary\n")
            f.write("=" * 23 + "\n")
            f.write(f"Training episodes: {args.training_episodes}\n")
            f.write("Final evaluation episodes: 10\n")
            f.write(f"Average test reward: {avg_reward:.2f}\n")
            f.write(f"Model saved at: {model_path}\n")
            f.write("\n")

            # Add episode rewards
            f.write("Training Episode Rewards:\n")
            f.write("=" * 26 + "\n")
            for i, reward in enumerate(episode_rewards):
                f.write(f"Episode {i+1:2d}: {reward:8.2f}\n")
            f.write("\n")

            # Add summary statistics
            if episode_rewards:
                f.write("Training Reward Statistics:\n")
                f.write(f"  Total episodes: {len(episode_rewards)}\n")
                f.write(f"  Average reward: {np.mean(episode_rewards):8.2f}\n")
                f.write(f"  Best reward:    {np.max(episode_rewards):8.2f}\n")
                f.write(f"  Worst reward:   {np.min(episode_rewards):8.2f}\n")
                f.write(f"  Std deviation:  {np.std(episode_rewards):8.2f}\n")
                f.write("\n")

        print_colored("\n🎉 Training pipeline completed successfully!", "green")
        print_colored(f"📊 Average test reward: {avg_reward:.2f}", "green")
        print_colored(f"📁 Results saved in: {experiment_dir}", "green")
        eval_env.close()
        return

    # Handle MAPPO mode (train)
    print_colored("=" * 60, "yellow")
    print_colored("STARTING MAPPO TRAINING PIPELINE", "yellow")
    print_colored("=" * 60, "yellow")

    # Load and preprocess data
    print_colored("1. Loading and preprocessing data...", "blue")
    data = load_data(config)
    data = remove_short_cases(data)
    train, test = split_data(data)

    # Fit duration distributions on training data ONCE
    print_colored("2. Fitting duration distributions on training data...", "blue")
    fitted_distributions = fit_duration_distributions_on_training_data(train)
    activity_durations_dict, stats_dict, global_activity_medians = fitted_distributions

    # Print summary of fitted distributions
    print_distribution_summary(activity_durations_dict, stats_dict)

    simulation_parameters = SimulationParameters(
        {"start_timestamp": data["start_timestamp"].min()}
    )

    # Create timestamped directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiments/mappo_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    print_colored(f"Experiment directory: {experiment_dir}", "green")

    # Save fitted distributions for reproducibility
    distributions_path = os.path.join(experiment_dir, "fitted_distributions.pkl")
    save_fitted_distributions(
        activity_durations_dict, stats_dict, global_activity_medians, distributions_path
    )

    # Step 3: Single test run with random actions (using fitted distributions)
    print_colored("\n3. Running single test episode with random actions...", "blue")
    test_env = AgentOptimizerEnvironment(
        train,
        simulation_parameters,
        experiment_dir=os.path.join(experiment_dir, "test_run"),
        pre_fitted_distributions=fitted_distributions,  # Use fitted distributions
    )

    # Run one episode with random actions to verify environment
    observations, _ = test_env.reset()
    episode_reward = 0
    step_count = 0

    print_colored("Running test episode with random actions...", "cyan")
    while True:
        actions = {agent.id: np.random.choice([0, 1]) for agent in test_env.agents}
        observations, rewards, terminations, truncations, _ = test_env.step(actions)
        episode_reward += sum(rewards.values())
        step_count += 1

        if any(terminations.values()) or any(truncations.values()):
            break

    print_colored(
        f"Test episode completed: {step_count} steps, total reward: {episode_reward:.2f}",
        "green",
    )
    test_env.close()

    # Step 4: Training (using fitted distributions from training data)
    print_colored("\n4. Starting MAPPO training...", "blue")
    train_env = AgentOptimizerEnvironment(
        train,
        simulation_parameters,
        experiment_dir=experiment_dir,
        pre_fitted_distributions=fitted_distributions,  # Use fitted distributions
    )

    # Train MAPPO agent with configurable parameters
    agent, episode_rewards = train_mappo(
        train_env,
        experiment_dir,
        total_training_episodes=args.training_episodes,  # Use configurable parameter
        policy_update_epochs=args.policy_epochs,  # Use configurable parameter
        online_training=args.online_training,  # Use online training flag
    )

    # Save the trained agent
    model_path = os.path.join(experiment_dir, "mappo_agent.pth")
    agent.save_models(model_path)
    print_colored(f"Trained model saved to: {model_path}", "green")
    train_env.close()

    # Step 5: Final evaluation on test set (using fitted distributions from training data)
    print_colored("\n5. Evaluating trained agent on test set...", "blue")
    eval_env = AgentOptimizerEnvironment(
        test,
        simulation_parameters,
        experiment_dir=os.path.join(experiment_dir, "final_evaluation"),
        pre_fitted_distributions=fitted_distributions,  # Use fitted distributions from training data
    )

    # Evaluate the trained agent
    avg_reward = evaluate_agent(
        eval_env,
        agent,
        episodes=10,  # Number of evaluation episodes
        output_dir=os.path.join(experiment_dir, "final_evaluation"),
    )

    # Save final summary
    with open(os.path.join(experiment_dir, "training_summary.txt"), "w") as f:
        f.write("MAPPO Training Summary\n")
        f.write("=" * 25 + "\n")
        f.write(f"Training episodes: {args.training_episodes}\n")
        f.write(f"Policy update epochs per episode: {args.policy_epochs}\n")
        f.write("Final evaluation episodes: 10\n")
        f.write(f"Average test reward: {avg_reward:.2f}\n")
        f.write(f"Model saved at: {model_path}\n")
        f.write("\n")

        # Add model architecture information
        model_architecture = get_model_architecture_summary(agent)
        f.write(model_architecture)

        # Add episode rewards
        f.write("Training Episode Rewards:\n")
        f.write("=" * 26 + "\n")
        for i, reward in enumerate(episode_rewards):
            f.write(f"Episode {i+1:2d}: {reward:8.2f}\n")
        f.write("\n")

        # Add summary statistics
        if episode_rewards:
            f.write("Training Reward Statistics:\n")
            f.write(f"  Total episodes: {len(episode_rewards)}\n")
            f.write(f"  Average reward: {np.mean(episode_rewards):8.2f}\n")
            f.write(f"  Best reward:    {np.max(episode_rewards):8.2f}\n")
            f.write(f"  Worst reward:   {np.min(episode_rewards):8.2f}\n")
            f.write(f"  Std deviation:  {np.std(episode_rewards):8.2f}\n")
            f.write("\n")

    print_colored("\n🎉 Training pipeline completed successfully!", "green")
    print_colored(f"📊 Average test reward: {avg_reward:.2f}", "green")
    print_colored(f"📁 Results saved in: {experiment_dir}", "green")

    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the custom environment with MAPPO."
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Enable debugging output",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "random", "baseline", "qmix"],
        default="train",
        help="Mode to run: train (MAPPO), qmix (QMIX), evaluate, random actions, or baseline comparison",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--training-episodes",
        type=int,
        default=50,
        help="Number of training episodes (for train mode)",
    )
    parser.add_argument(
        "--policy-epochs",
        type=int,
        default=5,
        help="Number of policy update epochs per training episode",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/mappo_final",
        help="Path to load model from for evaluation",
    )
    parser.add_argument(
        "--online-training",
        action="store_true",
        default=False,
        help="Use online training mode (update policy at every timestep)",
    )

    args = parser.parse_args()
    env_config.DEBUG = args.debug

    if env_config.DEBUG:
        env_config.debug_print_colored("Debugging mode is enabled.", "green")

    if args.mode == "evaluate":
        # Load and preprocess data
        data = load_data(config)
        data = remove_short_cases(data)
        train, test = split_data(data)
        simulation_parameters = SimulationParameters(
            {"start_timestamp": data["start_timestamp"].min()}
        )

        # Determine experiment folder from model path
        experiment_dir = os.path.dirname(args.model_path.rstrip("/"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(experiment_dir, f"evaluation_{timestamp}")
        os.makedirs(eval_dir, exist_ok=True)

        # Try to load pre-fitted distributions from the experiment directory
        distributions_path = os.path.join(experiment_dir, "fitted_distributions.pkl")
        fitted_distributions = None

        if os.path.exists(distributions_path):
            print_colored("Loading pre-fitted distributions from training...", "blue")
            fitted_distributions = load_fitted_distributions(distributions_path)
            activity_durations_dict, stats_dict, global_activity_medians = (
                fitted_distributions
            )
            print_distribution_summary(activity_durations_dict, stats_dict)
        else:
            print_colored(
                "Pre-fitted distributions not found, fitting on training data...",
                "yellow",
            )
            fitted_distributions = fit_duration_distributions_on_training_data(train)

        env = AgentOptimizerEnvironment(
            test,
            simulation_parameters,
            experiment_dir=eval_dir,  # Pass evaluation directory for logs
            pre_fitted_distributions=fitted_distributions,  # Use fitted distributions
        )

        # Load trained agent with device and hyperparameters
        device = get_device()
        agent = MAPPOAgent(
            env,
            hidden_size=hp.ACTOR_HIDDEN_SIZE,
            lr_actor=hp.LEARNING_RATE_ACTOR,
            lr_critic=hp.LEARNING_RATE_CRITIC,
            gamma=hp.GAMMA,
            gae_lambda=hp.GAE_LAMBDA,
            clip_param=hp.PPO_CLIP_PARAM,
            batch_size=hp.BATCH_SIZE,
            mini_batch_size=hp.MINI_BATCH_SIZE,
            num_epochs=hp.NUM_EPOCHS,
            device=device,
            use_amp=True,
        )
        agent.load_models(args.model_path)

        # Evaluate the agent and save logs
        avg_reward = evaluate_agent(
            env, agent, episodes=args.episodes, output_dir=eval_dir
        )

        # Save summary
        with open(os.path.join(eval_dir, "summary.txt"), "w") as f:
            f.write("Evaluation completed\n")
            f.write(f"Average reward over {args.episodes} episodes: {avg_reward:.2f}\n")

        # Create plots
        print_colored("Creating evaluation plots...", "blue")
        # Take the logs in the current directory
        plot_log_metrics(experiment_dir=eval_dir)

        print_colored(f"Evaluation results saved in: {eval_dir}", "green")

        evaluate_baselines(
            env,
            model_path=args.model_path if os.path.exists(args.model_path) else None,
            num_episodes=10,
        )

        env.close()

    elif args.mode == "baseline":
        # Load and preprocess data
        data = load_data(config)
        data = remove_short_cases(data)
        train, test = split_data(data)
        simulation_parameters = SimulationParameters(
            {"start_timestamp": data["start_timestamp"].min()}
        )

        # Fit duration distributions on training data for consistent evaluation
        print_colored("Fitting duration distributions on training data...", "blue")
        fitted_distributions = fit_duration_distributions_on_training_data(train)

        # Create timestamped directory for baseline evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_dir = f"./experiments/baseline_{timestamp}"
        os.makedirs(baseline_dir, exist_ok=True)

        # Save fitted distributions
        distributions_path = os.path.join(baseline_dir, "fitted_distributions.pkl")
        save_fitted_distributions(*fitted_distributions, distributions_path)

        # Use test data for evaluation with fitted distributions from training data
        env = AgentOptimizerEnvironment(
            test,
            simulation_parameters,
            experiment_dir=baseline_dir,
            pre_fitted_distributions=fitted_distributions,  # Use fitted distributions
        )

        # Run baseline evaluation
        print_colored("Running baseline comparison evaluation...", "blue")
        results = evaluate_baselines(
            env,
            model_path=args.model_path if os.path.exists(args.model_path) else None,
            num_episodes=100,
        )

        # Save detailed results
        import json

        with open(os.path.join(baseline_dir, "baseline_results.json"), "w") as f:
            # Convert results to be JSON serializable
            serializable_results: dict = {}
            for name, result in results.items():
                serializable_results[name] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[name][key] = value.tolist()
                    elif hasattr(value, "item"):  # numpy scalars
                        serializable_results[name][key] = float(value)
                    else:
                        serializable_results[name][key] = value
            json.dump(serializable_results, f, indent=2)

        print_colored(f"Baseline evaluation results saved in: {baseline_dir}", "green")
        env.close()

    elif args.mode == "random":
        # Load and preprocess data
        data = load_data(config)
        data = remove_short_cases(data)
        train, test = split_data(data)
        simulation_parameters = SimulationParameters(
            {"start_timestamp": data["start_timestamp"].min()}
        )

        # Fit duration distributions on training data for consistency
        print_colored("Fitting duration distributions on training data...", "blue")
        fitted_distributions = fit_duration_distributions_on_training_data(train)

        env = AgentOptimizerEnvironment(
            train,
            simulation_parameters,
            pre_fitted_distributions=fitted_distributions,  # Use fitted distributions
        )
        # Run with random actions for baseline comparison
        i = 0
        start_time = time.perf_counter()
        try:
            observations, _ = env.reset()
            while True:
                env_config.debug_print_colored(
                    f"Step {i}, time: {env.current_time.time()}", "blue"
                )
                actions = {agent.id: np.random.choice([0, 1]) for agent in env.agents}
                observations, rewards, terminations, truncations, _ = env.step(actions)
                if any(terminations.values()):
                    print("Episode finished.")
                    break
                if any(truncations.values()):
                    print("Episode truncated.")
                    break
                i += 1
                if env_config.DEBUG:
                    env.render()
        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print_colored(f"\nElapsed time: {elapsed_time:.2f} seconds", "green")

        # Close the environment
        env.close()

    elif args.mode == "train":
        main(args)

    elif args.mode == "qmix":
        main(args)
