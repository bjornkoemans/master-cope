#!/usr/bin/env python3
"""
Phase 3: Training Only
Trains MAPPO agent using preprocessed data and fitted distributions.
Can be run MULTIPLE TIMES with different configurations for hyperparameter tuning.
"""

import time
import argparse
import pickle
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
import torch

# Set number of threads for CPU operations
num_cpu_cores = os.cpu_count() or 4
torch.set_num_threads(num_cpu_cores)
torch.set_num_interop_threads(num_cpu_cores)
os.environ['OMP_NUM_THREADS'] = str(num_cpu_cores)
os.environ['MKL_NUM_THREADS'] = str(num_cpu_cores)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.custom_environment import AgentOptimizerEnvironment, SimulationParameters
from src.algorithms.mappo.agent import MAPPOAgent
from src.algorithms.mappo.trainer import MAPPOTrainer
from src.core.display import print_colored
from src.utils.duration_fitting import load_fitted_distributions


def load_config(config_path):
    """Load configuration from YAML file with defaults."""
    # Default config
    default_config = {
        'training': {
            'episodes': 50,
            'policy_update_epochs': 5,
            'online_training': False
        },
        'network': {
            'actor_hidden_size': 128,
            'critic_hidden_size': 256,
            'dropout_rate': 0.2,
            'weight_init': 'xavier_uniform'
        },
        'learning': {
            'lr_actor': 0.0003,
            'lr_critic': 0.0003
        },
        'ppo': {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_param': 0.2,
            'batch_size': 32768,
            'num_epochs': 10
        },
        'evaluation': {
            'episodes': 3,
            'deterministic': True
        },
        'device': {
            'use_gpu': True
        }
    }

    # Load user config
    if config_path:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)

        # Merge configs (user config overrides defaults)
        def deep_merge(default, user):
            if isinstance(default, dict) and isinstance(user, dict):
                for key in user:
                    if key in default:
                        default[key] = deep_merge(default[key], user[key])
                    else:
                        default[key] = user[key]
            else:
                return user
            return default

        config = deep_merge(default_config, user_config)
    else:
        config = default_config

    return config


def get_device(use_gpu=True):
    """Get compute device."""
    if not use_gpu:
        return torch.device("cpu")

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


def train(data_file, distributions_file, config_path=None, output_dir=None, name=None):
    """
    Train MAPPO agent.

    Args:
        data_file: Path to preprocessed data
        distributions_file: Path to fitted distributions
        config_path: Path to config YAML file
        output_dir: Output directory for experiments
        name: Experiment name
    """
    print_colored("="*60, "yellow")
    print_colored("PHASE 3: MAPPO TRAINING", "yellow")
    print_colored("="*60, "yellow")

    # Load configuration
    print_colored("\n1. Loading configuration...", "blue")
    config = load_config(config_path)
    if config_path:
        print_colored(f"   Loaded config from: {config_path}", "green")
    else:
        print_colored(f"   Using default configuration", "green")

    # Print key hyperparameters
    print_colored("\n   Key Hyperparameters:", "cyan")
    print_colored(f"   - Training episodes: {config['training']['episodes']}", "white")
    print_colored(f"   - Actor hidden size: {config['network']['actor_hidden_size']}", "white")
    print_colored(f"   - Learning rate (actor): {config['learning']['lr_actor']}", "white")
    print_colored(f"   - PPO clip param: {config['ppo']['clip_param']}", "white")
    print_colored(f"   - Batch size: {config['ppo']['batch_size']}", "white")

    # Load preprocessed data
    print_colored(f"\n2. Loading preprocessed data from {data_file}...", "blue")
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    train_data = data_dict['train']
    test_data = data_dict['test']
    print_colored(f"   Train set: {len(train_data)} events", "green")
    print_colored(f"   Test set: {len(test_data)} events", "green")

    # Load fitted distributions
    print_colored(f"\n3. Loading fitted distributions from {distributions_file}...", "blue")
    fitted_distributions = load_fitted_distributions(distributions_file)
    print_colored(f"   Distributions loaded successfully", "green")

    # Create experiment directory
    if output_dir is None:
        output_dir = "./experiments"

    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"mappo_{timestamp}"

    experiment_dir = os.path.join(output_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)

    print_colored(f"\n4. Experiment directory: {experiment_dir}", "blue")

    # Save configuration for reproducibility
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print_colored(f"   Saved config to: {config_save_path}", "green")

    # Initialize environment
    print_colored("\n5. Initializing training environment...", "blue")
    simulation_parameters = SimulationParameters(
        {"start_timestamp": train_data["start_timestamp"].min()}
    )

    train_env = AgentOptimizerEnvironment(
        train_data,
        simulation_parameters,
        experiment_dir=experiment_dir,
        pre_fitted_distributions=fitted_distributions,
    )
    print_colored(f"   Environment initialized with {len(train_env.agents)} agents", "green")

    # Get device
    device = get_device(config['device']['use_gpu'])
    print_colored(f"\n6. Using device: {device}", "blue")

    # Initialize MAPPO agent
    print_colored("\n7. Initializing MAPPO agent...", "blue")
    mappo_agent = MAPPOAgent(
        train_env,
        hidden_size=config['network']['actor_hidden_size'],
        lr_actor=config['learning']['lr_actor'],
        lr_critic=config['learning']['lr_critic'],
        gamma=config['ppo']['gamma'],
        num_epochs=config['training']['policy_update_epochs'],
        gae_lambda=config['ppo']['gae_lambda'],
        clip_param=config['ppo']['clip_param'],
        batch_size=config['ppo']['batch_size'],
        dropout_rate=config['network']['dropout_rate'],
        weight_init=config['network']['weight_init'],
        device=device,
    )

    print_gpu_memory()

    # Initialize trainer
    trainer = MAPPOTrainer(
        train_env,
        mappo_agent,
        total_training_episodes=config['training']['episodes'],
        should_eval=True,
        eval_episodes=config['evaluation']['episodes'],
        experiment_dir=experiment_dir,
    )

    # Start training
    print_colored("\n8. Starting MAPPO training...", "green")
    print_colored("="*60, "yellow")

    start_time = time.time()
    episode_rewards = trainer.train()
    training_time = time.time() - start_time

    print_colored("="*60, "yellow")
    print_colored(f"\n✅ Training completed in {training_time/60:.2f} minutes!", "green")

    # Save final model
    model_path = os.path.join(experiment_dir, "final_model")
    mappo_agent.save_models(model_path)
    print_colored(f"\nFinal model saved to: {model_path}", "cyan")

    # Save training summary
    summary_path = os.path.join(experiment_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("MAPPO Training Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training Time: {training_time/60:.2f} minutes\n")
        f.write(f"Total Episodes: {config['training']['episodes']}\n")
        f.write(f"Final Model: {model_path}\n\n")

        f.write("Hyperparameters:\n")
        f.write(f"  Actor Hidden Size: {config['network']['actor_hidden_size']}\n")
        f.write(f"  Critic Hidden Size: {config['network']['critic_hidden_size']}\n")
        f.write(f"  Learning Rate (Actor): {config['learning']['lr_actor']}\n")
        f.write(f"  Learning Rate (Critic): {config['learning']['lr_critic']}\n")
        f.write(f"  PPO Clip Param: {config['ppo']['clip_param']}\n")
        f.write(f"  Gamma: {config['ppo']['gamma']}\n")
        f.write(f"  Batch Size: {config['ppo']['batch_size']}\n")
        f.write(f"  Dropout Rate: {config['network']['dropout_rate']}\n")
        f.write(f"  Weight Init: {config['network']['weight_init']}\n")

    print_colored(f"Training summary saved to: {summary_path}", "cyan")

    print_colored("\n" + "="*60, "yellow")
    print_colored("Next step: Evaluate on test set with:", "cyan")
    print_colored(f"python scripts/4_evaluate.py \\", "cyan")
    print_colored(f"  --model {model_path} \\", "cyan")
    print_colored(f"  --data {data_file} \\", "cyan")
    print_colored(f"  --distributions {distributions_file}", "cyan")

    return experiment_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Train MAPPO agent"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preprocessed data (from phase 1)"
    )
    parser.add_argument(
        "--distributions",
        type=str,
        required=True,
        help="Path to fitted distributions (from phase 2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for experiments (default: ./experiments)"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Experiment name (default: mappo_TIMESTAMP)"
    )

    args = parser.parse_args()

    train(
        data_file=args.data,
        distributions_file=args.distributions,
        config_path=args.config,
        output_dir=args.output,
        name=args.name
    )
