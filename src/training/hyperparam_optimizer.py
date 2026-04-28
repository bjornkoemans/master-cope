"""Hyperparameter optimization using Optuna."""
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, RandomSampler
from pathlib import Path
from datetime import datetime
import json

from configs.config_loader import ExperimentConfig
from preprocessing.data_loader import load_and_preprocess_data
from environment.simulator import AgentOptimizerEnvironment


def create_trial_config(config: ExperimentConfig, trial: optuna.Trial) -> dict:
    """Create a config dict with hyperparameters suggested by Optuna.

    Args:
        config: Base experiment configuration
        trial: Optuna trial object

    Returns:
        Dictionary with suggested hyperparameters
    """
    search_space = config['hyperparam_search']['search_space']
    trial_params = {}

    for param_name, param_config in search_space.items():
        param_type = param_config['type']

        if param_type == 'uniform':
            value = trial.suggest_float(
                param_name,
                param_config['low'],
                param_config['high']
            )
        elif param_type == 'loguniform':
            value = trial.suggest_float(
                param_name,
                param_config['low'],
                param_config['high'],
                log=True
            )
        elif param_type == 'int':
            value = trial.suggest_int(
                param_name,
                param_config['low'],
                param_config['high']
            )
        elif param_type == 'categorical':
            value = trial.suggest_categorical(
                param_name,
                param_config['choices']
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

        trial_params[param_name] = value

    return trial_params


def objective(trial: optuna.Trial, config: ExperimentConfig, train_data, output_dir: Path):
    """Objective function for Optuna optimization.

    Args:
        trial: Optuna trial
        config: Base configuration
        train_data: Training dataset
        output_dir: Directory for results

    Returns:
        Mean evaluation reward (to maximize)
    """
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*60}")

    # Get hyperparameters for this trial
    trial_params = create_trial_config(config, trial)

    # Print trial parameters
    for param, value in trial_params.items():
        print(f"  {param}: {value}")

    # Create trial-specific directory
    trial_dir = output_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save trial parameters
    with open(trial_dir / "params.json", 'w') as f:
        json.dump(trial_params, f, indent=2)

    # Initialize environment
    env = AgentOptimizerEnvironment(
        data=train_data,
        simulation_parameters={
            "start_timestamp": train_data["assign_timestamp"].min()
            if "assign_timestamp" in train_data.columns
            else train_data["start_timestamp"].min()
        },
        experiment_dir=str(trial_dir),
        max_steps=config.get('environment.max_steps', 100_000),
        max_episodes=config.get('environment.max_episodes', 1000),
    )

    # Select agent type
    agent_type = config.get('agent.type', 'mappo').lower()

    try:
        if agent_type == 'mappo':
            from agents.mappo.trainer import MAPPOTrainer

            # Override config hyperparameters with trial suggestions
            config_dict = dict(config.config)  # Make a copy
            config_dict['hyperparameters'] = trial_params

            # Create temporary config
            from types import SimpleNamespace
            trial_config = SimpleNamespace(
                get=lambda k, d=None: config_dict.get(k, d),
                __getitem__=lambda k: config_dict[k],
                config=config_dict
            )

            trainer = MAPPOTrainer(
                env=env,
                config=trial_config,
                experiment_dir=str(trial_dir)
            )

        elif agent_type == 'qmix':
            from agents.qmix.trainer import QMIXTrainer

            config_dict = dict(config.config)
            config_dict['hyperparameters'] = trial_params

            from types import SimpleNamespace
            trial_config = SimpleNamespace(
                get=lambda k, d=None: config_dict.get(k, d),
                __getitem__=lambda k: config_dict[k],
                config=config_dict
            )

            trainer = QMIXTrainer(
                env=env,
                config=trial_config,
                experiment_dir=str(trial_dir)
            )

        else:
            raise ValueError(f"Hyperparameter search not supported for agent type: {agent_type}")

        # Train for shorter time (as specified in config)
        total_timesteps = config.get('training.total_timesteps', 500000)
        eval_frequency = config.get('evaluation.eval_frequency', 10000)
        n_eval_episodes = config.get('evaluation.n_eval_episodes', 5)

        # Train and get final evaluation reward
        mean_reward = trainer.train(
            total_timesteps=total_timesteps,
            eval_frequency=eval_frequency,
            n_eval_episodes=n_eval_episodes
        )

        print(f"\nTrial {trial.number} completed: Mean reward = {mean_reward:.2f}")

        return mean_reward

    except Exception as e:
        print(f"\nTrial {trial.number} failed: {e}")
        # Return a bad score so this trial is pruned
        return float('-inf')


def run_hyperparameter_search(config: ExperimentConfig):
    """Run hyperparameter optimization.

    Args:
        config: Experiment configuration with hyperparam_search settings
    """
    print("\nStarting Hyperparameter Optimization")
    print("=" * 60)

    # Load data once
    print(f"Loading data from: {config.get('data.input_file')}")
    train_data, test_data = load_and_preprocess_data(
        data_path=config.get('data.input_file'),
        train_split=config.get('data.train_split', 0.8),
        min_case_length=config.get('data.min_case_length', 3)
    )

    print(f"   Training samples: {len(train_data)}")

    # Create output directory
    output_dir = Path(config.get('output.save_dir')) / config.get('experiment.name')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = output_dir / timestamp
    study_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults will be saved to: {study_dir}")

    # Create Optuna study
    sampler_type = config.get('hyperparam_search.sampler', 'TPE')
    if sampler_type == 'TPE':
        sampler = TPESampler(seed=config.get('experiment.seed', 42))
    elif sampler_type == 'Random':
        sampler = RandomSampler(seed=config.get('experiment.seed', 42))
    else:
        sampler = TPESampler(seed=config.get('experiment.seed', 42))

    pruner = MedianPruner()

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=config.get('experiment.name')
    )

    # Run optimization
    n_trials = config.get('hyperparam_search.n_trials', 50)
    n_jobs = config.get('hyperparam_search.n_jobs', 1)

    print(f"\nRunning {n_trials} trials with {n_jobs} parallel jobs")

    study.optimize(
        lambda trial: objective(trial, config, train_data, study_dir),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    # Print results
    print("\n" + "=" * 60)
    print("Hyperparameter Search Completed.")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"   Best reward: {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for param, value in study.best_params.items():
        print(f"   {param}: {value}")

    # Save results
    results_file = study_dir / "best_params.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_trial': study.best_trial.number,
            'best_value': study.best_value,
            'best_params': study.best_params
        }, f, indent=2)

    print(f"\nBest parameters saved to: {results_file}")

    # Create a new config file with best parameters
    best_config_file = study_dir / "best_config.yaml"
    import yaml

    best_config = dict(config.config)
    best_config['hyperparameters'] = study.best_params

    with open(best_config_file, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)

    print(f"Best config saved to: {best_config_file}")
    print(f"\n   You can now train with this config:")
    print(f"   python main.py --config {best_config_file}")
