"""Main training orchestrator."""
import os
from pathlib import Path
from datetime import datetime

from configs.config_loader import ExperimentConfig
from preprocessing.data_loader import load_and_preprocess_data
from environment.simulator import AgentOptimizerEnvironment


def run_training(config: ExperimentConfig, summary: bool = False, resume_episodes: int = 0):
    """Run training with the given configuration.

    Args:
        config: Experiment configuration
        summary: Whether to print summary metrics after training
        resume_episodes: If > 0, resume training from last checkpoint for this many episodes
    """
    # Set random seed
    import random
    import numpy as np
    import torch

    seed = config.get('experiment.seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Limit PyTorch CPU threads (prevents IC3Net from saturating all cores)
    torch_num_threads = config.get('gpu.torch_num_threads', None)
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))
        print(f"PyTorch CPU threads: {torch_num_threads}")

    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)
    print(f"Loading data from: {config.get('data.input_file')}")

    # Load and preprocess data
    collaboration_config = config.get('agent.collaboration', None)
    train_data, test_data = load_and_preprocess_data(
        data_path=config.get('data.input_file'),
        train_split=config.get('data.train_split', 0.8),
        min_case_length=config.get('data.min_case_length', 3),
        collaboration_config=collaboration_config,
    )

    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    # Create output directory
    input_file = Path(config.get('data.input_file'))
    data_name = input_file.stem  # e.g. "cvs_pharmacy"
    flat_output = config.get('output.flat', False)

    if flat_output:
        # Flat mode (used by run_experiment.py): save_dir/experiment_name/
        # No data_name subfolder, no timestamp subfolder
        experiment_dir = Path(config.get('output.save_dir')) / config.get('experiment.name')
    else:
        # Standard mode: save_dir/data_name/experiment_name/timestamp/
        output_dir = Path(config.get('output.save_dir')) / data_name / config.get('experiment.name')
        output_dir.mkdir(parents=True, exist_ok=True)

        if resume_episodes > 0:
            # Resume: find the latest existing timestamp directory
            import re
            ts_dirs = sorted(
                [d for d in output_dir.iterdir()
                 if d.is_dir() and re.match(r"^\d{8}_\d{6}$", d.name)],
                key=lambda d: d.name,
            )
            if ts_dirs:
                experiment_dir = ts_dirs[-1]
                print(f"  Resuming from existing run: {experiment_dir}")
            else:
                print(f"  ERROR: No existing run found in {output_dir} to resume from.")
                return
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = output_dir / timestamp

    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Results will be saved to: {experiment_dir}")

    # Save config copy and experiment notes (skip on resume to preserve originals)
    if resume_episodes == 0:
        import shutil
        if hasattr(config, 'config_path') and config.config_path and Path(config.config_path).exists():
            shutil.copy2(config.config_path, experiment_dir / "config.yaml")
            print(f"  Saved config.yaml")
        description = config.get('experiment.description', '')
        if description:
            with open(experiment_dir / "NOTES.md", "w") as f:
                f.write(f"# {config.get('experiment.name', 'Experiment')}\n\n")
                f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write(f"**Description**: {description}\n\n")
                reward_cfg = config.get('reward', {})
                if reward_cfg:
                    f.write(f"## Reward Config\n")
                    for k, v in reward_cfg.items():
                        f.write(f"- {k}: {v}\n")
            print(f"  Saved NOTES.md")

        # Save processed data to results folder
        train_data.to_csv(experiment_dir / "train_data.csv", index=False)
        test_data.to_csv(experiment_dir / "test_data.csv", index=False)
        print(f"  Saved train_data.csv ({len(train_data)} rows) and test_data.csv ({len(test_data)} rows)")
    else:
        print(f"  Resume mode: using existing data and config from {experiment_dir}")

    # Initialize environment
    print("\n" + "=" * 70)
    print("ENVIRONMENT INITIALIZATION")
    print("=" * 70)
    # Get reward configuration (configurable weights and values)
    reward_config = config.get('reward', None)

    work_schedule_enabled = config.get('environment.work_schedule_enabled', False)
    work_start_hour = config.get('environment.work_start_hour', 8)
    work_end_hour = config.get('environment.work_end_hour', 20)
    use_agent_identity = config.get('environment.use_agent_identity', True)

    # Parallel task groups (fan-out / fan-in)
    parallel_tasks_config = config.get('environment.parallel_tasks', None)
    parallel_task_groups = None
    if parallel_tasks_config:
        parallel_task_groups = [g['group'] for g in parallel_tasks_config if 'group' in g]
        if not parallel_task_groups:
            parallel_task_groups = None

    env = AgentOptimizerEnvironment(
        data=train_data,
        simulation_parameters={
            "start_timestamp": train_data["assign_timestamp"].min()
            if "assign_timestamp" in train_data.columns
            else train_data["start_timestamp"].min()
        },
        experiment_dir=str(experiment_dir),
        max_steps=config.get('environment.max_steps', 100_000),
        max_episodes=config.get('environment.max_episodes', 1000),
        reward_config=reward_config,
        work_schedule_enabled=work_schedule_enabled,
        work_start_hour=work_start_hour,
        work_end_hour=work_end_hour,
        use_agent_identity=use_agent_identity,
        parallel_task_groups=parallel_task_groups,
    )
    print(f"  Work schedule: {'enabled (Mon-Fri %02d:00-%02d:00)' % (work_start_hour, work_end_hour) if work_schedule_enabled else 'disabled (24/7)'}")

    # Save fitted distributions for reproducibility (like old system)
    import pickle
    fitted_distributions = (
        env._cached_activity_durations,
        env._cached_stats_dict,
        env.global_activity_medians,
    )
    with open(experiment_dir / "fitted_distributions.pkl", "wb") as f:
        pickle.dump(fitted_distributions, f)
    print(f"  Saved fitted_distributions.pkl")

    # Select and initialize agent
    agent_type = config.get('agent.type', 'mappo').lower()

    print("\n" + "=" * 70)
    print("AGENT INITIALIZATION")
    print("=" * 70)

    if agent_type == 'mappo':
        print("Agent type: MAPPO")
        from agents.mappo.agent import MAPPOAgent
        from agents.mappo.trainer import MAPPOTrainer

        # Get hyperparameters from config
        hyperparams = config.get('hyperparameters', {})

        # Get GPU config
        compile_models = config.get('gpu.compile_model', False)

        # Get communication config (IC3Net, etc.)
        communication_config = config.get('agent.communication', None)

        # Create MAPPO agent
        mappo_agent = MAPPOAgent(
            env=env,
            hidden_size=hyperparams.get('hidden_size', 256),
            lr_actor=hyperparams.get('learning_rate', 0.0003),
            lr_critic=hyperparams.get('learning_rate', 0.0003),
            gamma=hyperparams.get('gamma', 0.99),
            gae_lambda=hyperparams.get('gae_lambda', 0.95),
            clip_param=hyperparams.get('clip_epsilon', 0.2),
            batch_size=hyperparams.get('batch_size', 4096),
            buffer_size=hyperparams.get('buffer_size', None),
            num_epochs=hyperparams.get('n_epochs', 10),
            compile_models=compile_models,
            communication_config=communication_config,
            volunteer_threshold=hyperparams.get('volunteer_threshold', None),
            use_coma=hyperparams.get('use_coma', True),
            gate_in_ratio=hyperparams.get('gate_in_ratio', False),
            gate_entropy_coef=hyperparams.get('gate_entropy_coef', 0.5),
        )

        # Create trainer
        trainer = MAPPOTrainer(
            env=env,
            mappo_agent=mappo_agent,
            total_training_episodes=config.get('training.n_episodes', 50),
            eval_freq_episodes=config.get('evaluation.eval_freq_episodes', 1),
            save_freq_episodes=config.get('training.save_freq_episodes', 10),
            log_freq_episodes=config.get('training.log_freq_episodes', 1),
            eval_episodes=config.get('evaluation.n_eval_episodes', 10),
            should_eval=True,
            experiment_dir=str(experiment_dir),
            test_data=test_data,
            fitted_distributions=fitted_distributions,
            entropy_coef=hyperparams.get('entropy_coef', 0.05),
            entropy_coef_min=hyperparams.get('entropy_coef_min', 0.03),
            warmup_frac=hyperparams.get('warmup_frac', 0.3),
            skip_idle_steps=config.get('training.skip_idle_steps', False),
            n_final_eval_episodes=config.get('evaluation.n_final_eval_episodes', 10),
        )

    elif agent_type == 'qmix':
        print("Agent type: QMIX")
        from agents.qmix.trainer import QMIXTrainer

        trainer = QMIXTrainer(
            env=env,
            config=config,
            experiment_dir=str(experiment_dir)
        )

    elif agent_type == 'baseline':
        print("Agent type: BASELINE")
        from agents.baselines.baselines import run_baseline

        baseline_type = config.get('agent.baseline_type', 'random')
        n_episodes = config.get('evaluation.n_episodes', 100)

        run_baseline(
            env=env,
            baseline_type=baseline_type,
            experiment_dir=str(experiment_dir),
            n_episodes=n_episodes,
            test_data=test_data,
            fitted_distributions=fitted_distributions,
        )
        if summary:
            from analysis.summary import print_summary
            print_summary(str(experiment_dir))
        return

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Train (or resume) the agent
    print("\n" + "=" * 70)
    if resume_episodes > 0:
        print(f"RESUMING TRAINING (+{resume_episodes} episodes)")
    else:
        print("TRAINING")
    print("=" * 70)

    if resume_episodes > 0 and hasattr(trainer, 'resume'):
        trainer.resume(resume_episodes)
    else:
        trainer.train()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {experiment_dir}")

    if summary:
        from analysis.summary import print_summary
        print_summary(str(experiment_dir))
