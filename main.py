"""
Main entry point for MARL Business Process Management experiments.

Usage:
    python main.py --config src/configs/phase_b_standard_rl/mappo_baseline.yaml
    python main.py --config mappo_baseline
    python main.py --list-configs
"""
import argparse
import sys
from pathlib import Path

# Add src/ to Python path so modules can import each other
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from configs.config_loader import (
    ExperimentConfig,
    list_configs,
    print_config_summary
)


def run_experiment(config: ExperimentConfig, summary: bool = False, resume_episodes: int = 0):
    """Run a training experiment with the given configuration.

    Args:
        config: Experiment configuration
        summary: Whether to print summary metrics after training
        resume_episodes: If > 0, resume training from last checkpoint for this many episodes
    """
    print_config_summary(config)

    # Check if hyperparameter search is enabled
    if config.get('hyperparam_search.enabled', False):
        print("\nStarting hyperparameter optimization...")
        from training.hyperparam_optimizer import run_hyperparameter_search
        run_hyperparameter_search(config)
    else:
        from training.trainer import run_training
        run_training(config, summary=summary, resume_episodes=resume_episodes)


def main():
    parser = argparse.ArgumentParser(
        description='MARL Business Process Management - Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available configurations
  python main.py --list-configs

  # Run MAPPO baseline (full path or shorthand)
  python main.py --config configs/phase_b_standard_rl/mappo_baseline.yaml
  python main.py --config mappo_baseline

  # Run MAPPO with collaboration
  python main.py --config mappo_collaboration

  # Run QMIX
  python main.py --config qmix_baseline

  # Run with summary metrics after training
  python main.py --config mappo_baseline --summary
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path or name of configuration file (e.g., configs/mappo_baseline.yaml or mappo_baseline)'
    )

    parser.add_argument(
        '--list-configs', '-l',
        action='store_true',
        help='List all available configuration files'
    )

    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Print summary metrics (cycle time, waiting time, processing time, utilization) after training'
    )

    parser.add_argument(
        '--resume',
        type=int,
        default=0,
        metavar='N',
        help='Resume training from last checkpoint for N additional episodes'
    )

    args = parser.parse_args()

    # List configs if requested
    if args.list_configs:
        configs_by_phase = list_configs()
        print("\nAvailable Configurations (Organized by Phase):")
        print("=" * 80)

        setting_labels = {
            "no_collab_no_comm": "No Collaboration, No Communication",
            "collab": "Collaboration Only",
            "comm": "Communication Only",
            "collab_comm": "Collaboration + Communication",
        }

        for setting_dir in sorted(configs_by_phase.keys()):
            if setting_dir == "_root" or setting_dir.startswith("phase_"):
                continue

            label = setting_labels.get(setting_dir, setting_dir)
            print(f"\n{label}")
            print("-" * 80)

            for config_path in configs_by_phase[setting_dir]:
                config_name = Path(config_path).stem
                shorthand = f"{setting_dir}/{config_name}"
                print(f"  • {shorthand:40} → src/configs/{config_path}")

        # Also show legacy phase configs if they exist
        for phase_dir in sorted(configs_by_phase.keys()):
            if not phase_dir.startswith("phase_"):
                continue
            print(f"\n{phase_dir} (legacy)")
            print("-" * 80)
            for config_path in configs_by_phase[phase_dir]:
                config_name = Path(config_path).stem
                print(f"  • {config_name:40} → src/configs/{config_path}")

        print("\n" + "=" * 80)
        print("Usage: python main.py --config <setting>/<method>")
        print("Example: python main.py --config collab/baseline_random")
        return

    # Validate config argument
    if not args.config:
        parser.print_help()
        print("\nERROR: Please specify a config file with --config")
        print("       Use --list-configs to see available configurations")
        sys.exit(1)

    # Resolve shorthand config names (e.g., "mappo_baseline" → full path)
    config_path = args.config
    if not Path(config_path).exists():
        # Try appending .yaml if missing
        if not config_path.endswith('.yaml'):
            config_path_yaml = config_path + '.yaml'
        else:
            config_path_yaml = config_path

        # Search in src/configs/ subdirectories
        # First try exact relative path (e.g. "collab/baseline_random.yaml")
        direct_path = Path('src/configs') / config_path_yaml
        if direct_path.exists():
            matches = [direct_path]
        else:
            # Only fallback to filename search if the input looks like a shorthand (no path separators)
            if '/' not in args.config and '\\' not in args.config:
                matches = list(Path('src/configs').rglob(Path(config_path_yaml).name))
            else:
                # Full path was given but doesn't exist
                print(f"\nERROR: Config file not found: {config_path}")
                sys.exit(1)
        if len(matches) == 1:
            config_path = str(matches[0])
            print(f"Resolved config: {config_path}")
        elif len(matches) > 1:
            print(f"\nERROR: Ambiguous config name '{args.config}'. Multiple matches found:")
            for m in matches:
                print(f"  - {m}")
            print("\nPlease specify the full path.")
            sys.exit(1)

    # Load and run experiment
    try:
        config = ExperimentConfig(config_path)
        run_experiment(config, summary=args.summary, resume_episodes=args.resume)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("       Use --list-configs to see available configurations")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nWARNING: Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
