"""Configuration loader for experiment configs."""
import yaml
from pathlib import Path
from typing import Dict, Any


class ExperimentConfig:
    """Loads and manages experiment configurations."""

    def __init__(self, config_path: str):
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key (supports nested keys with dots).

        Args:
            key: Config key (e.g., 'agent.type' or 'hyperparameters.learning_rate')
            default: Default value if key not found

        Returns:
            Config value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Get config section."""
        return self.config[key]

    def __repr__(self) -> str:
        return f"ExperimentConfig({self.config_path.name})"


def list_configs(config_dir: str = "src/configs") -> dict[str, list[str]]:
    """List all available configuration files organized by phase.

    Args:
        config_dir: Directory containing config files

    Returns:
        Dictionary mapping phase names to list of config file paths
    """
    config_path = Path(config_dir)
    if not config_path.exists():
        return {}

    # Find all YAML files recursively
    configs_by_phase = {}

    # Search in phase subdirectories
    for phase_dir in sorted(config_path.glob("phase_*")):
        if phase_dir.is_dir():
            phase_name = phase_dir.name
            configs = [str(f.relative_to(config_path)) for f in phase_dir.glob("*.yaml")]
            if configs:
                configs_by_phase[phase_name] = sorted(configs)

    # Also check for any configs in the root configs directory
    root_configs = [f.name for f in config_path.glob("*.yaml")]
    if root_configs:
        configs_by_phase["_root"] = sorted(root_configs)

    return configs_by_phase


def print_config_summary(config: ExperimentConfig):
    """Print a summary of the experiment configuration.

    Args:
        config: Experiment configuration
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"Name:          {config.get('experiment.name')}")
    print(f"Description:   {config.get('experiment.description')}")
    print(f"Agent Type:    {config.get('agent.type').upper()}")
    print(f"Collaboration: {'ENABLED' if config.get('agent.collaboration.enabled') else 'DISABLED'}")
    print(f"Communication: {'ENABLED' if config.get('agent.communication.enabled') else 'DISABLED'}")

    if config.get('agent.communication.enabled'):
        print(f"  Communication Type: {config.get('agent.communication.type')}")

    if config.get('hyperparam_search.enabled'):
        print(f"Hyperparameter Search: ENABLED ({config.get('hyperparam_search.n_trials')} trials)")

    print(f"Output Directory: {config.get('output.save_dir')}")
    print("=" * 70)
