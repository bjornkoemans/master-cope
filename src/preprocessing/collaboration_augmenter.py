"""
Data augmentation for collaboration requirements.

This module adds collaboration metadata to event logs based on configuration rules.
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any


class CollaborationAugmenter:
    """Augments event log data with collaboration requirements."""

    def __init__(self, rules_config_path: str = "configs/collaboration_rules.yaml"):
        """
        Initialize the augmenter with collaboration rules.

        Args:
            rules_config_path: Path to YAML file with collaboration rules
        """
        self.rules_config_path = rules_config_path
        self.rules = []
        self.agent_type_mapping = {}
        self.default_agents_required = 1
        self.random_seed = 42

        self._load_rules()

    def _load_rules(self):
        """Load collaboration rules from YAML config."""
        config_path = Path(self.rules_config_path)

        if not config_path.exists():
            print(f"No collaboration rules found at {self.rules_config_path}")
            print("   Using default: all tasks require 1 agent")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load global settings
        if 'global' in config:
            self.default_agents_required = config['global'].get('default_agents_required', 1)
            self.random_seed = config['global'].get('random_seed', 42)

        # Load rules
        if 'rules' in config:
            self.rules = config['rules']
            print(f"Loaded {len(self.rules)} collaboration rules")

        # Load agent type mapping
        if 'agent_type_mapping' in config:
            self.agent_type_mapping = config['agent_type_mapping']
            print(f"Loaded agent type mapping for {len(self.agent_type_mapping)} agents")

    def get_agent_type(self, resource_name: str) -> str:
        """
        Get the type of an agent/resource.

        Args:
            resource_name: Name of the resource

        Returns:
            Agent type (e.g., 'Technician', 'Pharmacist', 'System')
        """
        return self.agent_type_mapping.get(resource_name, "Unknown")

    def find_matching_rule(self, activity: str) -> Dict[str, Any] | None:
        """
        Find a collaboration rule that matches the given activity.

        Args:
            activity: Activity name

        Returns:
            Matching rule dict or None
        """
        for rule in self.rules:
            if rule['activity'] == activity:
                return rule
        return None

    def augment_dataset(self, df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """
        Augment dataset with collaboration requirements.

        Adds columns:
        - agents_required: number of agents needed (1, 2, 3, etc.)
        - agent_types_required: comma-separated list of required agent types
        - is_collaborative: boolean indicating if task needs collaboration

        Args:
            df: Event log DataFrame with columns: case_id, activity_name, resource, etc.
            seed: Random seed for reproducibility

        Returns:
            Augmented DataFrame with collaboration metadata
        """
        if seed is None:
            seed = self.random_seed

        rng = np.random.RandomState(seed)

        print(f"\nAugmenting dataset with collaboration requirements...")
        print(f"   Random seed: {seed}")

        # Create a copy to avoid modifying original
        df_aug = df.copy()

        # Initialize new columns
        df_aug['agents_required'] = self.default_agents_required
        df_aug['agent_types_required'] = ''
        df_aug['is_collaborative'] = False

        # Add agent type column
        df_aug['agent_type'] = df_aug['resource'].map(
            lambda x: self.get_agent_type(x)
        )

        # Group by activity to apply rules
        activities = df_aug['activity_name'].unique()

        total_collaborative = 0

        for activity in activities:
            rule = self.find_matching_rule(activity)

            if rule is None:
                # No rule: default to 1 agent
                continue

            # Get all events for this activity
            activity_mask = df_aug['activity_name'] == activity
            n_events = activity_mask.sum()

            # Determine which instances need collaboration
            collab_prob = rule.get('collaboration_probability', 0.0)
            n_collaborative = int(n_events * collab_prob)

            if n_collaborative == 0:
                continue

            # Randomly select which instances need collaboration
            activity_indices = df_aug[activity_mask].index.tolist()
            collaborative_indices = rng.choice(
                activity_indices,
                size=n_collaborative,
                replace=False
            )

            # Update collaboration metadata for selected instances
            min_agents = rule.get('min_agents', 2)
            max_agents = rule.get('max_agents', 2)
            required_types = rule.get('required_agent_types', [])

            # For now, use min_agents (you can randomize between min and max if needed)
            n_agents = min_agents

            df_aug.loc[collaborative_indices, 'agents_required'] = n_agents
            df_aug.loc[collaborative_indices, 'agent_types_required'] = ','.join(required_types)
            df_aug.loc[collaborative_indices, 'is_collaborative'] = True

            total_collaborative += len(collaborative_indices)

            print(f"   • {activity}: {len(collaborative_indices)}/{n_events} "
                  f"({collab_prob*100:.0f}%) require {n_agents} agents")

        print(f"\nAugmentation Summary:")
        print(f"   Total events: {len(df_aug)}")
        print(f"   Collaborative events: {total_collaborative} "
              f"({total_collaborative/len(df_aug)*100:.1f}%)")
        print(f"   Single-agent events: {len(df_aug) - total_collaborative}")

        # Distribution of agents_required
        agents_dist = df_aug['agents_required'].value_counts().sort_index()
        print(f"\n   Distribution of agents_required:")
        for n_agents, count in agents_dist.items():
            print(f"      {n_agents} agent{'s' if n_agents > 1 else ''}: "
                  f"{count} events ({count/len(df_aug)*100:.1f}%)")

        return df_aug

    def save_augmented_data(self, df: pd.DataFrame, output_path: str):
        """
        Save augmented dataset to CSV.

        Args:
            df: Augmented DataFrame
            output_path: Output file path
        """
        df.to_csv(output_path, index=False)
        print(f"\nAugmented dataset saved to: {output_path}")


def augment_event_log(
    input_path: str,
    output_path: str,
    rules_config_path: str = "configs/collaboration_rules.yaml",
    seed: int = 42
):
    """
    Convenience function to augment an event log file.

    Args:
        input_path: Path to input CSV file
        output_path: Path to save augmented CSV
        rules_config_path: Path to collaboration rules YAML
        seed: Random seed

    Returns:
        Augmented DataFrame
    """
    print(f"Loading event log from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} events from {df['case_id'].nunique()} cases")

    augmenter = CollaborationAugmenter(rules_config_path)
    df_aug = augmenter.augment_dataset(df, seed=seed)

    augmenter.save_augmented_data(df_aug, output_path)

    return df_aug


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python collaboration_augmenter.py <input_csv> <output_csv> [rules_yaml]")
        print("\nExample:")
        print("  python preprocessing/collaboration_augmenter.py \\")
        print("         data/cvs_pharmacy/processed/cvs_pharmacy.csv \\")
        print("         data/cvs_pharmacy/processed/train_collaborative.csv \\")
        print("         configs/collaboration_rules.yaml")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    rules_path = sys.argv[3] if len(sys.argv) > 3 else "configs/collaboration_rules.yaml"

    augment_event_log(input_path, output_path, rules_path)
