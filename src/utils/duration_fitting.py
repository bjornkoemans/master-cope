"""
Module for fitting duration distributions on training data.
This module provides functionality to fit duration distributions on training data
and reuse them during evaluation to ensure consistent behavior.
"""

import pickle
import os
from typing import Dict, Mapping, Optional, Tuple
import pandas as pd

from ..environment.data_handling import (
    compute_activity_duration_distribution_per_agent,
    compute_global_activity_medians,
)
from ..environment.duration_distribution import (
    DurationDistribution,
)


def fit_duration_distributions_on_training_data(
    training_data: pd.DataFrame,
) -> Tuple[
    Mapping[str, Mapping[str, Optional[DurationDistribution]]],
    Mapping[str, Mapping[str, Optional[Dict[str, float]]]],
    Dict[str, float],
]:
    """
    Fit duration distributions and compute statistics using training data only.

    This function fits duration distributions for each agent-activity combination
    using only the training dataset. These fitted distributions can then be used
    consistently across training and evaluation phases.

    Args:
        training_data: Pandas DataFrame containing the training event log data

    Returns:
        Tuple containing:
        - activity_durations_dict: Mapping from agent to activity to fitted DurationDistribution
        - stats_dict: Mapping from agent to activity to statistics dict
        - global_activity_medians: Dict mapping activity names to global median durations
    """
    print("Fitting duration distributions on training data...")

    # Fit distributions for each agent and activity using training data
    activity_durations_dict, stats_dict = (
        compute_activity_duration_distribution_per_agent(training_data)
    )

    # Compute global historical medians for each activity (across all agents)
    global_activity_medians = compute_global_activity_medians(training_data)

    print(
        f"Fitted distributions for {len(activity_durations_dict)} agents and "
        f"{len(set(training_data['activity_name']))} activities"
    )

    return activity_durations_dict, stats_dict, global_activity_medians


def save_fitted_distributions(
    activity_durations_dict: Mapping[str, Mapping[str, Optional[DurationDistribution]]],
    stats_dict: Mapping[str, Mapping[str, Optional[Dict[str, float]]]],
    global_activity_medians: Dict[str, float],
    save_path: str,
) -> None:
    """
    Save fitted duration distributions to disk.

    Args:
        activity_durations_dict: Fitted duration distributions
        stats_dict: Statistics for each agent-activity combination
        global_activity_medians: Global activity medians
        save_path: Path where to save the distributions
    """
    distributions_data = {
        "activity_durations_dict": activity_durations_dict,
        "stats_dict": stats_dict,
        "global_activity_medians": global_activity_medians,
    }

    save_dir = os.path.dirname(save_path)
    if save_dir:  # Only create directory if there is one
        os.makedirs(save_dir, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(distributions_data, f)

    print(f"Saved fitted distributions to: {save_path}")


def load_fitted_distributions(
    load_path: str,
) -> Tuple[
    Mapping[str, Mapping[str, Optional[DurationDistribution]]],
    Mapping[str, Mapping[str, Optional[Dict[str, float]]]],
    Dict[str, float],
]:
    """
    Load fitted duration distributions from disk.

    Args:
        load_path: Path from where to load the distributions

    Returns:
        Tuple containing the same data as fit_duration_distributions_on_training_data
    """
    with open(load_path, "rb") as f:
        distributions_data = pickle.load(f)

    return (
        distributions_data["activity_durations_dict"],
        distributions_data["stats_dict"],
        distributions_data["global_activity_medians"],
    )


def print_distribution_summary(
    activity_durations_dict: Mapping[str, Mapping[str, Optional[DurationDistribution]]],
    stats_dict: Mapping[str, Mapping[str, Optional[Dict[str, float]]]],
) -> None:
    """
    Print a summary of the fitted distributions.

    Args:
        activity_durations_dict: Fitted duration distributions
        stats_dict: Statistics for each agent-activity combination
    """
    print("\nDuration Distribution Summary:")
    print("=" * 50)

    total_distributions = 0
    distribution_types = {}

    for agent, activities in activity_durations_dict.items():
        print(f"\nAgent: {agent}")
        for activity, distribution in activities.items():
            if distribution is not None:
                total_distributions += 1
                dist_type = distribution.type.value
                distribution_types[dist_type] = distribution_types.get(dist_type, 0) + 1

                stats = stats_dict[agent][activity]
                if stats:
                    print(
                        f"  {activity}: {dist_type} (mean: {stats['mean']:.2f}s, "
                        f"std: {stats['std']:.2f}s)"
                    )
                else:
                    print(f"  {activity}: {dist_type}")

    print(f"\nTotal fitted distributions: {total_distributions}")
    print("Distribution types used:")
    for dist_type, count in distribution_types.items():
        print(f"  {dist_type}: {count}")
