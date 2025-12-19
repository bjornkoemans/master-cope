from typing import Optional, Tuple, Dict, List, Mapping
import numpy as np
import pandas as pd
from .duration_distribution import DurationDistribution, get_best_fitting_distribution


def compute_agent_activity_durations(
    data: pd.DataFrame,
) -> dict[str, dict[str, List[float]]]:
    """Collects the activity durations for each agent in the event log.
    This function is an adaptation from the original code (_compute_activity_duration_distribution) made for the AgentSim paper.

    Args:
        data (pd.DataFrame): The standardized event log data

    Returns:
        dict: A dictionary with activity durations for each agent
    """
    activities = sorted(set(data["activity_name"]))
    agents = sorted(set(data["resource"]))
    activity_durations = {key: {k: [] for k in activities} for key in agents}

    for _, row in data.iterrows():
        agent = row["resource"]
        activity = row["activity_name"]
        duration = (row["end_timestamp"] - row["start_timestamp"]).total_seconds()

        # Check if the duration is a valid number
        if pd.isna(duration) or not isinstance(duration, (int, float)):
            raise ValueError(
                f"Invalid duration for agent {agent} and activity {activity}"
            )
        activity_durations[agent][activity].append(duration)

    return activity_durations


def sample_normal(mean: float, std: float, min: float, max: float) -> float:
    """Sample from a normal distribution with given mean and standard deviation.

    Args:
        mean (float): Mean of the normal distribution
        std (float): Standard deviation of the normal distribution
        min (float): Minimum value for the sample
        max (float): Maximum value for the sample

    Returns:
        np.ndarray: Array of samples from the normal distribution
    """
    value = np.random.normal(loc=mean, scale=std)
    if value < min:
        return min
    if value > max:
        return max
    return value


def compute_activity_duration_distribution_per_agent(
    data: pd.DataFrame,
) -> Tuple[
    Mapping[str, Mapping[str, Optional[DurationDistribution]]],
    Mapping[str, Mapping[str, Optional[Dict[str, float]]]],
]:
    """
    Compute the best fitting distribution of activity durations per agent.

    Args:
        data: Event log in pandas format

    Returns:
        Tuple containing:
        - A dict storing for each agent the best fitting distribution for each activity
        - A dict storing for each agent the statistics for each activity
    """
    activity_durations_dict = compute_agent_activity_durations(data)

    agents = activity_durations_dict.keys()
    activities = sorted(set(data["activity_name"]))

    activity_duration_distribution_per_agent: Dict[
        str, Dict[str, Optional[DurationDistribution]]
    ] = {agent: {activity: None for activity in activities} for agent in agents}
    stats_dict: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {
        agent: {activity: None for activity in activities} for agent in agents
    }

    for agent, val in activity_durations_dict.items():
        for act, duration_list in val.items():
            if len(duration_list) > 0:
                # Check if there are any negative values in the duration list
                if any(d < 0 for d in duration_list):
                    raise ValueError(
                        f"Negative duration found for agent {agent} and activity {act}"
                    )

                # Calculate statistics
                mean = float(np.mean(duration_list))
                median = float(np.median(duration_list))
                std = float(np.std(duration_list))
                min_val = float(np.min(duration_list))
                max_val = float(np.max(duration_list))

                # Store statistics
                stats_dict[agent][act] = {
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "min": min_val,
                    "max": max_val,
                }

                # Fit best distribution
                try:
                    # Check for edge cases before fitting
                    if std == 0 or len(set(duration_list)) == 1:
                        # All durations are the same - use fixed distribution
                        activity_duration_distribution_per_agent[agent][act] = (
                            DurationDistribution(
                                "fix",
                                mean=mean,
                                std=0,
                                minimum=min_val,
                                maximum=max_val,
                            )
                        )
                    # elif std / mean < 0.001:  # Very low coefficient of variation
                    #     # Use normal distribution for very low variance cases
                    #     activity_duration_distribution_per_agent[agent][act] = (
                    #         DurationDistribution(
                    #             "norm",
                    #             mean=mean,
                    #             std=max(std, 0.01),
                    #             minimum=min_val,
                    #             maximum=max_val,
                    #         )
                    #     )
                    else:
                        best_distribution = get_best_fitting_distribution(
                            duration_list, filter_outliers=False
                        )
                        activity_duration_distribution_per_agent[agent][
                            act
                        ] = best_distribution
                except Exception as e:
                    print(
                        f"Warning: Could not fit distribution for agent {agent} and activity {act}: {e}"
                    )
                    # Fallback to normal distribution if fitting fails
                    # Ensure std is positive for normal distribution
                    fallback_std = max(std, 0.01) if std == 0 else std
                    activity_duration_distribution_per_agent[agent][act] = (
                        DurationDistribution(
                            "norm",
                            mean=mean,
                            std=fallback_std,
                            minimum=min_val,
                            maximum=max_val,
                        )
                    )

    return activity_duration_distribution_per_agent, stats_dict


def compute_global_activity_medians(data: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the global median duration for each activity across all agents.

    Args:
        data: Event log in pandas format

    Returns:
        Dict mapping activity names to their global median durations in seconds
    """
    global_medians = {}
    activities = sorted(set(data["activity_name"]))

    for activity in activities:
        # Get all durations for this activity across all agents
        activity_data = data[data["activity_name"] == activity]
        durations = []

        for _, row in activity_data.iterrows():
            duration = (row["end_timestamp"] - row["start_timestamp"]).total_seconds()

            # Check if the duration is valid
            if (
                pd.isna(duration)
                or not isinstance(duration, (int, float))
                or duration < 0
            ):
                continue

            durations.append(duration)

        # Compute median if we have valid durations
        if durations:
            global_medians[activity] = float(np.median(durations))
        else:
            # Fallback to 0 if no valid durations found
            global_medians[activity] = 0.0
            print(f"Warning: No valid durations found for activity {activity}")

    return global_medians
