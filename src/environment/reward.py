# Can't import CustomEnvironment because it causes circular import issues, but the selfs are of that type
import numpy as np
from .objects import Status
from ..core.env_config import debug_print_colored


def get_reward(self) -> float:
    """Return the reward for the current state.
    The reward is calculated as the negative squared difference between the actual duration
    and the historical median duration for the specific task/activity.

    R = -((actual_duration - historical_median)^2)

    The reward is 100 if there are no further cases being processed, and 0 if no task was completed.

    Returns:
        float: Reward of the current state
    """
    # If no cases are being processed, return high reward
    if not self.pending_cases and not self.future_cases:
        return 100.0

    # Get completed task by checking only the most recently completed tasks
    completed_task = self.completed_task

    # First check pending cases - these are more likely to have recently completed tasks
    for case in self.pending_cases:
        # Only check the current task if it exists
        if (
            case.current_task
            and case.current_task.status == Status.COMPLETED
            and case.current_task.completion_timestamp == self.current_time
        ):
            completed_task = case.current_task
            break

    # If no task found in pending cases, check completed cases
    if not completed_task:
        # Only check the most recently completed case
        if self.completed_cases:
            last_case = self.completed_cases[-1]
            if (
                last_case.current_task
                and last_case.current_task.status == Status.COMPLETED
                and last_case.current_task.completion_timestamp == self.current_time
            ):
                completed_task = last_case.current_task

    # If no task was completed at this timestep, return base reward
    if not completed_task:
        debug_print_colored("No task completed in this step", "yellow")
        return 0.0

    # Calculate actual task duration in seconds
    duration = completed_task.completion_timestamp - completed_task.assigned_timestamp
    actual_duration_seconds = duration.total_seconds()

    if actual_duration_seconds < 0:
        debug_print_colored(
            f"Warning: Negative duration {actual_duration_seconds} for task {completed_task.format()}",
            "red",
        )
        return 0.0

    # Get the task ID and look up historical median
    task_id = completed_task.id
    task_name = self.inv_task_dict.get(task_id, f"Unknown_Task_{task_id}")

    # Get historical median for this task
    historical_median = None

    if historical_median is None:
        debug_print_colored(
            f"Warning: No historical median found for task {task_name} (ID: {task_id})",
            "red",
        )
        # Fallback to the old reward calculation if no median available
        duration_minutes = actual_duration_seconds / 60
        reward = -duration_minutes
    else:
        # Calculate normalized reward based on relative performance vs historical median
        duration_diff = actual_duration_seconds - historical_median

        # Normalize the difference by the historical median to make it scale-invariant
        # This handles tasks with very different typical durations (e.g., 30s vs 3000s)
        if historical_median > 0:
            relative_diff = duration_diff / historical_median
        else:
            relative_diff = duration_diff  # fallback for edge case

        # Apply a bounded reward function that's more stable for learning
        # Using tanh to bound rewards between -1 and +1, scaled by performance
        # - Perfect performance (actual = median) gives reward ≈ 0
        # - Better than median (actual < median) gives positive rewards up to +1
        # - Worse than median (actual > median) gives negative rewards down to -1

        # Scale factor controls sensitivity: larger values make the function more sensitive to small differences
        scale_factor = 2.0
        reward = -np.tanh(scale_factor * relative_diff)

        debug_print_colored(
            f"Task {completed_task.format()} ({task_name}) completed: "
            f"actual={actual_duration_seconds:.2f}s, "
            f"median={historical_median:.2f}s, "
            f"diff={duration_diff:.2f}s, "
            f"rel_diff={relative_diff:.3f}, "
            f"reward={reward:.3f}",
            "green",
        )

    return reward
