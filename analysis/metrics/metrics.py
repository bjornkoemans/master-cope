import pandas as pd
from typing import Literal, Dict

# Define a type for time units
TimeUnit = Literal["seconds", "minutes", "hours"]

# Time unit conversion factors
time_units: Dict[TimeUnit, int] = {
    "seconds": 1,
    "minutes": 60,
    "hours": 3600,
}


# ==== EFFICIENCY METRICS ====
def throughput_time(
    end_time: pd.Timestamp, assigned_time: pd.Timestamp, unit: TimeUnit = "minutes"
) -> float:
    """
    Calculate the total throughput time in seconds.
    Arguments:
    - end_time: The time when the case/task ended.
    - assigned_time: The time when the case/task was assigned.
    """
    return (end_time - assigned_time).total_seconds() / time_units[unit]


def waiting_time(
    start_time: pd.Timestamp, assigned_time: pd.Timestamp, unit: TimeUnit = "minutes"
) -> float:
    """
    Calculate the waiting (queue) time in seconds.
    Arguments:
    - start_time: The time when the case/task started processing.
    - assigned_time: The time when the case/task was assigned.
    """
    return (start_time - assigned_time).total_seconds() / time_units[unit]


def processing_time(
    start_time: pd.Timestamp, end_time: pd.Timestamp, unit: TimeUnit = "minutes"
) -> float:
    """
    Calculate the actual processing time in seconds.
    Arguments:
    - start_time: The time when the case/task started processing.
    - end_time: The time when the case/task ended.
    """
    return (end_time - start_time).total_seconds() / time_units[unit]


# ==== UTILIZATION METRICS ====
def calculate_busy_time(
    end_times: pd.Series, assigned_times: pd.Series, unit: TimeUnit = "minutes"
) -> float:
    """
    Calculate the busy time in seconds.
    """
    time_differences: pd.Series = end_times - assigned_times
    busy_time: float = time_differences.sum().total_seconds() / time_units[unit]

    return busy_time


def utilization(
    end_times: pd.Series,
    starting_times: pd.Series,
    available_time: pd.Timedelta,
    unit: TimeUnit = "minutes",
) -> float:
    """
    Calculate the utilization time as fraction of available time.
    """
    busy_time = calculate_busy_time(end_times, starting_times, unit)
    utilization = busy_time / available_time.total_seconds()

    return utilization


def idle_time(
    end_times: pd.Series,
    assigned_times: pd.Series,
    available_time: pd.Timedelta,
    unit: TimeUnit = "minutes",
) -> float:
    """
    Calculate the idle time in seconds.
    """
    busy_time = calculate_busy_time(end_times, assigned_times, unit)
    idle_time = available_time.total_seconds() - busy_time

    return idle_time
