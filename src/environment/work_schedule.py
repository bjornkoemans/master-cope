"""
Work schedule utilities for realistic evaluation.

During training the simulator runs 24/7 for faster learning.
During evaluation, tasks respect a work schedule (default Mon-Fri 08:00-19:00)
so that throughput times are comparable to the original event log.
"""
import pandas as pd


# Default work schedule constants (can be overridden via configure())
WORK_START_HOUR = 8   # 08:00
WORK_END_HOUR = 20    # 20:00 (CVS data shows tasks starting until 19:xx)
WORK_DAYS = {0, 1, 2, 3, 4}  # Monday=0 through Friday=4
WORK_DAY_SECONDS = (WORK_END_HOUR - WORK_START_HOUR) * 3600  # 11h = 39600s


def configure(work_start: int = 8, work_end: int = 20):
    """Set the module-level work schedule defaults."""
    global WORK_START_HOUR, WORK_END_HOUR, WORK_DAY_SECONDS
    WORK_START_HOUR = work_start
    WORK_END_HOUR = work_end
    WORK_DAY_SECONDS = (work_end - work_start) * 3600


def is_within_work_hours(
    timestamp: pd.Timestamp,
    work_start: int = WORK_START_HOUR,
    work_end: int = WORK_END_HOUR,
    work_days: set = WORK_DAYS,
) -> bool:
    """Check if a timestamp falls within working hours."""
    if timestamp.dayofweek not in work_days:
        return False
    hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
    return work_start <= hour < work_end


def next_work_start(
    timestamp: pd.Timestamp,
    work_start: int = WORK_START_HOUR,
    work_days: set = WORK_DAYS,
) -> pd.Timestamp:
    """Find the next work start time at or after the given timestamp.

    If timestamp is already within work hours, returns timestamp unchanged.
    Otherwise returns the start of the next working day.
    """
    if is_within_work_hours(timestamp, work_start, WORK_END_HOUR, work_days):
        return timestamp

    # Move to start of next day, then skip non-work days
    dt = timestamp.normalize() + pd.Timedelta(hours=work_start)

    # If we're before work_start on a work day, use today
    if (timestamp.dayofweek in work_days
            and timestamp.hour < work_start):
        return dt

    # Otherwise move to next day
    dt += pd.Timedelta(days=1)
    while dt.dayofweek not in work_days:
        dt += pd.Timedelta(days=1)

    return dt


def working_seconds_between(
    start: pd.Timestamp,
    end: pd.Timestamp,
    work_start: int = WORK_START_HOUR,
    work_end: int = WORK_END_HOUR,
    work_days: set = WORK_DAYS,
) -> float:
    """Calculate the number of working seconds between two timestamps.

    Only counts time within work hours (default Mon-Fri 08:00-19:00).
    Weekend and night hours are excluded.

    Args:
        start: Start timestamp
        end: End timestamp
        work_start: Work day start hour (default 8)
        work_end: Work day end hour (default 19)
        work_days: Set of working weekdays, 0=Mon (default {0,1,2,3,4})

    Returns:
        Number of working seconds between start and end

    Examples:
        Mon 08:00 → Mon 09:00 = 3600s
        Fri 18:00 → Mon 09:00 = 3600s + 3600s = 7200s (1h Fri + 1h Mon)
        Sat 12:00 → Mon 09:00 = 3600s (only Mon 08:00-09:00 counts)
    """
    if end <= start:
        return 0.0

    work_day_seconds = (work_end - work_start) * 3600
    total = 0.0

    # Clamp start to next work start if outside work hours
    current = next_work_start(start, work_start, work_days)

    if current >= end:
        return 0.0

    while True:
        # End of work today
        today_end = current.normalize() + pd.Timedelta(hours=work_end)

        if end <= today_end:
            # End falls within today's work hours
            total += (end - current).total_seconds()
            break
        else:
            # Consume rest of today
            total += (today_end - current).total_seconds()

            # Move to next work day
            next_day = today_end.normalize() + pd.Timedelta(days=1, hours=work_start)
            while next_day.dayofweek not in work_days:
                next_day += pd.Timedelta(days=1)
            current = next_day

            if current >= end:
                break

    return max(total, 0.0)


def adjust_completion_time(
    start: pd.Timestamp,
    duration_seconds: float,
    work_start: int = WORK_START_HOUR,
    work_end: int = WORK_END_HOUR,
    work_days: set = WORK_DAYS,
) -> pd.Timestamp:
    """Calculate task completion time respecting work hours.

    Given a start time and task duration, compute when the task would
    complete if agents only work during work hours (default Mon-Fri 08:00-19:00).

    If start falls outside work hours, the task begins at the next work start.

    Args:
        start: When the task starts (or is assigned to begin)
        duration_seconds: Task duration in seconds
        work_start: Work day start hour (default 8)
        work_end: Work day end hour (default 19)
        work_days: Set of working weekdays, 0=Mon (default {0,1,2,3,4})

    Returns:
        Timestamp when the task would complete

    Examples:
        Mon 08:00 + 3600s  → Mon 09:00  (normal, within work hours)
        Fri 18:50 + 900s   → Mon 08:05  (spans weekend)
        Mon 18:55 + 600s   → Tue 08:05  (spans overnight)
        Sat 12:00 + 3600s  → Mon 09:00  (starts outside work hours)
    """
    if duration_seconds <= 0:
        return start

    work_day_seconds = (work_end - work_start) * 3600
    remaining = duration_seconds

    # If start is outside work hours, move to next work start
    current = next_work_start(start, work_start, work_days)

    # Calculate remaining work time today
    today_end = current.normalize() + pd.Timedelta(hours=work_end)
    available_today = (today_end - current).total_seconds()

    # If the task fits within today's remaining work hours
    if remaining <= available_today:
        return current + pd.Timedelta(seconds=remaining)

    # Consume today's remaining time
    remaining -= available_today

    # Skip full work days
    full_days = int(remaining // work_day_seconds)
    remaining -= full_days * work_day_seconds

    # Move to the next work day after consuming full days
    current = today_end + pd.Timedelta(days=1)
    current = current.normalize() + pd.Timedelta(hours=work_start)
    days_added = 0
    while days_added < full_days:
        if current.dayofweek in work_days:
            days_added += 1
            if days_added < full_days:
                current += pd.Timedelta(days=1)
        else:
            current += pd.Timedelta(days=1)

    # Skip to next work day if needed (for the remaining partial day)
    while current.dayofweek not in work_days:
        current += pd.Timedelta(days=1)

    # Add remaining seconds on the final work day
    if remaining > 0:
        return current + pd.Timedelta(seconds=remaining)
    else:
        return current
