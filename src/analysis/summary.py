"""Post-training summary metrics computation."""
import os
import glob
import pandas as pd


def print_summary(experiment_dir: str) -> None:
    """Compute and print summary metrics from the latest experiment log.

    Args:
        experiment_dir: Path to the experiment directory containing logs/
    """
    log_dir = os.path.join(experiment_dir, "logs")
    if not os.path.exists(log_dir):
        print(f"No logs directory found at {log_dir}")
        return

    # Find the latest log file
    log_files = sorted(glob.glob(os.path.join(log_dir, "log_*.csv")))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    log_file = log_files[-1]
    df = pd.read_csv(log_file)

    if df.empty:
        print("Log file is empty, no metrics to compute.")
        return

    # Robustly parse datetime columns (handles timezone-aware timestamps
    # that parse_dates= can silently fail on, leaving them as strings)
    datetime_cols = [
        "case_open_time", "case_completed_time",
        "task_assigned_time", "task_started_time", "task_completed_time"
    ]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='ISO8601', utc=True)

    # --- Case-level metrics ---
    cases = df.groupby("case_id").agg(
        case_open_time=("case_open_time", "first"),
        case_completed_time=("case_completed_time", "first"),
        total_waiting=("task_assigned_time", lambda x: (
            (df.loc[x.index, "task_started_time"] - df.loc[x.index, "task_assigned_time"])
            .dt.total_seconds().sum()
        )),
        total_processing=("task_assigned_time", lambda x: (
            (df.loc[x.index, "task_completed_time"] - df.loc[x.index, "task_started_time"])
            .dt.total_seconds().sum()
        )),
    )

    cases["cycle_time"] = (
        cases["case_completed_time"] - cases["case_open_time"]
    ).dt.total_seconds()

    avg_cycle = cases["cycle_time"].mean()
    avg_waiting = cases["total_waiting"].mean()
    avg_processing = cases["total_processing"].mean()

    # --- Resource utilization ---
    sim_start = df["task_assigned_time"].min()
    sim_end = df["task_completed_time"].max()
    sim_span = (sim_end - sim_start).total_seconds()

    # Handle collaborative tasks: task_agent_id can be comma-separated ("3,5")
    # Explode into individual rows for per-agent utilization
    agent_rows = []
    agent_id_to_name: dict[str, str] = {}
    for _, row in df.iterrows():
        agent_id_str = str(row["task_agent_id"])
        agent_name_str = str(row.get("task_agent_name", ""))
        task_duration = (row["task_completed_time"] - row["task_started_time"]).total_seconds()
        ids = [x.strip() for x in agent_id_str.split(",") if x.strip()]
        names = [x.strip() for x in agent_name_str.split(",") if x.strip()]
        for i, aid in enumerate(ids):
            agent_rows.append({"agent_id": aid, "busy_seconds": task_duration})
            # Map agent_id → agent_name (take the first non-empty name seen)
            if aid not in agent_id_to_name and i < len(names):
                agent_id_to_name[aid] = names[i]

    if agent_rows:
        agent_df = pd.DataFrame(agent_rows)
        agent_busy = agent_df.groupby("agent_id")["busy_seconds"].sum()
        agent_util = agent_busy / sim_span if sim_span > 0 else agent_busy * 0
    else:
        agent_util = pd.Series(dtype=float)

    # --- Print ---
    def fmt_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Log file:             {os.path.basename(log_file)}")
    print(f"  Cases completed:      {len(cases)}")
    print(f"  Total tasks logged:   {len(df)}")
    print()
    print("  Case-Level Metrics (averages)")
    print("  " + "-" * 40)
    print(f"    Cycle time:         {fmt_time(avg_cycle)}")
    print(f"    Waiting time:       {fmt_time(avg_waiting)}")
    print(f"    Processing time:    {fmt_time(avg_processing)}")
    print()
    print("  Resource Utilization")
    print("  " + "-" * 40)
    for agent_id in sorted(agent_util.index):
        name = agent_id_to_name.get(agent_id, "Unknown")
        print(f"    Agent {agent_id} ({name}): {agent_util[agent_id] * 100:>6.1f}%")
    print(f"    {'Average:':>30s} {agent_util.mean() * 100:>6.1f}%")
    print("=" * 70)
