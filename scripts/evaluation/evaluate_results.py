#!/usr/bin/env python3
"""
Thesis Results Reproducer - Evaluation Script
==============================================
Extracts throughput, waiting, and processing time metrics from simulation
eval logs and reproduces the statistical analysis tables from the thesis.

Supports all 5 method types:
  1. baseline_random       - Random task assignment
  2. baseline_best_median  - Best median performer assignment
  3. baseline_ground_truth - Oracle (actual event log assignments)
  4. mappo_baseline        - MAPPO reinforcement learning
  5. qmix_baseline         - QMIX reinforcement learning

Usage:
  # Compare ALL methods (baselines + mappo) for a dataset:
  python evaluate_results.py --dataset results/cvs_pharmacy

  # Compare only baselines:
  python evaluate_results.py --dataset results/cvs_pharmacy/baselines

  # Compare MAPPO variants:
  python evaluate_results.py --compare-settings results/cvs_pharmacy/mappo

  # Evaluate a specific run:
  python evaluate_results.py --run results/cvs_pharmacy/mappo/collab/20260224_145832

  # Only show descriptive stats (skip statistical tests):
  python evaluate_results.py --dataset results/cvs_pharmacy --no-stats

  # Export to CSV:
  python evaluate_results.py --dataset results/cvs_pharmacy --export results_table.csv
"""

import argparse
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Method Registry ────────────────────────────────────────────────────────

# Maps method folder names to display names (as used in the thesis)
METHOD_DISPLAY_NAMES = {
    # Baselines (nested structure: baselines/<name>)
    "random_no_collab": "Random (no collab)",
    "random_collab": "Random (collab)",
    "ground_truth": "Ground truth",
    # Baselines (flat structure: baseline_<name>)
    "baseline_random_no_collab": "Random (no collab)",
    "baseline_random_collab": "Random (collab)",
    "baseline_random": "Random",
    "baseline_best_median": "Best median",
    "baseline_ground_truth": "Ground truth",
    # MAPPO variants (nested structure: mappo/<name>)
    "no_collab_no_comm": "MAPPO",
    "collab": "MAPPO+Collab",
    "comm": "MAPPO+Comm",
    "collab_comm": "MAPPO+Collab+Comm",
    # MAPPO variants (flat structure: mappo_<name>)
    "mappo_no_collab_no_comm": "MAPPO",
    "mappo_collab": "MAPPO+Collab",
    "mappo_comm": "MAPPO+Comm",
    "mappo_collab_comm": "MAPPO+Collab+Comm",
    # Legacy names
    "mappo_baseline": "MAPPO",
    "qmix_baseline": "QMIX",
    "mappo": "MAPPO",
    "qmix": "QMIX",
}

# The canonical order for display (matching thesis tables)
METHOD_ORDER = [
    # Baselines
    "Random (no collab)",
    "Random (collab)",
    "Random",
    "Best median",
    "Ground truth",
    # MAPPO variants
    "MAPPO",
    "MAPPO+Collab",
    "MAPPO+Comm",
    "MAPPO+Collab+Comm",
    "QMIX",
]


# ─── Working-Time Calculation ──────────────────────────────────────────────
# Removes "dead time" (nights + weekends) from duration metrics so that only
# Mon-Fri 08:00-19:00 minutes are counted.  This makes method comparisons
# meaningful when work_schedule_enabled=True in the simulation.

_WORK_START = 8   # 08:00
_WORK_END   = 19  # 19:00
_WORK_DAYS  = {0, 1, 2, 3, 4}  # Monday-Friday
_WORK_DAY_MINUTES = (_WORK_END - _WORK_START) * 60  # 660 min = 11 h


def _clamp_to_work_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Advance *ts* to the next work-period start if it falls outside hours."""
    dow = ts.dayofweek
    h = ts.hour + ts.minute / 60 + ts.second / 3600
    if dow in _WORK_DAYS and _WORK_START <= h < _WORK_END:
        return ts                          # already in work hours
    # Move to next working 08:00
    if dow in _WORK_DAYS and h < _WORK_START:
        return ts.replace(hour=_WORK_START, minute=0, second=0, microsecond=0)
    # Past end-of-day or weekend → advance day-by-day
    candidate = (ts + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=_WORK_START)
    while candidate.dayofweek not in _WORK_DAYS:
        candidate += pd.Timedelta(days=1)
    return candidate


def _work_end_of_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Return 19:00 on the same calendar day as *ts*."""
    return ts.normalize() + pd.Timedelta(hours=_WORK_END)


def working_minutes(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Minutes between *start* and *end* that fall within work hours.

    Work hours: Mon-Fri 08:00-19:00 (11 h/day).
    Time outside these windows is excluded.

    Returns 0.0 if start >= end after clamping.
    """
    if pd.isna(start) or pd.isna(end):
        return np.nan

    start = _clamp_to_work_start(start)
    end_clamped = _clamp_to_work_start(end)  # end not yet in work hours → skip

    # If end is within work hours, keep it; otherwise snap to 19:00 of prev day
    dow_e = end.dayofweek
    h_e = end.hour + end.minute / 60 + end.second / 3600
    if dow_e in _WORK_DAYS and _WORK_START <= h_e <= _WORK_END:
        end_clamped = end
    elif dow_e in _WORK_DAYS and h_e > _WORK_END:
        end_clamped = _work_end_of_day(end)
    # else end_clamped already moved to next work start — but we want the
    # tail of the previous work day instead
    if end_clamped < start:
        return 0.0

    # Same calendar day?
    if start.date() == end_clamped.date():
        return max(0.0, (end_clamped - start).total_seconds() / 60)

    # Multi-day: first-day remainder + full days + last-day portion
    first_day_end = _work_end_of_day(start)
    first_day_min = max(0.0, (first_day_end - start).total_seconds() / 60)

    last_day_start = end_clamped.normalize() + pd.Timedelta(hours=_WORK_START)
    last_day_min = max(0.0, (end_clamped - last_day_start).total_seconds() / 60)

    # Count full working days strictly between start and end dates
    full_days = 0
    d = (start + pd.Timedelta(days=1)).normalize()
    end_date = end_clamped.normalize()
    while d < end_date:
        if d.dayofweek in _WORK_DAYS:
            full_days += 1
        d += pd.Timedelta(days=1)

    return first_day_min + full_days * _WORK_DAY_MINUTES + last_day_min


# Vectorized wrapper for pd.Series of timestamps
_working_minutes_vec = np.vectorize(
    lambda s, e: working_minutes(pd.Timestamp(s), pd.Timestamp(e)),
    otypes=[float],
)


# ─── Core Metric Extraction ────────────────────────────────────────────────

def extract_case_metrics(
    log_file: Path,
    use_working_time: bool = False,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract per-case throughput, waiting, and processing times from a single
    episode CSV log file.

    Metrics (all in minutes, one value per case):
      - throughput: case_completed_time - case_open_time   (end-to-end case duration)
      - waiting:    first task_started_time - case_open_time (time until work begins)
      - processing: sum of (task_completed - task_started) across tasks (actual work time)

    Returns:
        (throughput_times, waiting_times, processing_times) - lists of per-case values in minutes
    """
    if not log_file.exists():
        return [], [], []

    try:
        log_df = pd.read_csv(log_file, low_memory=False)
    except Exception as e:
        print(f"  Warning: Could not read {log_file.name}: {e}")
        return [], [], []

    if log_df.empty:
        return [], [], []

    # ── Throughput: true end-to-end case duration ──
    # Use case_open_time and case_completed_time if available (preferred)
    has_case_cols = {"case_open_time", "case_completed_time"}.issubset(set(log_df.columns))
    has_task_cols = {"task_assigned_time", "task_started_time", "task_completed_time"}.issubset(set(log_df.columns))

    if not has_task_cols and not has_case_cols:
        print(f"  Warning: {log_file.name} missing required timestamp columns")
        return [], [], []

    # Parse all timestamp columns present
    ts_cols = ["case_open_time", "case_completed_time",
               "task_assigned_time", "task_started_time", "task_completed_time"]
    for col in ts_cols:
        if col in log_df.columns:
            log_df[col] = pd.to_datetime(log_df[col], errors="coerce")

    if has_case_cols:
        # ── Case-level throughput from explicit case timestamps ──
        case_times = log_df.groupby("case_id").agg(
            case_open=("case_open_time", "first"),
            case_completed=("case_completed_time", "first"),
        )
    else:
        # Fallback: derive from task timestamps
        case_times = log_df.groupby("case_id").agg(
            case_open=("task_assigned_time", "min"),
            case_completed=("task_completed_time", "max"),
        )

    if use_working_time:
        # ── Working-time mode: only count Mon-Fri 08:00-19:00 ──
        case_times["_throughput"] = _working_minutes_vec(
            case_times["case_open"].values,
            case_times["case_completed"].values,
        )
    else:
        # ── Wall-clock mode (default) ──
        case_times["_throughput"] = (
            (case_times["case_completed"] - case_times["case_open"])
            .dt.total_seconds() / 60
        )

    # ── Waiting: time from case open until first task starts ──
    if has_task_cols:
        first_start = log_df.groupby("case_id")["task_started_time"].min()

        if use_working_time:
            case_times["_waiting"] = _working_minutes_vec(
                case_times["case_open"].values,
                first_start.values,
            )
        else:
            case_times["_waiting"] = (
                (first_start - case_times["case_open"]).dt.total_seconds() / 60
            )

        # ── Processing: sum of actual work time across tasks ──
        if use_working_time:
            log_df["_proc"] = _working_minutes_vec(
                log_df["task_started_time"].values,
                log_df["task_completed_time"].values,
            )
        else:
            log_df["_proc"] = (
                (log_df["task_completed_time"] - log_df["task_started_time"])
                .dt.total_seconds() / 60
            )
        case_times["_processing"] = log_df.groupby("case_id")["_proc"].sum()
    else:
        case_times["_waiting"] = 0.0
        case_times["_processing"] = case_times["_throughput"]

    # Filter valid (non-NaN, non-negative)
    throughput_times = case_times["_throughput"].dropna()
    throughput_times = throughput_times[throughput_times >= 0].tolist()

    waiting_times = case_times["_waiting"].dropna()
    waiting_times = waiting_times[waiting_times >= 0].tolist()

    processing_times = case_times["_processing"].dropna()
    processing_times = processing_times[processing_times >= 0].tolist()

    return throughput_times, waiting_times, processing_times


def extract_assignment_types(log_file: Path) -> Dict[str, int]:
    """
    Extract assignment type distribution from a single episode CSV log file.

    Returns:
        Dict mapping assignment_type -> count
    """
    if not log_file.exists():
        return {}

    try:
        usecols = ["task_assignment_type"]
        log_df = pd.read_csv(log_file, usecols=usecols, low_memory=False)
    except (ValueError, Exception):
        try:
            log_df = pd.read_csv(log_file, low_memory=False)
        except Exception:
            return {}

    if log_df.empty or "task_assignment_type" not in log_df.columns:
        return {}

    counts = log_df["task_assignment_type"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def compute_volunteer_rate(assignment_counts: Dict[str, int]) -> float:
    """
    Compute volunteer rate from assignment type counts.

    Volunteer types: anything containing 'volunteer'
    Fallback types: anything containing 'fallback' or 'random' (without volunteer)

    Returns:
        Volunteer rate as a percentage (0-100)
    """
    volunteer_count = sum(v for k, v in assignment_counts.items() if 'volunteer' in k.lower())
    total = sum(assignment_counts.values())
    if total == 0:
        return 0.0
    return volunteer_count / total * 100


# ─── Run Discovery & Loading ───────────────────────────────────────────────

def find_latest_run(method_dir: Path) -> Optional[Path]:
    """Find the latest timestamped run folder inside a method directory."""
    if not method_dir.exists():
        return None

    # Look for timestamp-named subdirectories (format: YYYYMMDD_HHMMSS)
    run_dirs = sorted(
        [d for d in method_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: d.name,
        reverse=True,
    )

    if run_dirs:
        return run_dirs[0]

    # If no timestamped subdirs, the method_dir itself might be the run
    return method_dir


def find_eval_logs(run_dir: Path, eval_pattern: str = "log_eval_*") -> List[Path]:
    """
    Find evaluation log CSV files in a run directory.

    Searches in these locations (in priority order):
      1. run_dir/logs/eval/           (new subdirectory structure)
      2. run_dir/logs/final_eval/     (new subdirectory structure)
      3. run_dir/logs/                (flat legacy structure)
      4. run_dir/evaluation/logs/
      5. run_dir/test_run/logs/
    """
    pattern = eval_pattern + ".csv" if not eval_pattern.endswith(".csv") else eval_pattern

    # New subdirectory structure: logs/eval/ and logs/final_eval/
    for subdir in ["eval", "final_eval"]:
        search_dir = run_dir / "logs" / subdir
        if search_dir.exists():
            files = sorted(search_dir.glob(pattern))
            if files:
                return files

    # Legacy flat structure and other locations
    search_dirs = [
        run_dir / "logs",
        run_dir / "evaluation" / "logs",
        run_dir / "test_run" / "logs",
    ]

    for search_dir in search_dirs:
        if search_dir.exists():
            files = sorted(search_dir.glob(pattern))
            if files:
                return files

    # Fallback: for baselines that might only have one or two log files
    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        # Check subdirectories first
        all_files = []
        for sub in sorted(logs_dir.iterdir()):
            if sub.is_dir():
                all_files.extend(sorted(sub.glob("log_*.csv")))
        if not all_files:
            all_files = sorted(logs_dir.glob("log_*.csv"))
        # Filter to only eval files if possible
        eval_files = [f for f in all_files if "eval" in f.name]
        if eval_files:
            return eval_files
        # If no explicit eval files, return all (baseline case)
        return all_files

    return []


def load_method_metrics(
    run_dir: Path,
    eval_pattern: str = "log_eval_*",
    use_train: bool = False,
    use_working_time: bool = False,
) -> Dict[str, List[float]]:
    """
    Load and aggregate metrics from all eval logs in a run directory.

    Args:
        use_working_time: If True, only count Mon-Fri 08:00-19:00 in durations.

    Returns:
        {"throughput": [...], "waiting": [...], "processing": [...]}
    """
    if use_train:
        # New structure: logs/train/  |  Legacy: logs/log_train_*
        train_dir = run_dir / "logs" / "train"
        if train_dir.exists():
            log_files = sorted(train_dir.glob("log_train_*.csv"))
        else:
            logs_dir = run_dir / "logs"
            log_files = sorted(logs_dir.glob("log_train_*.csv")) if logs_dir.exists() else []
    else:
        log_files = find_eval_logs(run_dir, eval_pattern)

    if not log_files:
        print(f"  Warning: No eval logs found in {run_dir}")
        return {"throughput": [], "waiting": [], "processing": []}

    all_throughput = []
    all_waiting = []
    all_processing = []
    all_assignment_types: Dict[str, int] = {}

    for lf in log_files:
        t, w, p = extract_case_metrics(lf, use_working_time=use_working_time)
        all_throughput.extend(t)
        all_waiting.extend(w)
        all_processing.extend(p)

        # Extract assignment type distribution
        at = extract_assignment_types(lf)
        for k, v in at.items():
            all_assignment_types[k] = all_assignment_types.get(k, 0) + v

    return {
        "throughput": all_throughput,
        "waiting": all_waiting,
        "processing": all_processing,
        "assignment_types": all_assignment_types,
        "volunteer_rate": compute_volunteer_rate(all_assignment_types),
    }


def _is_method_dir(d: Path) -> bool:
    """Check if a directory looks like a method run (has logs/ or episodes/)."""
    return (d / "logs").is_dir() or (d / "episodes").is_dir()


def discover_methods(dataset_dir: Path) -> Dict[str, Path]:
    """
    Auto-discover available method run directories within a dataset folder.

    Supports three structures:
      1. Flat: run_dir/mappo_collab/, run_dir/baseline_random_collab/
      2. Nested: dataset/baselines/<method>/<timestamp>, dataset/mappo/<variant>/<timestamp>
      3. Legacy: dataset/<method>/<timestamp>

    Returns:
        {"method_name": Path_to_latest_run, ...}
    """
    found = {}
    skip = {".", "old", "remote", "analysis", "plots", "logs", "configs", ".tmp_configs"}

    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in skip:
            continue

        # Flat structure: dir is itself a method run (has logs/ or episodes/)
        if _is_method_dir(child):
            found[child.name] = child
            continue

        # Check if this is a category dir (baselines/ or mappo/) with sub-methods
        sub_dirs = [d for d in child.iterdir() if d.is_dir() and not d.name.startswith(".")]
        sub_methods = [d for d in sub_dirs if not d.name[0].isdigit()]
        has_timestamp = any(d.name[0].isdigit() for d in sub_dirs)

        if sub_methods and not has_timestamp:
            # Category directory (e.g. baselines/, mappo/) — go one level deeper
            for sub in sorted(sub_methods):
                run_dir = find_latest_run(sub)
                if run_dir:
                    found[sub.name] = run_dir
        else:
            # Direct method directory with timestamp subdirs (legacy)
            run_dir = find_latest_run(child)
            if run_dir:
                found[child.name] = run_dir

    return found


# ─── Descriptive Statistics ─────────────────────────────────────────────────

def compute_descriptive_stats(values: List[float]) -> Dict[str, float]:
    """Compute Mean, Median, Std. Dev., Min, Max for a list of values."""
    if not values:
        return {"Mean": float("nan"), "Median": float("nan"), "Std. Dev.": float("nan"),
                "Min.": float("nan"), "Max.": float("nan"), "N": 0}

    arr = np.array(values)
    return {
        "Mean": np.mean(arr),
        "Median": np.median(arr),
        "Std. Dev.": np.std(arr),
        "Min.": np.min(arr),
        "Max.": np.max(arr),
        "N": len(arr),
    }


def print_descriptive_table(
    all_metrics: Dict[str, Dict[str, List[float]]],
    metric_key: str,
    metric_label: str,
):
    """Print a thesis-style descriptive statistics table for one metric."""
    print(f"\n{'='*80}")
    print(f"  {metric_label} distribution statistics (minutes)")
    print(f"{'='*80}")

    # Header
    header = f"  {'Agent':<18} {'Mean':>10} {'Median':>10} {'Std. Dev.':>10} {'Min.':>10} {'Max.':>10} {'N':>8}"
    print(header)
    print(f"  {'-'*76}")

    # Sort by canonical order
    ordered = []
    for display_name in METHOD_ORDER:
        for method_key, metrics in all_metrics.items():
            mdn = METHOD_DISPLAY_NAMES.get(method_key, method_key)
            if mdn == display_name:
                ordered.append((display_name, metrics))
                break

    # Add any remaining methods not in the canonical order
    seen = {name for name, _ in ordered}
    for method_key, metrics in all_metrics.items():
        mdn = METHOD_DISPLAY_NAMES.get(method_key, method_key)
        if mdn not in seen:
            ordered.append((mdn, metrics))

    for display_name, metrics in ordered:
        stats = compute_descriptive_stats(metrics[metric_key])
        print(f"  {display_name:<18} {stats['Mean']:>10.2f} {stats['Median']:>10.2f} "
              f"{stats['Std. Dev.']:>10.2f} {stats['Min.']:>10.2f} {stats['Max.']:>10.2f} "
              f"{int(stats['N']):>8}")


# ─── Statistical Tests ──────────────────────────────────────────────────────

def ks_normality_test(values: List[float]) -> Tuple[float, float]:
    """Kolmogorov-Smirnov normality test. Returns (statistic, p-value)."""
    from scipy.stats import kstest
    arr = np.array(values)
    if len(arr) < 3:
        return float("nan"), float("nan")
    # Standardize for KS test
    stat, p = kstest(arr, "norm", args=(np.mean(arr), np.std(arr)))
    return stat, p


def kruskal_wallis_test(groups: List[List[float]]) -> Tuple[float, float, float]:
    """
    Kruskal-Wallis H-test. Returns (H-statistic, p-value, epsilon-squared effect size).
    """
    from scipy.stats import kruskal
    valid = [np.array(g) for g in groups if len(g) > 0]
    if len(valid) < 2:
        return float("nan"), float("nan"), float("nan")

    # Guard against identical values across all groups (scipy raises ValueError)
    try:
        H, p = kruskal(*valid)
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    n_total = sum(len(g) for g in valid)
    k = len(valid)
    # Epsilon squared effect size
    eps_sq = (H - k + 1) / (n_total - k)
    return H, p, eps_sq


def mann_whitney_u_test(group_a: List[float], group_b: List[float]) -> Tuple[float, float]:
    """Mann-Whitney U test. Returns (U-statistic, p-value)."""
    from scipy.stats import mannwhitneyu
    a = np.array(group_a)
    b = np.array(group_b)
    if len(a) < 1 or len(b) < 1:
        return float("nan"), float("nan")
    try:
        U, p = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan")
    return U, p


def cliffs_delta(group_a: List[float], group_b: List[float]) -> Tuple[float, str]:
    """
    Compute Cliff's delta effect size.
    Positive delta means group_a tends to have larger values.

    Returns (delta, magnitude_label).
    """
    a = np.array(group_a)
    b = np.array(group_b)

    if len(a) == 0 or len(b) == 0:
        return float("nan"), "N/A"

    # Efficient computation using sorting
    n_a, n_b = len(a), len(b)

    # Use rank-based method for large arrays (O(n log n) instead of O(n²))
    if n_a * n_b > 1_000_000:
        # Efficient: use Mann-Whitney U statistic relationship
        # delta = (2U / (n_a * n_b)) - 1
        from scipy.stats import mannwhitneyu
        try:
            U, _ = mannwhitneyu(a, b, alternative="two-sided")
            delta_val = (2 * U) / (n_a * n_b) - 1
            abs_d = abs(delta_val)
            if abs_d < 0.147:
                mag = "Negligible"
            elif abs_d < 0.33:
                mag = "Small"
            elif abs_d < 0.474:
                mag = "Medium"
            else:
                mag = "Large"
            return delta_val, mag
        except Exception:
            pass

    # Direct computation for smaller arrays
    more = sum(np.sum(a_i > b) for a_i in a)
    less = sum(np.sum(a_i < b) for a_i in a)

    delta = (more - less) / (n_a * n_b)

    # Magnitude thresholds (Romano et al., 2006)
    abs_d = abs(delta)
    if abs_d < 0.147:
        mag = "Negligible"
    elif abs_d < 0.33:
        mag = "Small"
    elif abs_d < 0.474:
        mag = "Medium"
    else:
        mag = "Large"

    return delta, mag


def run_statistical_analysis(
    all_metrics: Dict[str, Dict[str, List[float]]],
    target_method: str = "mappo_baseline",
):
    """
    Run the full statistical analysis pipeline from the thesis:
    1. KS normality test per method per metric
    2. Kruskal-Wallis H-test across all methods per metric
    3. Mann-Whitney U-test + Cliff's delta: target vs each baseline
    """
    from scipy.stats import mannwhitneyu

    target_display = METHOD_DISPLAY_NAMES.get(target_method, target_method)
    metrics_keys = ["throughput", "waiting", "processing"]
    metric_labels = {
        "throughput": "Throughput",
        "waiting": "Waiting",
        "processing": "Processing",
    }

    # Identify the target method key
    target_key = None
    for mk in all_metrics:
        if METHOD_DISPLAY_NAMES.get(mk, mk) == target_display:
            target_key = mk
            break

    if target_key is None:
        print(f"\n  Warning: Target method '{target_method}' not found in results. "
              f"Available: {list(all_metrics.keys())}")
        # Try to find any RL method as target
        for mk in all_metrics:
            dn = METHOD_DISPLAY_NAMES.get(mk, mk)
            if dn in ("MAPPO", "QMIX"):
                target_key = mk
                target_display = dn
                print(f"  Using '{dn}' as target method instead.")
                break

    if target_key is None:
        print("  Cannot run statistical tests without a target RL method.")
        return

    # ── 1. KS Normality Tests ──
    print(f"\n{'='*80}")
    print(f"  Kolmogorov-Smirnov Normality Tests")
    print(f"{'='*80}")
    print(f"  {'Method':<20} {'Metric':<15} {'Statistic':>10} {'p-value':>12} {'Normal?':>10}")
    print(f"  {'-'*67}")

    for mk, metrics in all_metrics.items():
        dn = METHOD_DISPLAY_NAMES.get(mk, mk)
        for mkey in metrics_keys:
            if metrics[mkey]:
                stat, p = ks_normality_test(metrics[mkey])
                normal = "No" if p < 0.05 else "Yes"
                print(f"  {dn:<20} {metric_labels[mkey]:<15} {stat:>10.4f} {p:>12.6f} {normal:>10}")

    # ── 2. Kruskal-Wallis H-test ──
    print(f"\n{'='*80}")
    print(f"  Kruskal-Wallis H-test (across all methods)")
    print(f"{'='*80}")
    print(f"  {'Metric':<15} {'H':>12} {'ε²':>10} {'p-value':>12} {'Significant?':>14}")
    print(f"  {'-'*63}")

    for mkey in metrics_keys:
        groups = [metrics[mkey] for metrics in all_metrics.values() if metrics[mkey]]
        H, p, eps_sq = kruskal_wallis_test(groups)
        sig = "Yes" if p < 0.001 else ("Yes" if p < 0.05 else "No")
        print(f"  {metric_labels[mkey]:<15} {H:>12.2f} {eps_sq:>10.4f} {p:>12.6f} {sig:>14}")

    # ── 3. Mann-Whitney U-test + Cliff's delta ──
    # Determine baselines (everything except the target)
    baselines = [(mk, METHOD_DISPLAY_NAMES.get(mk, mk))
                 for mk in all_metrics if mk != target_key]

    n_comparisons = len(baselines)
    bonferroni_alpha = 0.05 / max(n_comparisons, 1)

    print(f"\n{'='*80}")
    print(f"  Mann-Whitney U-test: {target_display} vs each baseline")
    print(f"  (Bonferroni-corrected α = {bonferroni_alpha:.4f}, {n_comparisons} comparisons)")
    print(f"{'='*80}")
    cliffs_header = "Cliff's δ"
    print(f"  {'Metric':<15} {'Baseline':<20} {'Significant?':>14} {cliffs_header:>12} {'Magnitude':>12}")
    print(f"  {'-'*73}")

    for mkey in metrics_keys:
        target_data = all_metrics[target_key][mkey]
        if not target_data:
            continue

        for bk, bdn in baselines:
            baseline_data = all_metrics[bk][mkey]
            if not baseline_data:
                continue

            U, p = mann_whitney_u_test(target_data, baseline_data)
            delta, mag = cliffs_delta(target_data, baseline_data)

            sig = "Yes" if p < bonferroni_alpha else "No"

            # Note: positive delta means target has higher values
            # For throughput/waiting/processing, LOWER is better
            # So negative delta = target is better
            print(f"  {metric_labels[mkey]:<15} {bdn:<20} {sig:>14} {delta:>12.4f} {mag:>12}")


# ─── Main Entry Point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Thesis Results Reproducer - Extract and compare evaluation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare ALL methods (baselines + MAPPO variants):
  python evaluate_results.py --dataset results/cvs_pharmacy

  # Compare only baselines:
  python evaluate_results.py --dataset results/cvs_pharmacy/baselines

  # Compare MAPPO variants (between-settings):
  python evaluate_results.py --compare-settings results/cvs_pharmacy/mappo

  # Evaluate one specific run:
  python evaluate_results.py --run results/cvs_pharmacy/mappo/collab/20260224_145832

  # Skip statistical tests:
  python evaluate_results.py --dataset results/cvs_pharmacy --no-stats

  # Export descriptive stats to CSV:
  python evaluate_results.py --dataset results/cvs_pharmacy --export stats.csv
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset", "-d",
        type=str,
        help="Path to dataset results folder (e.g. results/cvs_pharmacy). "
             "Auto-discovers baselines/ and mappo/ subfolders.",
    )
    group.add_argument(
        "--run", "-r",
        type=str,
        help="Path to a specific run folder (e.g. results/cvs_pharmacy/mappo/collab/20260224_145832)",
    )
    group.add_argument(
        "--compare-settings",
        type=str,
        help="Path to a category folder (e.g. results/cvs_pharmacy/mappo). "
             "Compares all variants within that category.",
    )

    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        help="Specific methods to include (e.g. mappo_baseline baseline_random). "
             "Default: all discovered methods.",
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        default=None,
        help="Target RL method for pairwise statistical tests (default: auto-detect MAPPO or QMIX).",
    )
    parser.add_argument(
        "--eval-pattern",
        type=str,
        default="log_eval_*",
        help="Glob pattern for eval log files (default: 'log_eval_*').",
    )
    parser.add_argument(
        "--use-train",
        action="store_true",
        help="Use training logs instead of evaluation logs.",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip statistical tests, only show descriptive tables.",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export descriptive statistics to a CSV file.",
    )

    args = parser.parse_args()

    # ── Load metrics ──
    all_metrics: Dict[str, Dict[str, List[float]]] = OrderedDict()

    if args.compare_settings:
        # Between-setting comparison mode
        # Compares variants within a category (e.g. results/cvs_pharmacy/mappo)
        root_dir = Path(args.compare_settings)
        if not root_dir.exists():
            print(f"Error: Root directory not found: {root_dir}")
            sys.exit(1)

        # Discover variant subdirectories
        variants = sorted([
            d for d in root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and not d.name[0].isdigit()
        ])

        if args.methods:
            variants = [v for v in variants if v.name in args.methods]

        print(f"\nComparing variants in: {root_dir}")
        print(f"\nRun selection:")

        for variant_dir in variants:
            variant_name = variant_dir.name
            run_dir = find_latest_run(variant_dir)
            if run_dir is None:
                continue

            metrics = load_method_metrics(run_dir, args.eval_pattern, use_train=args.use_train)
            all_metrics[variant_name] = metrics

            log_source = "train" if args.use_train else "eval"
            if args.use_train:
                train_dir = run_dir / "logs" / "train"
                if train_dir.exists():
                    n_logs = len(sorted(train_dir.glob("log_train_*.csv")))
                else:
                    n_logs = len(sorted((run_dir / "logs").glob("log_train_*.csv"))) if (run_dir / "logs").exists() else 0
            else:
                n_logs = len(find_eval_logs(run_dir, args.eval_pattern))
            n_cases = len(metrics["throughput"])
            display = METHOD_DISPLAY_NAMES.get(variant_name, variant_name)
            print(f"  {display:<25} {run_dir}  ({n_logs} {log_source} logs, {n_cases} cases)")

        if not all_metrics:
            print(f"Error: No variant results found under {root_dir}")
            sys.exit(1)

    elif args.run:
        # Single run mode
        run_dir = Path(args.run)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)

        # Infer method name from path
        method_name = run_dir.parent.name if run_dir.parent.name != run_dir.name else run_dir.name
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)

        print(f"\nLoading metrics from: {run_dir}")
        metrics = load_method_metrics(run_dir, args.eval_pattern, use_train=args.use_train)
        all_metrics[method_name] = metrics
        n_logs = len(find_eval_logs(run_dir, args.eval_pattern))
        n_cases = len(metrics["throughput"])
        print(f"  {display_name}: {n_logs} log files, {n_cases} cases extracted")

    else:
        # Dataset mode - auto-discover methods
        dataset_dir = Path(args.dataset)
        if not dataset_dir.exists():
            print(f"Error: Dataset directory not found: {dataset_dir}")
            sys.exit(1)

        discovered = discover_methods(dataset_dir)

        if args.methods:
            # Filter to requested methods
            filtered = {k: v for k, v in discovered.items() if k in args.methods}
            if not filtered:
                print(f"Error: None of the requested methods found. Available: {list(discovered.keys())}")
                sys.exit(1)
            discovered = filtered

        if not discovered:
            print(f"Error: No method directories found in {dataset_dir}")
            sys.exit(1)

        print(f"\nDataset: {dataset_dir}")
        print(f"Available methods: {list(discovered.keys())}")
        if args.methods:
            print(f"Selected methods: {args.methods}")
        else:
            print(f"Selected methods: {list(discovered.keys())}")

        print(f"\nRun selection (latest timestamp per method):")
        for method_name, run_dir in discovered.items():
            display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
            log_source = "train" if args.use_train else "eval"
            metrics = load_method_metrics(run_dir, args.eval_pattern, use_train=args.use_train)
            all_metrics[method_name] = metrics

            if args.use_train:
                train_dir = run_dir / "logs" / "train"
                if train_dir.exists():
                    n_logs = len(sorted(train_dir.glob("log_train_*.csv")))
                else:
                    n_logs = len(sorted((run_dir / "logs").glob("log_train_*.csv"))) if (run_dir / "logs").exists() else 0
            else:
                n_logs = len(find_eval_logs(run_dir, args.eval_pattern))
            n_cases = len(metrics["throughput"])
            print(f"  {display_name:<20} {run_dir}  ({n_logs} {log_source} logs, {n_cases} cases)")

    # ── Print descriptive statistics tables ──
    print_descriptive_table(all_metrics, "throughput",
                           "Throughput time (t_end − t_assigned)")
    print_descriptive_table(all_metrics, "waiting",
                           "Total waiting time (t_start − t_assigned)")
    print_descriptive_table(all_metrics, "processing",
                           "Total processing time (t_end − t_start)")

    # ── Print volunteer rate / assignment type distribution ──
    print(f"\n{'='*80}")
    print(f"  Assignment type distribution & volunteer rate")
    print(f"{'='*80}")
    print(f"  {'Agent':<20} {'Vol. Rate':>10} {'Volunteer':>12} {'Fallback':>12} {'Other':>10} {'Total':>10}")
    print(f"  {'-'*74}")
    for mk in all_metrics:
        dn = METHOD_DISPLAY_NAMES.get(mk, mk)
        at = all_metrics[mk].get("assignment_types", {})
        vol_rate = all_metrics[mk].get("volunteer_rate", 0.0)
        volunteer_total = sum(v for k, v in at.items() if 'volunteer' in k.lower())
        fallback_total = sum(v for k, v in at.items() if 'fallback' in k.lower())
        other_total = sum(v for k, v in at.items()
                         if 'volunteer' not in k.lower() and 'fallback' not in k.lower())
        total = sum(at.values())
        print(f"  {dn:<20} {vol_rate:>9.1f}% {volunteer_total:>12} {fallback_total:>12} {other_total:>10} {total:>10}")
        # Print detailed breakdown
        for atype, count in sorted(at.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total else 0
            print(f"    {atype:<40} {count:>8} ({pct:>5.1f}%)")

    # ── Export to CSV if requested ──
    if args.export:
        rows = []
        for method_key, metrics in all_metrics.items():
            dn = METHOD_DISPLAY_NAMES.get(method_key, method_key)
            for metric_key, metric_label in [("throughput", "Throughput"),
                                              ("waiting", "Waiting"),
                                              ("processing", "Processing")]:
                stats = compute_descriptive_stats(metrics[metric_key])
                rows.append({
                    "Agent": dn,
                    "Metric": metric_label,
                    **stats,
                })
        export_df = pd.DataFrame(rows)
        export_df.to_csv(args.export, index=False)
        print(f"\nExported descriptive statistics to: {args.export}")

    # ── Statistical tests ──
    if not args.no_stats and len(all_metrics) >= 2:
        try:
            import scipy.stats  # noqa: F401
        except ImportError:
            print("\nWarning: scipy not installed. Skipping statistical tests.")
            print("  Install with: pip install scipy")
            return

        # Determine target method
        target = args.target
        if target is None:
            # Auto-detect: prefer MAPPO, then QMIX
            for candidate in ["mappo_baseline", "qmix_baseline", "mappo", "qmix"]:
                if candidate in all_metrics:
                    target = candidate
                    break

        if target:
            run_statistical_analysis(all_metrics, target_method=target)

            # If both MAPPO and QMIX are present, also run QMIX analysis
            if target in ("mappo_baseline", "mappo"):
                for qmix_key in ("qmix_baseline", "qmix"):
                    if qmix_key in all_metrics:
                        print(f"\n{'#'*80}")
                        print(f"  Repeating analysis with QMIX as target")
                        print(f"{'#'*80}")
                        run_statistical_analysis(all_metrics, target_method=qmix_key)
                        break
        else:
            print("\nNo RL method found as target for pairwise tests.")
            print("Use --target to specify one manually.")

    elif not args.no_stats:
        print("\nSkipping statistical tests (need at least 2 methods to compare).")

    print("\nDone.")


if __name__ == "__main__":
    main()
