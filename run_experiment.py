#!/usr/bin/env python3
"""
Experiment Orchestrator — Run all configs in a directory in parallel.

Point to a config directory (e.g. src/configs/cvs/) and all YAML files in it
are started as separate subprocesses. Results go to results/runs/{dataset}/{run_name}/.

Usage:
  # Run all CVS configs:
  python run_experiment.py src/configs/cvs

  # Run all BPIC configs:
  python run_experiment.py src/configs/bpic

  # Custom run name:
  python run_experiment.py src/configs/cvs --name "reward_v2"

  # Run specific YAMLs from a directory:
  python run_experiment.py src/configs/cvs --only mappo_collab_comm random_collab

  # Dry run — show what would be started:
  python run_experiment.py src/configs/cvs --dry-run

  # List available config directories:
  python run_experiment.py --list

Run directory structure:
  results/runs/{dataset}/{run_name}/
    ├── run_info.json
    ├── configs/                        # copies of original YAMLs
    ├── logs/                           # stdout/stderr per process
    ├── mappo_collab_comm/              # flat output per method
    │   ├── episodes/
    │   ├── logs/
    │   └── final_evaluation/
    ├── mappo_collab/
    └── baseline_random_collab/
"""
import argparse
import fcntl
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ANSI colors for process output prefixes
COLORS = [
    "\033[36m",   # cyan
    "\033[33m",   # yellow
    "\033[35m",   # magenta
    "\033[32m",   # green
    "\033[34m",   # blue
    "\033[91m",   # light red
    "\033[93m",   # light yellow
    "\033[96m",   # light cyan
]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[91m"
YELLOW = "\033[33m"


# ─── Dashboard: process state tracking ────────────────────────────────────────

@dataclass
class ProcessState:
    """Tracks the live state of a subprocess by parsing its stdout."""
    name: str
    color: str = ""
    phase: str = "starting"       # starting / rollout / update / eval / final_eval / done / error
    episode: int = 0
    total_episodes: int = 0
    step: int = 0
    reward: float = 0.0
    step_time: float = 0.0
    entropy: float = 0.0
    eval_reward: float = 0.0
    best_reward: float = float("-inf")
    vol_rate: float = 0.0
    avg_p_vol: float = 0.0
    start_time: float = 0.0
    finish_time: float = 0.0
    exit_code: int | None = None
    last_lines: list[str] = field(default_factory=list)
    _max_last_lines: int = 30


# Regex patterns for parsing trainer output
_RE_EPISODE     = re.compile(r"EPISODE\s+(\d+)/(\d+)\s*[-–—]\s*COLLECTING")
_RE_STEP        = re.compile(r"Step\s+(\d+)\s*\|\s*Reward:\s*([\d.eE+-]+)\s*\|\s*Time:\s*([\d.]+)")
_RE_UPDATE      = re.compile(r"POLICY UPDATE.*entropy_coef=([\d.]+)")
_RE_EVAL        = re.compile(r"EVALUATION\s*[-–—]\s*TESTING")
_RE_EVAL_REWARD = re.compile(r"Evaluation reward:\s*([\d.eE+-]+)")
_RE_BEST_MODEL  = re.compile(r"New best model.*reward:\s*([\d.eE+-]+)")
_RE_VOL_RATE    = re.compile(r"Volunteer rate:\s*([\d.]+)%")
_RE_AVG_P_VOL   = re.compile(r"Avg p\(volunteer\):\s*([\d.]+)")
_RE_DONE        = re.compile(r"Training completed after")
_RE_FINAL_EVAL  = re.compile(r"FINAL EVALUATION")
_RE_BASELINE_EP = re.compile(r"Starting.*episode\s+(\d+)/(\d+)")
_RE_ROLLOUT_DONE = re.compile(r"Rollout complete:\s*(\d+)\s*steps")


def parse_line(state: ProcessState, line: str) -> None:
    """Update ProcessState by matching a single output line against known patterns."""
    # Store last N lines for verbose fallback
    stripped = line.strip()
    if stripped and not stripped.startswith("=" * 5):
        state.last_lines.append(stripped)
        if len(state.last_lines) > state._max_last_lines:
            state.last_lines.pop(0)

    # Episode start → rollout phase
    m = _RE_EPISODE.search(line)
    if m:
        state.episode = int(m.group(1))
        state.total_episodes = int(m.group(2))
        state.phase = "rollout"
        state.step = 0
        return

    # Step progress during rollout
    m = _RE_STEP.search(line)
    if m:
        state.step = int(m.group(1))
        state.reward = float(m.group(2))
        state.step_time = float(m.group(3))
        return

    # Rollout complete (update step count to final)
    m = _RE_ROLLOUT_DONE.search(line)
    if m:
        state.step = int(m.group(1))
        return

    # Policy update phase
    m = _RE_UPDATE.search(line)
    if m:
        state.phase = "update"
        state.entropy = float(m.group(1))
        return

    # Evaluation phase
    if _RE_EVAL.search(line):
        state.phase = "eval"
        return

    # Evaluation reward
    m = _RE_EVAL_REWARD.search(line)
    if m:
        state.eval_reward = float(m.group(1))
        return

    # Best model saved
    m = _RE_BEST_MODEL.search(line)
    if m:
        state.best_reward = float(m.group(1))
        return

    # Volunteer rate
    m = _RE_VOL_RATE.search(line)
    if m:
        state.vol_rate = float(m.group(1))
        return

    # Avg p(volunteer)
    m = _RE_AVG_P_VOL.search(line)
    if m:
        state.avg_p_vol = float(m.group(1))
        return

    # Final evaluation
    if _RE_FINAL_EVAL.search(line):
        state.phase = "final_eval"
        return

    # Training completed
    if _RE_DONE.search(line):
        state.phase = "done"
        return

    # Baseline episode progress (only if no training episodes seen yet,
    # to avoid eval "Starting evaluation episode 1/1" overwriting training progress)
    m = _RE_BASELINE_EP.search(line)
    if m and state.phase in ("starting", "eval") and state.total_episodes <= 1:
        state.episode = int(m.group(1))
        state.total_episodes = int(m.group(2))
        state.phase = "eval"
        return


# ─── System monitoring ─────────────────────────────────────────────────────────

_prev_cpu_stat: dict[int, list[int]] | None = None
_prev_cpu_time: float = 0.0
_cached_cpu_usage: dict[int, float] = {}

_cached_gpu_info: dict | None = None
_gpu_cache_time: float = 0.0


def _read_proc_stat() -> dict[int, list[int]] | None:
    """Read /proc/stat per-CPU times (Linux only)."""
    try:
        with open("/proc/stat") as f:
            cpus = {}
            for line in f:
                if line.startswith("cpu") and not line.startswith("cpu "):
                    parts = line.split()
                    cpu_id = int(parts[0][3:])
                    cpus[cpu_id] = [int(x) for x in parts[1:9]]
            return cpus if cpus else None
    except (FileNotFoundError, PermissionError):
        return None


def _get_cpu_usage() -> dict[int, float]:
    """Get per-core CPU usage (0.0-1.0) between two /proc/stat samples."""
    global _prev_cpu_stat, _prev_cpu_time, _cached_cpu_usage

    curr = _read_proc_stat()
    if curr is None:
        return {}

    now = time.time()
    if _prev_cpu_stat is not None and now - _prev_cpu_time > 0.1:
        usages = {}
        for cpu_id in curr:
            if cpu_id in _prev_cpu_stat:
                prev_t = _prev_cpu_stat[cpu_id]
                curr_t = curr[cpu_id]
                delta_total = sum(curr_t) - sum(prev_t)
                delta_idle = (curr_t[3] + curr_t[4]) - (prev_t[3] + prev_t[4])
                if delta_total > 0:
                    usages[cpu_id] = 1.0 - (delta_idle / delta_total)
                else:
                    usages[cpu_id] = 0.0
        _cached_cpu_usage = usages

    _prev_cpu_stat = curr
    _prev_cpu_time = now
    return _cached_cpu_usage


def _get_gpu_info() -> dict | None:
    """Get GPU utilization and memory via nvidia-smi (cached 1s)."""
    global _cached_gpu_info, _gpu_cache_time

    now = time.time()
    if _cached_gpu_info is not None and now - _gpu_cache_time < 1.0:
        return _cached_gpu_info

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            _cached_gpu_info = {
                "util": int(parts[0]),
                "mem_used": int(parts[1]),
                "mem_total": int(parts[2]),
            }
            _gpu_cache_time = now
            return _cached_gpu_info
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


def _render_system_line() -> str:
    """Render a compact system resource line: CPU per-core + GPU."""
    parts = []

    # CPU per-core usage
    cpu = _get_cpu_usage()
    if cpu:
        n_cores = len(cpu)
        bars = []
        active = 0
        for i in sorted(cpu):
            u = cpu[i]
            if u > 0.8:
                bars.append("█")
                active += 1
            elif u > 0.4:
                bars.append("▓")
                active += 1
            elif u > 0.1:
                bars.append("▒")
            else:
                bars.append("░")
        avg = sum(cpu.values()) / n_cores * 100
        parts.append(f"CPU {''.join(bars)} {avg:.0f}% ({active}/{n_cores} busy)")

    # RAM info
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                key, val = line.split(":")
                meminfo[key.strip()] = int(val.split()[0])  # kB
            total_gb = meminfo["MemTotal"] / 1024 / 1024
            avail_gb = meminfo["MemAvailable"] / 1024 / 1024
            used_gb = total_gb - avail_gb
            pct = used_gb / total_gb * 100
            parts.append(f"RAM {used_gb:.1f}/{total_gb:.1f}G ({pct:.0f}%)")
    except (FileNotFoundError, KeyError, ValueError):
        pass

    # GPU info
    gpu = _get_gpu_info()
    if gpu:
        mem_used_g = gpu["mem_used"] / 1024
        mem_total_g = gpu["mem_total"] / 1024
        parts.append(f"GPU {gpu['util']}% | {mem_used_g:.1f}/{mem_total_g:.1f}G")

    if parts:
        return f"  {DIM}{' │ '.join(parts)}{RESET}"
    return ""


def _progress_bar(fraction: float, width: int = 24) -> str:
    """Render a progress bar like ████████░░░░░░░░."""
    filled = int(fraction * width)
    return "█" * filled + "░" * (width - filled)


def _format_time(seconds: float) -> str:
    """Format seconds as Xm Ys or Xs."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def render_dashboard(
    states: dict[str, ProcessState],
    config_dir_name: str,
    run_name: str,
    logs_dir: str,
    start_time: float,
) -> str:
    """Build the full dashboard string for terminal display."""
    elapsed = time.time() - start_time
    lines = []

    lines.append(f" {'═'*68}")
    lines.append(f"  {BOLD}{config_dir_name}{RESET} | {run_name} | Elapsed: {_format_time(elapsed)}")
    lines.append(f" {'═'*68}")
    lines.append("")

    for state in states.values():
        color = state.color
        name_padded = f"{state.name:<26}"

        if state.phase == "done":
            end = state.finish_time if state.finish_time else time.time()
            proc_elapsed = _format_time(end - state.start_time) if state.start_time else ""
            status = f"{GREEN}✓ Done{RESET}"
            if state.best_reward > float("-inf"):
                status += f"  best: {state.best_reward:.0f}"
            if proc_elapsed:
                status += f"  ({proc_elapsed})"
            lines.append(f"  {color}●{RESET} {name_padded} {status}")

        elif state.phase == "error":
            lines.append(f"  {RED}✗{RESET} {name_padded} {RED}Error (exit={state.exit_code}){RESET}")
            # Show last few lines of output for debugging
            error_lines = state.last_lines[-10:]
            for eline in error_lines:
                lines.append(f"      {DIM}{eline}{RESET}")

        elif state.phase == "starting":
            lines.append(f"  {color}●{RESET} {name_padded} {DIM}starting...{RESET}")

        else:
            # Show episode progress bar
            if state.total_episodes > 0:
                frac = state.episode / state.total_episodes
                bar = _progress_bar(frac)
                pct = f"{frac * 100:.0f}%"
                ep_str = f"ep {state.episode}/{state.total_episodes}"
                lines.append(f"  {color}●{RESET} {name_padded} {ep_str:<10} {bar} {pct}")
            else:
                lines.append(f"  {color}●{RESET} {name_padded} {state.phase}")

            # Phase detail line
            if state.phase == "rollout":
                detail = f"ROLLOUT  Step {state.step:>6}"
                if state.reward != 0:
                    detail += f" | Reward: {state.reward:>10.0f}"
                if state.step_time > 0:
                    detail += f" | {state.step_time:.1f}s"
                lines.append(f"      {DIM}{detail}{RESET}")
            elif state.phase == "update":
                lines.append(f"      {DIM}TRAINING  entropy={state.entropy:.4f}{RESET}")
            elif state.phase == "eval":
                detail = "EVALUATION"
                if state.eval_reward != 0:
                    detail += f"  reward: {state.eval_reward:.0f}"
                lines.append(f"      {DIM}{detail}{RESET}")
            elif state.phase == "final_eval":
                lines.append(f"      {DIM}FINAL EVALUATION on test data{RESET}")

            # Show volunteer metrics if available
            if state.vol_rate > 0 or state.avg_p_vol > 0:
                metrics = []
                if state.vol_rate > 0:
                    metrics.append(f"vol: {state.vol_rate:.1f}%")
                if state.avg_p_vol > 0:
                    metrics.append(f"p(vol): {state.avg_p_vol:.3f}")
                if state.best_reward > float("-inf"):
                    metrics.append(f"best: {state.best_reward:.0f}")
                lines.append(f"      {DIM}{' | '.join(metrics)}{RESET}")

        lines.append("")  # blank line between processes

    # System resources
    sys_line = _render_system_line()
    if sys_line:
        lines.append(sys_line)
        lines.append("")

    lines.append(f" {'═'*68}")
    lines.append(f"  Logs: {logs_dir}/  {DIM}(--verbose for full output){RESET}")
    lines.append(f" {'═'*68}")

    return "\n".join(lines)


def get_flat_name(yaml_path: Path) -> str:
    """Derive a flat output directory name from the YAML config.

    Reads experiment.name from the YAML. Falls back to the filename stem.
    Examples: "mappo_collab_comm", "baseline_random_collab"
    """
    try:
        text = yaml_path.read_text()
        match = re.search(r'^\s*name:\s*["\']?(.+?)["\']?\s*$', text, re.MULTILINE)
        if match:
            name = match.group(1).strip()
            # Already flat (e.g. "mappo_collab_comm") → use as-is
            if "/" not in name:
                return name
            # "mappo/collab_comm" → "mappo_collab_comm"
            # "baselines/random_collab" → "baseline_random_collab"
            return name.replace("/", "_").replace("baselines_", "baseline_")
    except Exception:
        pass
    return yaml_path.stem


def get_dataset_name(config_path: Path) -> str:
    """Extract dataset name from config file."""
    try:
        text = config_path.read_text()
        match = re.search(r'input_file:\s*["\']?(.+?)["\']?\s*$', text, re.MULTILINE)
        if match:
            # "data/cvs_pharmacy/processed/cvs_pharmacy.csv" → "cvs_pharmacy"
            return Path(match.group(1).strip()).stem
    except Exception:
        pass
    return "unknown"


def get_dataset_short(config_dir: Path) -> str:
    """Get dataset path relative to the configs directory.

    src/configs/cvs → "cvs"
    src/configs/cvs/regular → "cvs/regular"
    src/configs/bpic → "bpic"
    """
    try:
        return str(config_dir.resolve().relative_to(
            Path("src/configs").resolve()
        ))
    except ValueError:
        return config_dir.name


def create_override_config(
    original_path: Path, save_dir: str, flat_name: str, tmp_dir: Path,
) -> Path:
    """Create a temporary config copy with save_dir and experiment name overridden."""
    text = original_path.read_text()

    # Replace save_dir value
    text = re.sub(
        r'(save_dir:\s*["\']?).*?(["\']?\s*)$',
        rf'\g<1>{save_dir}\g<2>',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    # Replace experiment name with flat name
    text = re.sub(
        r'(name:\s*["\']?).*?(["\']?\s*)$',
        rf'\g<1>{flat_name}\g<2>',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    # Add flat: true to output section (after save_dir line)
    if "flat:" not in text:
        text = re.sub(
            r'(save_dir:.*$)',
            r'\g<1>\n  flat: true',
            text,
            count=1,
            flags=re.MULTILINE,
        )

    tmp_path = tmp_dir / original_path.name
    tmp_path.write_text(text)
    return tmp_path


def list_config_dirs():
    """List all available config directories under src/configs/."""
    configs_root = Path("src/configs")
    if not configs_root.exists():
        print("No src/configs/ directory found.")
        return

    print(f"\n{'='*60}")
    print(f"  AVAILABLE CONFIG DIRECTORIES")
    print(f"{'='*60}\n")

    for d in sorted(configs_root.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        yamls = sorted(d.glob("*.yaml"))
        if not yamls:
            continue
        # Get dataset from first yaml
        dataset = get_dataset_name(yamls[0])
        print(f"  {BOLD}{d}{RESET}  ({len(yamls)} configs, dataset: {dataset})")
        for y in yamls:
            flat = get_flat_name(y)
            print(f"    {DIM}•{RESET} {y.name:<40} → {flat}")
        print()

    print(f"{'='*60}")
    print(f"  Usage: python run_experiment.py src/configs/<dir>")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run all experiment configs in a directory in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config_dir",
        nargs="?",
        type=str,
        default=None,
        help="Path to config directory (e.g. src/configs/cvs)",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Run name (default: timestamp). Used as directory name under results/runs/{dataset}/",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only run these configs from the directory (by filename stem, without .yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be started without actually running",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start processes and exit immediately (don't wait for completion)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full live output instead of dashboard (old behavior)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Plain text output without ANSI colors/dashboard (for logging to file)",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        metavar="N",
        help="Resume training for N additional episodes. config_dir should point to an existing "
             "run directory (e.g. results/runs/loan_app/earth_e/20260331_183848). "
             "Only MAPPO configs (with checkpoints) are resumed; baselines are skipped.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available config directories",
    )

    args = parser.parse_args()

    # Strip ANSI codes in plain mode (before any output)
    if args.plain:
        global COLORS, RESET, BOLD, DIM, GREEN, RED, YELLOW
        COLORS = [""] * len(COLORS)
        RESET = BOLD = DIM = GREEN = RED = YELLOW = ""

    # List mode
    if args.list:
        list_config_dirs()
        return

    # Validate config_dir
    if not args.config_dir:
        parser.print_help()
        print(f"\n{BOLD}ERROR:{RESET} Please specify a config directory.")
        print(f"       Use --list to see available directories.\n")
        sys.exit(1)

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        print(f"ERROR: '{config_dir}' is not a directory.")
        sys.exit(1)

    # ─── Resume mode ─────────────────────────────────────────────────────────
    resume_mode = args.resume > 0
    resume_episodes = args.resume

    if resume_mode:
        # config_dir should be an existing run directory like:
        #   results/runs/loan_app/earth_e/20260331_183848
        source_dir = config_dir

        # Find all subdirectories that have checkpoints (= trainable configs)
        # and a config.yaml we can use to resume
        resumable = {}   # flat_name → source subdir path
        skipped = []     # flat_name (baselines without checkpoints)

        for subdir in sorted(source_dir.iterdir()):
            if not subdir.is_dir() or subdir.name in ("logs", "configs", ".tmp_configs"):
                continue
            config_yaml = subdir / "config.yaml"
            checkpoint_dir = subdir / "checkpoints" / "final"
            if not config_yaml.exists():
                continue
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                resumable[subdir.name] = subdir
            else:
                skipped.append(subdir.name)

        if not resumable:
            print(f"ERROR: No resumable configs found in '{source_dir}'.")
            print(f"  (Looking for subdirectories with checkpoints/final/)")
            if skipped:
                print(f"  Skipped (no checkpoints): {', '.join(skipped)}")
            sys.exit(1)

        # Create the resumed copy: 20260331_183848 → 20260331_183848_resumed20
        resumed_name = f"{source_dir.name}_resumed{resume_episodes}"
        run_dir = source_dir.parent / resumed_name
        logs_dir = run_dir / "logs"

        if run_dir.exists():
            print(f"ERROR: Resume directory already exists: {run_dir}")
            print(f"  Delete it first or use a different --resume value.")
            sys.exit(1)

        # Extract run info for display
        run_info_path = source_dir / "run_info.json"
        if run_info_path.exists():
            with open(run_info_path) as f:
                run_info = json.load(f)
            dataset_full = run_info.get("dataset", "unknown")
        else:
            dataset_full = "unknown"

        run_name = resumed_name

        # Read previous episode counts for display
        prev_episodes = {}
        for flat_name, subdir in resumable.items():
            state_file = subdir / "trainer_state.json"
            prev_eps = "?"
            if state_file.exists():
                try:
                    with open(state_file) as f:
                        prev_eps = json.load(f).get("episodes_done", "?")
                except Exception:
                    pass
            prev_episodes[flat_name] = prev_eps

        # Print plan
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT ORCHESTRATOR — RESUME (+{resume_episodes} episodes)")
        print(f"{'='*70}")
        print(f"  Source:      {source_dir}")
        print(f"  Copy to:     {run_dir}")
        print(f"  Dataset:     {dataset_full}")
        print(f"  Resuming:    {len(resumable)} trainable config(s)")
        if skipped:
            print(f"  Skipping:    {len(skipped)} baseline(s): {', '.join(skipped)}")
        print()
        for i, (flat_name, subdir) in enumerate(resumable.items()):
            color = COLORS[i % len(COLORS)]
            prev_eps = prev_episodes[flat_name]
            target = f"{int(prev_eps) + resume_episodes}" if isinstance(prev_eps, int) else "?"
            print(f"    {color}●{RESET} {flat_name:<30} ep {prev_eps} → {target}")
        print(f"{'='*70}\n")

        if args.dry_run:
            print("DRY RUN — nothing started.")
            return

        # ─── Copy source → resumed directory ─────────────────────────────────
        print(f"  Copying {source_dir.name} → {resumed_name}...")
        copy_start = time.time()
        shutil.copytree(source_dir, run_dir)
        copy_time = time.time() - copy_start
        print(f"  Copied in {copy_time:.1f}s")

        # Delete final_evaluation/ from each resumable subdir in the COPY
        for flat_name in resumable:
            final_eval = run_dir / flat_name / "final_evaluation"
            if final_eval.exists():
                shutil.rmtree(final_eval)
                print(f"  Removed {flat_name}/final_evaluation/")

        # Build configs/flat_names from the copied directory
        configs = {}
        flat_names = {}
        override_configs = {}
        for flat_name in resumable:
            copied_config = run_dir / flat_name / "config.yaml"
            # Update save_dir in the copied config to point to new run_dir
            config_text = copied_config.read_text()
            config_text = re.sub(
                r'(save_dir:\s*["\']?).*?(["\']?\s*)$',
                rf'\g<1>{run_dir}\g<2>',
                config_text,
                count=1,
                flags=re.MULTILINE,
            )
            copied_config.write_text(config_text)

            configs[flat_name] = copied_config
            flat_names[flat_name] = flat_name
            override_configs[flat_name] = copied_config

        # Ensure logs dir exists
        logs_dir.mkdir(parents=True, exist_ok=True)

    else:
        # ─── Normal mode ─────────────────────────────────────────────────────
        # Discover all YAML files in the directory
        all_yamls = sorted(config_dir.glob("*.yaml"))
        if not all_yamls:
            print(f"ERROR: No YAML files found in '{config_dir}'.")
            sys.exit(1)

        # Filter by --only if specified
        if args.only:
            only_set = set(args.only)
            yamls = [y for y in all_yamls if y.stem in only_set]
            missing = only_set - {y.stem for y in yamls}
            if missing:
                print(f"ERROR: Configs not found in {config_dir}: {', '.join(missing)}")
                print(f"Available: {', '.join(y.stem for y in all_yamls)}")
                sys.exit(1)
        else:
            yamls = all_yamls

        # Build config map: short_name → yaml_path
        configs = {}
        for yaml_path in yamls:
            short_name = yaml_path.stem
            configs[short_name] = yaml_path

        # Determine dataset and run directory
        dataset_short = get_dataset_short(config_dir)
        dataset_full = get_dataset_name(next(iter(configs.values())))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.name if args.name else timestamp
        run_dir = Path("results/runs") / dataset_short / run_name
        logs_dir = run_dir / "logs"

        # Compute flat names
        flat_names = {}
        for short_name, yaml_path in configs.items():
            flat_names[short_name] = get_flat_name(yaml_path)

        # Print plan
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT ORCHESTRATOR")
        print(f"{'='*70}")
        print(f"  Config dir:  {config_dir}")
        print(f"  Dataset:     {dataset_full} ({dataset_short})")
        print(f"  Run name:    {run_name}")
        print(f"  Output:      {run_dir}/")
        print(f"  Configs:     {len(configs)}")
        print()
        for i, (short_name, yaml_path) in enumerate(configs.items()):
            color = COLORS[i % len(COLORS)]
            flat = flat_names[short_name]
            print(f"    {color}●{RESET} {yaml_path.name:<40} → {flat}/")
        print(f"{'='*70}\n")

        if args.dry_run:
            print("DRY RUN — nothing started.")
            return

        # Create directories
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Save copies of original configs for reproducibility
        configs_copy_dir = run_dir / "configs"
        configs_copy_dir.mkdir(parents=True, exist_ok=True)
        for short_name, yaml_path in configs.items():
            shutil.copy2(yaml_path, configs_copy_dir / yaml_path.name)

        # Create temporary config copies with save_dir overridden to run_dir
        tmp_dir = run_dir / ".tmp_configs"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        override_configs = {}
        for short_name, yaml_path in configs.items():
            flat_name = flat_names[short_name]
            override_path = create_override_config(
                yaml_path, str(run_dir), flat_name, tmp_dir,
            )
            override_configs[short_name] = override_path

    # ─── Start processes ─────────────────────────────────────────────────────
    processes: dict[str, subprocess.Popen] = {}
    log_files: dict[str, object] = {}

    python = sys.executable

    for i, (short_name, _) in enumerate(configs.items()):
        override_path = override_configs[short_name]
        log_path = logs_dir / f"{short_name}.log"
        log_file = open(log_path, "a" if resume_mode else "w")
        log_files[short_name] = log_file

        # -u = unbuffered Python output
        cmd = [python, "-u", "main.py", "--config", str(override_path)]
        if resume_mode:
            cmd.extend(["--resume", str(resume_episodes)])

        color = COLORS[i % len(COLORS)]
        action = "Resuming" if resume_mode else "Starting"
        print(f"  {color}●{RESET} {action} {BOLD}{short_name}{RESET}... ", end="", flush=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            bufsize=0,
        )
        processes[short_name] = proc
        print(f"(PID {proc.pid})")

    # Save run metadata
    if resume_mode:
        # Update the copied run_info with resume details
        run_info_path = run_dir / "run_info.json"
        if run_info_path.exists():
            with open(run_info_path) as f:
                run_info = json.load(f)
        else:
            run_info = {}
        run_info["resumed_from"] = str(source_dir)
        run_info["resume_episodes"] = resume_episodes
        run_info["resume_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_info["status"] = "running (resumed)"
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=2)
    else:
        run_info = {
            "run_name": run_name,
            "timestamp": timestamp,
            "config_dir": str(config_dir),
            "dataset": dataset_full,
            "dataset_short": dataset_short,
            "configs": {name: str(path) for name, path in configs.items()},
            "flat_names": flat_names,
            "pids": {name: proc.pid for name, proc in processes.items()},
            "status": "running",
        }
        with open(run_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

    print(f"\n  All {len(processes)} processes started.")
    print(f"  Logs: {logs_dir}/")

    if args.no_wait:
        print(f"\n  --no-wait: Exiting. Processes continue in background.")
        print(f"  Note: Live output will stop, but logs continue writing.")
        return

    # --- Live output: dashboard (default), verbose, or plain mode ---
    plain = args.plain
    verbose = args.verbose or plain

    def signal_handler(sig, frame):
        # Restore cursor visibility and clear screen artifacts
        if not verbose:
            sys.stdout.write("\033[?25h\n")  # show cursor
            sys.stdout.flush()
        print(f"\n\n  Stopping all processes...")
        for name, proc in processes.items():
            if proc.poll() is None:
                proc.terminate()
                print(f"    Terminated {name} (PID {proc.pid})")
        time.sleep(2)
        for name, proc in processes.items():
            if proc.poll() is None:
                proc.kill()
                print(f"    Killed {name} (PID {proc.pid})")
        # Clean up tmp configs
        tmp_dir = run_dir / ".tmp_configs"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    # Build color map and fd→name map for multiplexed reading
    config_names = list(configs.keys())
    color_map = {name: COLORS[i % len(COLORS)] for i, name in enumerate(config_names)}
    fd_to_name = {}
    line_buffers = {}

    # Initialize process states for dashboard
    proc_states: dict[str, ProcessState] = {}
    for i, name in enumerate(config_names):
        proc_states[name] = ProcessState(
            name=flat_names[name],
            color=COLORS[i % len(COLORS)],
            start_time=time.time(),
        )

    for name, proc in processes.items():
        fd = proc.stdout.fileno()
        fd_to_name[fd] = name
        line_buffers[fd] = b""
        # Set non-blocking so select() works properly
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    start_time = time.time()
    completed = set()
    active_fds = set(fd_to_name.keys())
    last_render = 0.0
    config_dir_name = f"{run_dir.parent.name}/{run_dir.name}" if resume_mode else config_dir.name

    if verbose:
        print(f"\n{'='*70}")
        print(f"  LIVE OUTPUT  (Ctrl+C to stop all)")
        print(f"{'='*70}\n")
    else:
        # Hide cursor for cleaner dashboard updates
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    while len(completed) < len(processes):
        # Use select to wait for output from any process (250ms timeout for dashboard)
        timeout = 1.0 if verbose else 0.25
        if active_fds:
            try:
                readable, _, _ = select.select(list(active_fds), [], [], timeout)
            except (ValueError, OSError):
                readable = []

            for fd in readable:
                name = fd_to_name[fd]
                color = color_map[name]
                try:
                    data = os.read(fd, 8192)
                except (OSError, BlockingIOError):
                    data = b""

                if data:
                    # Write to log file
                    log_files[name].write(data.decode("utf-8", errors="replace"))
                    log_files[name].flush()

                    # Buffer and process complete lines
                    line_buffers[fd] += data
                    while b"\n" in line_buffers[fd]:
                        line, line_buffers[fd] = line_buffers[fd].split(b"\n", 1)
                        text = line.decode("utf-8", errors="replace").rstrip()
                        if text:
                            # Always parse for state tracking
                            parse_line(proc_states[name], text)
                            # In verbose mode, also print with prefix
                            if verbose:
                                tag = f"{color}[{flat_names[name]}]{RESET}"
                                print(f"  {tag} {text}")
                else:
                    # EOF — process closed stdout
                    active_fds.discard(fd)
                    # Flush remaining buffer
                    if line_buffers[fd]:
                        text = line_buffers[fd].decode("utf-8", errors="replace").rstrip()
                        if text:
                            parse_line(proc_states[name], text)
                            if verbose:
                                tag = f"{color}[{flat_names[name]}]{RESET}"
                                print(f"  {tag} {text}")
                        line_buffers[fd] = b""

        # Check for completed processes
        for name, proc in processes.items():
            if name in completed:
                continue
            ret = proc.poll()
            if ret is not None:
                completed.add(name)
                elapsed = time.time() - start_time
                color = color_map[name]
                state = proc_states[name]
                state.finish_time = time.time()
                if ret == 0:
                    state.phase = "done"
                else:
                    state.phase = "error"
                    state.exit_code = ret
                if verbose:
                    status = "✓" if ret == 0 else "✗"
                    print(f"\n  {color}{status} {BOLD}{flat_names[name]}{RESET} finished "
                          f"(exit={ret}, {_format_time(elapsed)})\n")

        # Dashboard mode: refresh display (throttled to ~4 Hz)
        if not verbose:
            now = time.time()
            if now - last_render >= 0.25:
                last_render = now
                dashboard = render_dashboard(
                    proc_states, config_dir_name, run_name,
                    str(logs_dir), start_time,
                )
                # Move cursor to top and redraw
                sys.stdout.write(f"\033[H\033[J{dashboard}\n")
                sys.stdout.flush()

    # Restore cursor after dashboard mode
    if not verbose:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        # Print final dashboard state
        dashboard = render_dashboard(
            proc_states, config_dir_name, run_name,
            str(logs_dir), start_time,
        )
        sys.stdout.write(f"\033[H\033[J{dashboard}\n\n")
        sys.stdout.flush()

    # Final status
    elapsed_total = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  ALL DONE — {elapsed_total/60:.1f} minutes total")
    print(f"{'='*70}")

    # Print error details for failed processes
    failed = {name: state for name, state in proc_states.items() if state.phase == "error"}
    if failed:
        print(f"\n{RED}{BOLD}{'='*70}")
        print(f"  FAILED PROCESSES ({len(failed)})")
        print(f"{'='*70}{RESET}\n")
        for name, state in failed.items():
            print(f"  {RED}{BOLD}✗ {name}{RESET} (exit={state.exit_code})")
            # Print last 30 lines of output
            if state.last_lines:
                print(f"  {DIM}{'─'*60}{RESET}")
                for eline in state.last_lines:
                    print(f"    {eline}")
                print(f"  {DIM}{'─'*60}{RESET}")
            # Point to log file
            log_path = logs_dir / f"{name}.log"
            if log_path.exists():
                print(f"  Full log: {log_path}")
            print()

    # Close log files
    for lf in log_files.values():
        lf.close()

    # Clean up tmp configs
    tmp_dir = run_dir / ".tmp_configs"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Update run info
    run_info["status"] = "completed"
    run_info["elapsed_minutes"] = round(elapsed_total / 60, 1)
    run_info["exit_codes"] = {name: proc.returncode for name, proc in processes.items()}

    # Find result directories
    run_info["results"] = {}
    for name in configs:
        flat_name = flat_names[name]
        result_dir = run_dir / flat_name
        if result_dir.exists():
            run_info["results"][name] = str(result_dir)

    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    # Print summary
    all_ok = all(proc.returncode == 0 for proc in processes.values())
    print(f"\n  Results: {run_dir}/")
    print(f"  Status:  {'ALL PASSED' if all_ok else 'SOME FAILED'}")

    for i, name in enumerate(configs):
        color = COLORS[i % len(COLORS)]
        rc = processes[name].returncode
        marker = "✓" if rc == 0 else "✗"
        result_path = run_info["results"].get(name, "(no result dir found)")
        print(f"    {color}{marker}{RESET} {name:<40} {result_path}")

    print()


if __name__ == "__main__":
    main()
