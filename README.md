# Master Thesis — MARL for Business Process Management

Multi-Agent Reinforcement Learning for resource allocation and collaboration
in business processes. Each resource is modeled as an autonomous agent that
learns when to volunteer for tasks, with optional inter-agent collaboration
and IC3Net-style communication.

## Quick start

```bash
pip install -r requirements.txt
```

### Single experiment — `main.py`

Run one experiment from a single config file:

```bash
python main.py --config configs/bpic_2012_config.yaml
python main.py --config configs/cvs_pharmacy_config.yaml
python main.py --config configs/loan_app_config.yaml
```

Print summary metrics (cycle time, waiting time, processing time, utilisation)
after training:

```bash
python main.py --config configs/cvs_pharmacy_config.yaml --summary
```

Resume training for N additional episodes. `main.py` looks up the most recent
timestamp folder under `results/<dataset>/<experiment_name>/` and continues
from its last checkpoint:

```bash
python main.py --config configs/cvs_pharmacy_config.yaml --resume 100
```

To resume a specific run directly, use `run_experiment.py --resume` (below)
and point it at the exact run folder.

List every config the loader can discover (under `configs/` and `src/configs/`):

```bash
python main.py --list-configs
```

`--config` accepts either a full path or a shorthand name; shorthand is
resolved by searching `src/configs/` recursively.

### Batch experiments — `run_experiment.py`

Run every config in a directory as parallel subprocesses. Results land in
`results/runs/<dataset>/<run_name>/`:

```bash
# Run every config in a directory in parallel
python run_experiment.py configs/

# Custom run name
python run_experiment.py configs/ --name "reward_v2"

# Only specific configs from the directory (by filename stem)
python run_experiment.py configs/ --only bpic_2012_config cvs_pharmacy_config

# Dry-run — show what would be started, without launching anything
python run_experiment.py configs/ --dry-run

# Start and detach (don't wait for completion)
python run_experiment.py configs/ --no-wait

# Show full live output instead of the dashboard
python run_experiment.py configs/ --verbose

# Plain text output (no ANSI colors) — useful for logging to file
python run_experiment.py configs/ --plain

# Resume an existing run for N additional episodes
python run_experiment.py results/runs/cvs_pharmacy/<run_name>/ --resume 200

# List available config directories
python run_experiment.py --list
```

### Inspecting results

```bash
tensorboard --logdir results/
```

## Example configurations

Three reference configurations live in `configs/`, one per dataset:

| File | Dataset | Setup |
|------|---------|-------|
| `configs/bpic_2012_config.yaml` | BPI Challenge 2012 | MAPPO, no collab, no comm |
| `configs/cvs_pharmacy_config.yaml` | CVS Pharmacy | MAPPO + collaboration + IC3Net communication |
| `configs/loan_app_config.yaml` | Loan Application | MAPPO + IC3Net communication |

Copy and edit any of them to define a new experiment.

## Layout

| Folder | Contents |
|--------|----------|
| `configs/` | Example experiment configurations (one per dataset) |
| `src/` | Core MARL code: environment, agents, training, config loader |
| `scripts/` | Analysis, evaluation, profiling, GPU diagnostics |
| `data/` | Event-log datasets (preprocessed CSVs) |

The codebase contains three main components:

- `src/environment/` — discrete-event simulator of the business process
- `src/agents/` — MAPPO, QMIX, and baseline agents
- `src/training/` — training loop, evaluation, and Optuna hyperparameter search

## Datasets

- **BPIC 2012** — public process-mining benchmark (continuous process)
- **CVS Pharmacy** — synthetic pharmacy process with work-schedule constraints
- **Loan Application** — synthetic loan-application process with parallel tasks

Place preprocessed CSVs under `data/<dataset>/processed/` (column schema:
`case_id`, `activity_name`, `resource`, `start_timestamp`).

## Results

Single runs (via `main.py`) write to:

```
results/<experiment_name>/<timestamp>/
├── logs/             # per-episode CSV logs
├── checkpoints/      # model weights
├── tensorboard/      # TensorBoard event files
└── evaluation/       # final evaluation metrics
```

Batch runs (via `run_experiment.py`) write to:

```
results/runs/<dataset>/<run_name>/
├── run_info.json
├── configs/          # copies of the YAMLs that were run
├── logs/             # stdout/stderr per process
└── <method>/         # one folder per method, with episodes/, logs/, final_evaluation/
```

## Visualisation

The CSV event logs produced during training can be explored interactively with
**[RETrace Studio](https://github.com/bjornkoemans/retrace-studio)** — a
companion Vue 3 viewer with timeline, heatmap, case-flow, and process-mining
statistics. RETrace Studio originated as a side-product of this project and is
now a standalone tool.

Live:
[bjornkoemans.nl/retrace-studio](https://bjornkoemans.nl/retrace-studio) ·
[retrace-studio.org](https://retrace-studio.org)
