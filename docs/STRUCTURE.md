# Repository Structuur - Gedetailleerde Uitleg

Dit document geeft een uitgebreide uitleg van wat elk bestand doet in de repository.

## 📂 `/src/` - Broncode

### `/src/core/` - Kern Configuratie

| Bestand | Verantwoordelijkheid |
|---------|---------------------|
| `config.py` | Dataset configuratie: definieer welke kolommen uit je event log gebruikt worden (case_id, activity, resource, timestamp) |
| `env_config.py` | Environment parameters en debug settings |
| `display.py` | Terminal output helpers voor colored printing en formatted lists |

### `/src/environment/` - MARL Environment

De kern van het systeem - de multi-agent reinforcement learning omgeving.

| Bestand | Verantwoordelijkheid |
|---------|---------------------|
| `custom_environment.py` | **Hoofdbestand**: PettingZoo ParallelEnv implementatie. Bevat de volledige simulatie logica: observation/action spaces, state management, episode flow. |
| `objects.py` | Datastructuren: `ResourceAgent` (agent state), `Task` (activiteit), `Case` (process instance), `Status` enum |
| `reward.py` | Reward functie berekening. Beloont agents voor snelle case completion vs historische baseline |
| `duration_distribution.py` | Modeling van task durations met verschillende distributies (Exponential, Weibull, LogNormal, Gamma, Uniform) |
| `data_handling.py` | Verwerkt event logs naar tasks/cases, berekent duration distributies per agent/activity |
| `typed_queue.py` | Type-safe queue implementatie voor task management |
| `constants.py` | Constanten zoals MAX_TASKS_PER_AGENT |

**Flow**: Event log → `data_handling.py` → `objects.py` (Tasks/Cases) → `custom_environment.py` (simulatie) → `reward.py` (feedback)

### `/src/algorithms/` - RL Algoritmes

#### `/src/algorithms/mappo/` - Multi-Agent PPO

| Bestand | Verantwoordelijkheid |
|---------|---------------------|
| `agent.py` | MAPPO agent klasse met actor-critic networks, policy updates, GAE advantage estimation |
| `trainer.py` | Training loop: rollout collectie, batch updates, episode management, checkpointing |
| `online_trainer.py` | Variant voor online learning (update policy elk timestep i.p.v. per episode) |
| `networks.py` | PyTorch neural network architectures voor actor en critic |

**Algoritme**: Proximal Policy Optimization aangepast voor multi-agent settings. Gebruikt centralized critic, decentralized actors.

#### `/src/algorithms/qmix/` - QMIX

| Bestand | Verantwoordelijkheid |
|---------|---------------------|
| `agent.py` | QMIX agent met individual Q-networks en mixing network. Combineert agent Q-values monotonisch |
| `trainer.py` | QMIX training loop met experience replay buffer en target networks |

**Algoritme**: Value-based method. Leert Q-value voor elke agent, combineert via mixing network voor team Q-value.

#### `/src/algorithms/baselines/` - Baseline Agents

| Bestand | Verantwoordelijkheid |
|---------|---------------------|
| `baselines.py` | Baseline agents voor vergelijking: <br>• **RandomAgent**: Selecteert acties random<br>• **BestMedianAgent**: Alleen snelste agent volunteert<br>• **GroundTruthAgent**: Volgt originele assignments<br>• **BaselineEvaluator**: Evalueert baseline performance |

### `/src/preprocessing/` - Data Preprocessing

| Bestand | Verantwoordelijkheid |
|---------|---------------------|
| `load_data.py` | Event log laden uit CSV, train/test split, data validatie |
| `preprocessing.py` | Data cleaning: verwijder korte cases, filter incomplete data |

### `/src/utils/` - Utilities

| Bestand | Verantwoordelijkheid |
|---------|---------------------|
| `duration_fitting.py` | Fit duration distributies op training data, save/load fitted distributions voor consistent gebruik tijdens evaluatie |

---

## 📜 `/scripts/` - Executable Scripts

### Main Script

| Bestand | Verantwoordelijkheid |
|---------|---------------------|
| `train.py` | 🚀 **HOOFDSCRIPT**: Orkestreert de volledige workflow:<br>1. Data laden en preprocessing<br>2. Duration distributions fitten<br>3. Environment initialisatie<br>4. Agent training (MAPPO/QMIX)<br>5. Evaluation vs baselines<br>6. Results opslaan |

**Gebruik**: `python scripts/train.py [--algorithm mappo/qmix] [--episodes N]`

### `/scripts/evaluate/` - Evaluatie Scripts

Scripts voor model evaluatie op verschillende datasets:

| Bestand | Doel |
|---------|------|
| `evaluate_baselines.py` | Evalueer baseline agents op test data |
| `evaluate_checkpoints.py` | Evalueer opgeslagen model checkpoints |
| `run_baseline_evaluation.py` | Run baseline evaluatie voor single dataset |
| `run_multi_dataset_baseline_evaluation.py` | Evalueer baselines over meerdere datasets tegelijk |
| `run_comprehensive_baseline_evaluation.py` | Complete evaluatie suite met alle baselines en settings |
| `test_baseline_script.py` | Quick test om te verifiëren dat evaluatie werkt |

### `/scripts/demo/` - Demo Scripts

| Bestand | Doel |
|---------|------|
| `demo_baseline_plots.py` | Generate demo plots voor baseline vergelijkingen |
| `demo_baseline_log_metrics.py` | Demo van log metrics visualisatie |

---

## 📊 `/analysis/` - Analyse & Visualisatie

### `/analysis/plotting/` - Plot Scripts

Scripts voor het genereren van visualisaties:

| Bestand | Doel |
|---------|------|
| `plot_rewards.py` | Plot training rewards over tijd |
| `plot_evaluation_metrics.py` | Plot evaluatie metrics (throughput, case duration, etc.) |
| `plot_baseline_results.py` | Vergelijk baseline performance |
| `plot_baseline_distributions.py` | Visualiseer duration distributies |
| `plot_distributions.py` | Plot log metrics en distributies |
| `plot_baseline_log_metrics.py` | Analyse baseline log metrics |

### `/analysis/metrics/` - Metrics

| Bestand | Doel |
|---------|------|
| `metrics.py` | Bereken evaluatie metrics: case duration, throughput, resource utilization, SLA compliance |

### `/analysis/notebooks/` - Jupyter Notebooks

| Notebook | Doel |
|----------|------|
| `qmix_evaluation_analysis.ipynb` | Diepgaande analyse van QMIX resultaten |
| `test.ipynb` | Exploratory data analysis en quick tests |

---

## 💾 `/data/` - Data Folders

| Folder | Inhoud |
|--------|--------|
| `input/` | Event logs (CSV): plaats hier je process mining event logs |
| `processed/` | Preprocessed data na cleaning en filtering |
| `distributions/` | Opgeslagen fitted duration distributies (pickle files) |

**Expected CSV format**:
```csv
case_id,activity,resource,timestamp,other_columns...
```

---

## 🧪 `/tests/` - Tests

Unit tests voor verschillende componenten:

| Bestand | Test Coverage |
|---------|---------------|
| `test_duration_fitting.py` | Test duration distribution fitting logic |
| `test_baseline_script.py` | Test baseline evaluation functionality |

---

## 📖 `/docs/` - Documentatie

Documentatie bestanden uit het project:

| Bestand | Inhoud |
|---------|--------|
| `BASELINE_EVALUATION_GUIDE.md` | Uitgebreide guide voor baseline evaluaties |
| `CONSOLIDATION_SUMMARY.md` | Samenvatting van code consolidatie |
| `evaluation_analysis_summary.md` | Evaluatie analyse overzicht |

---

## 🎯 Key Workflows

### 1. Training Workflow

```
data/input/*.csv
  → preprocessing (load_data, clean)
  → duration fitting (fit distributions)
  → environment init (custom_environment)
  → agent training (mappo/qmix trainer)
  → evaluation (vs baselines)
  → results → experiments/
```

### 2. Evaluation Workflow

```
trained model checkpoint
  → load model
  → init environment (with test data)
  → run episodes
  → compute metrics
  → compare vs baselines
  → generate plots
```

### 3. Analysis Workflow

```
experiments/ results
  → load results (JSON/CSV)
  → compute statistics
  → generate visualizations (plotting/)
  → notebook analysis
  → insights & conclusions
```

---

## 🔑 Key Concepts

### Agent
Een resource uit de event log. Heeft:
- State (current tasks, availability)
- Policy (learned strategy)
- Observation (ziet beschikbare tasks)
- Action (volunteer of niet)

### Task
Een activiteit uit de event log. Heeft:
- Activity type
- Duration (sampled uit distribution)
- Assignment (aan welke agent)
- Status (pending/open/in_progress/completed)

### Case
Een complete process instance. Heeft:
- Sequence van tasks
- Start/end timestamp
- Total duration
- Status

### Environment
De simulatie omgeving waar:
- Tasks arriveren volgens event log
- Agents volunteeren voor tasks
- Taken worden uitgevoerd met sampled durations
- Rewards worden gegeven bij case completion

### Episode
Een complete simulatie run:
- Start bij eerste case
- Verwerk alle cases
- End als alle cases completed
- Geeft episode reward

---

## 💡 Tips voor Gebruik

### Quick Start
```bash
# 1. Zet event log in data/input/train.csv
# 2. Update src/core/config.py met je kolom namen
# 3. Run training
python scripts/train.py
```

### Custom Algorithm Parameters
Pas parameters aan in `scripts/train.py`:
- Learning rate
- Network architecture (hidden_size)
- Training episodes
- Gamma (discount factor)
- GAE lambda

### Add New Baseline
1. Implementeer nieuwe baseline klasse in `src/algorithms/baselines/baselines.py`
2. Inherit van base klasse
3. Implementeer `select_actions()` method
4. Add to `create_baseline_agents()`

### Add New Metrics
1. Add metric functie in `analysis/metrics/metrics.py`
2. Compute in evaluation script
3. Add to results JSON
4. Create visualization in plotting scripts

---

## 🐛 Debugging

### Environment niet converging?
- Check reward function in `src/environment/reward.py`
- Verify duration distributions in `data/distributions/`
- Lower learning rate
- Increase training episodes

### Import errors?
- Verify Python path includes repository root
- Check all `__init__.py` files exist
- Verify relative imports are correct

### Memory issues?
- Reduce batch size in trainer
- Use smaller network (hidden_size)
- Process fewer episodes at once
- Clear GPU cache (`torch.cuda.empty_cache()`)

---

Dit document geeft een complete overview van de repository structuur en hoe alles samenwerkt!
