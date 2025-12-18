# MARL Resource Allocation - Master Thesis

Dit is de repository voor mijn master thesis project. In dit project ontwikkel ik een Multi-Agent Reinforcement Learning (MARL) oplossing voor resource allocation in business processen.

## 🎯 Concept

Het systeem modelleert elke resource uit event logs als een autonome agent. Deze agents "volunteeren" of ze een inkomende taak willen aannemen op basis van geleerde strategieën. Het doel is om de totale doorlooptijd van cases te optimaliseren.

## 📁 Repository Structuur

```
master-cope/
│
├── src/                           # Broncode modules
│   ├── core/                      # Kern configuratie en utilities
│   │   ├── config.py             # Dataset configuratie (kolom mapping)
│   │   ├── env_config.py         # Environment parameters
│   │   └── display.py            # Terminal output helpers
│   │
│   ├── environment/               # Custom MARL omgeving
│   │   ├── custom_environment.py # PettingZoo environment implementatie
│   │   ├── objects.py            # Agent, Task, Case objecten
│   │   ├── duration_distribution.py  # Task duration modeling
│   │   ├── reward.py             # Reward functie voor agents
│   │   ├── data_handling.py      # Event log verwerking
│   │   └── typed_queue.py        # Queue implementatie
│   │
│   ├── algorithms/                # RL algoritmes
│   │   ├── mappo/                # Multi-Agent PPO
│   │   │   ├── agent.py          # MAPPO agent (actor-critic)
│   │   │   ├── trainer.py        # Training loop
│   │   │   ├── online_trainer.py # Online training variant
│   │   │   └── networks.py       # Neural network architectures
│   │   │
│   │   ├── qmix/                 # QMIX algoritme
│   │   │   ├── agent.py          # QMIX agent met mixing network
│   │   │   └── trainer.py        # QMIX training loop
│   │   │
│   │   └── baselines/            # Baseline agents voor vergelijking
│   │       └── baselines.py      # Random, BestMedian, GroundTruth
│   │
│   ├── preprocessing/             # Data preprocessing
│   │   ├── load_data.py          # Event log laden en splitten
│   │   └── preprocessing.py      # Data cleaning
│   │
│   └── utils/                     # Utilities
│       └── duration_fitting.py   # Duration distribution fitting
│
├── scripts/                       # Executable scripts
│   ├── train.py                  # 🚀 HOOFDSCRIPT - Training en evaluatie
│   ├── evaluate/                 # Evaluatie scripts
│   └── demo/                     # Demo scripts
│
├── analysis/                      # Analyse en visualisatie
│   ├── notebooks/                # Jupyter notebooks voor analyse
│   ├── plotting/                 # Plot scripts
│   └── metrics/                  # Metric berekeningen
│
├── data/                          # Data folders
│   ├── input/                    # Event logs (CSV bestanden)
│   ├── processed/                # Preprocessed data
│   └── distributions/            # Fitted duration distributions
│
├── experiments/                   # Training runs en resultaten
├── tests/                         # Unit tests
├── docs/                          # Documentatie bestanden
└── requirements.txt              # Python dependencies

```

## 🚀 Gebruik

### 1. Training Run

Om het systeem te trainen op je event logs:

```bash
# Zet je event log in data/input/
# Pas src/core/config.py aan voor je dataset kolommen
python scripts/train.py
```

### 2. Configuratie

**Event Log Configuratie** (`src/core/config.py`):
- Definieer kolom mappings voor je dataset
- Specificeer case ID, activity, resource, timestamp kolommen

**Environment Parameters** (`src/core/env_config.py`):
- Debug settings
- Simulatie parameters

### 3. Preprocessing

Data preprocessing stappen:
1. Event logs laden vanuit `data/input/`
2. Korte cases verwijderen (< 3 stappen)
3. Train/test split
4. Duration distributions fitten

## 🧠 Algoritmes

### MAPPO (Multi-Agent Proximal Policy Optimization)
- Gebruikt in `src/algorithms/mappo/`
- Actor-critic architectuur
- Geschikt voor cooperative multi-agent settings

### QMIX
- Gebruikt in `src/algorithms/qmix/`
- Value-based methode met mixing network
- Combineert individuele agent Q-values

### Baselines
- **Random**: Selecteert acties random
- **BestMedian**: Alleen best presterende agent volunteert
- **GroundTruth**: Volgt werkelijke assignments uit data

## 📊 Evaluatie & Analyse

- **Evaluation scripts**: `scripts/evaluate/`
- **Plotting tools**: `analysis/plotting/`
- **Jupyter notebooks**: `analysis/notebooks/`
- **Resultaten**: Worden opgeslagen in `experiments/`

## 🔧 Dependencies

```bash
pip install -r requirements.txt
```

Belangrijkste dependencies:
- PyTorch (deep learning)
- PettingZoo (multi-agent environments)
- Pandas (data processing)
- NumPy, Matplotlib, etc.

## 📖 Environment Details

De environment is gebouwd met de **PettingZoo** library en is gebaseerd op het [AgentSimulator](https://github.com/lukaskirchdorfer/AgentSimulator) paper.

### Key Components:

1. **Agents (Resources)**: Elke resource wordt een zelfstandige agent
2. **Tasks**: Individuele activiteiten uit de event log
3. **Cases**: Complete process instances
4. **Observation Space**: Agent ziet eigen staat + beschikbare tasks
5. **Action Space**: Binary (volunteer voor task of niet)
6. **Reward**: Gebaseerd op case completion time vs historische performance

## 📝 Workflow

1. **Data Laden** → Event logs uit `data/input/`
2. **Preprocessing** → Cleaning, filtering, train/test split
3. **Distribution Fitting** → Task duration distributions fitten
4. **Training** → MAPPO/QMIX agent training
5. **Evaluation** → Performance vergelijking met baselines
6. **Analyse** → Resultaten visualiseren en interpreteren

## 🎓 Master Thesis Context

Dit project onderzoekt hoe MARL gebruikt kan worden voor resource allocation optimalisatie in business processen, met focus op:
- Autonome agent decision making
- Cooperative behavior learning
- Process optimization without explicit rules
- Comparison with traditional allocation strategies
