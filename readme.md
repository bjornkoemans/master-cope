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
│   ├── 1_prepare_data.py         # 📦 Fase 1 - Data preprocessing
│   ├── 2_fit_distributions.py    # 📊 Fase 2 - Distribution fitting
│   ├── 3_train.py                # 🚀 Fase 3 - Model training
│   ├── 4_evaluate.py             # 📈 Fase 4 - Model evaluation
│   ├── 5_compare_models.py       # 🔍 Fase 5 - Model comparison
│   ├── hyperparameter_search.py  # 🔬 Geautomatiseerde hyperparameter search
│   ├── train.py                  # 🚀 Legacy - Single script training
│   ├── evaluate/                 # Evaluatie scripts
│   └── demo/                     # Demo scripts
│
├── configs/                       # YAML configuratie bestanden
│   ├── default.yaml              # Default hyperparameters
│   └── experiments/              # Experiment-specifieke configs
│       ├── small_network.yaml    # Klein netwerk configuratie
│       ├── large_network.yaml    # Groot netwerk configuratie
│       └── high_lr.yaml          # Hoge learning rate config
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

### Optie 1: Modular Pipeline (Aanbevolen voor Hyperparameter Tuning)

De modular pipeline splitst training op in herbruikbare fasen, ideaal voor hyperparameter optimization:

#### **Fase 1: Data Preprocessing** (eenmalig per dataset)
```bash
python scripts/1_prepare_data.py \
  --input data/input/jouw_eventlog.csv \
  --output data/processed/preprocessed_data.pkl
```
- Laadt event log
- Verwijdert korte cases
- Splitst in train/test sets (83/17)
- Slaat op als pickle bestand

#### **Fase 2: Distribution Fitting** (eenmalig per dataset)
```bash
python scripts/2_fit_distributions.py \
  --data data/processed/preprocessed_data.pkl \
  --output data/distributions/fitted_distributions.pkl
```
- Fit duration distributions op training data
- Slaat gefitte distributies op voor hergebruik

#### **Fase 3: Training** (run meerdere keren met verschillende configs)
```bash
# Met default configuratie
python scripts/3_train.py \
  --data data/processed/preprocessed_data.pkl \
  --distributions data/distributions/fitted_distributions.pkl

# Met custom config
python scripts/3_train.py \
  --data data/processed/preprocessed_data.pkl \
  --distributions data/distributions/fitted_distributions.pkl \
  --config configs/experiments/large_network.yaml \
  --name "large_network_exp"
```
- Traint model met YAML configuratie
- Slaat model op in `experiments/`

#### **Fase 4: Evaluation**
```bash
python scripts/4_evaluate.py \
  --model experiments/exp_20231215_120000/models \
  --data data/processed/preprocessed_data.pkl \
  --distributions data/distributions/fitted_distributions.pkl \
  --episodes 20
```
- Evalueert trained model op test data
- Genereert evaluation metrics en resultaten

#### **Fase 5: Model Comparison**
```bash
python scripts/5_compare_models.py \
  --models experiments/exp1/models experiments/exp2/models experiments/exp3/models \
  --data data/processed/preprocessed_data.pkl \
  --distributions data/distributions/fitted_distributions.pkl \
  --episodes 20
```
- Vergelijkt meerdere models side-by-side
- Genereert comparison table en relatieve performance

#### **Hyperparameter Search**
```bash
# Grid search met default parameters
python scripts/hyperparameter_search.py \
  --data data/processed/preprocessed_data.pkl \
  --distributions data/distributions/fitted_distributions.pkl \
  --search-type grid

# Random search met custom parameters
python scripts/hyperparameter_search.py \
  --data data/processed/preprocessed_data.pkl \
  --distributions data/distributions/fitted_distributions.pkl \
  --search-type random \
  --param-config configs/param_search.json \
  --n-trials 20
```
- Automatiseert hyperparameter optimization
- Traint meerdere models met verschillende configuraties
- Vergelijkt automatisch alle resultaten

### Optie 2: Single-Script Training (Legacy)

Voor snelle single runs:

```bash
# Zet je event log in data/input/
# Pas src/core/config.py aan voor je dataset kolommen
python scripts/train.py
```

### Configuratie

#### **YAML Config Bestanden** (`configs/`)
- `configs/default.yaml`: Default hyperparameters
- `configs/experiments/small_network.yaml`: Kleiner netwerk (sneller)
- `configs/experiments/large_network.yaml`: Groter netwerk (meer capaciteit)
- `configs/experiments/high_lr.yaml`: Hogere learning rate

Config structuur:
```yaml
training:
  episodes: 100
  policy_update_epochs: 10

network:
  actor_hidden_size: 128
  critic_hidden_size: 256
  dropout_rate: 0.2
  weight_init: "xavier_uniform"

learning:
  lr_actor: 0.0003
  lr_critic: 0.0003
  gamma: 0.99

ppo:
  clip_param: 0.2
  batch_size: 32768
```

#### **Event Log Configuratie** (`src/core/config.py`):
- Definieer kolom mappings voor je dataset
- Specificeer case ID, activity, resource, timestamp kolommen

#### **Environment Parameters** (`src/core/env_config.py`):
- Debug settings
- Simulatie parameters

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

### Modular Pipeline Workflow (Aanbevolen)

```
1_prepare_data.py         → data/processed/preprocessed_data.pkl
         ↓
2_fit_distributions.py    → data/distributions/fitted_distributions.pkl
         ↓
3_train.py (meerdere runs met verschillende configs)
         ↓                ↓                ↓
    model_1/         model_2/         model_3/
         ↓                ↓                ↓
4_evaluate.py        4_evaluate.py    4_evaluate.py
         ↓                ↓                ↓
         └────────────────┴────────────────┘
                         ↓
                5_compare_models.py
                         ↓
            📊 Best model selectie
```

### Stappen:

1. **Data Preprocessing** (1x per dataset) → Event logs preprocessen en splitsen
2. **Distribution Fitting** (1x per dataset) → Task duration distributions fitten
3. **Training** (Nx per experiment) → Meerdere models trainen met verschillende configs
4. **Evaluation** → Individuele model performance evalueren
5. **Comparison** → Alle models vergelijken en beste selecteren
6. **Analyse** → Resultaten visualiseren en interpreteren

**Voordeel**: Fase 1 en 2 hoeven maar 1x uitgevoerd te worden. Fase 3 kan parallel voor meerdere hyperparameter configuraties, wat hyperparameter tuning veel efficiënter maakt.

## 🎓 Master Thesis Context

Dit project onderzoekt hoe MARL gebruikt kan worden voor resource allocation optimalisatie in business processen, met focus op:
- Autonome agent decision making
- Cooperative behavior learning
- Process optimization without explicit rules
- Comparison with traditional allocation strategies
