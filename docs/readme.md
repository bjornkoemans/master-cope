# AgentOptimizer

This is the repository for my master thesis project. In this project, I'm developing a Multi-agent Reinforcement Learning environment for optimizing business processes.

## Usage

To run the environment, drop your event logs in the `data/input` folder. Change the config in `config.py` to match your input data columns and run the main script.

```python main.py```

## Pre-processing

The code used for pre-processing the data is in the `preprocessing` folder. The steps in pre-processing include:

1. Loading the data
2. Eliminating short-path traces, i.e. shorter than 3 states (WIP)
3. Discovering source and target activities (WIP)

### Agent Discovery

## Environment

The environment is built using the PettingZoo library and it is a translation from the original Mesa environment made by the authors of the [AgentSimulator](https://github.com/lukaskirchdorfer/AgentSimulator) paper.
