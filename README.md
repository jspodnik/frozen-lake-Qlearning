# CPSC 6420 Assignment 2

Tabular Q-learning applied to Gymnasium's FrozenLake (8x8, non-slippery). The code covers a baseline agent, hyperparameter experiments, an evaluation of the best config, and a robustness experiment under added action noise.

## Project structure

```
.
├── baseline.py              # Baseline Q-learning agent
├── hyperparameter_test.py   # Script to compare different hyperparameter values
├── evaluate_best.py         # Trains the best config, evaluates greedy policy, records video
├── stochastic_test.py       # Adds action noise (0, 0.10, 0.25) and tests robustness
├── requirements.txt
└── results/                 # Output plots and the recorded agent video
```

## Setup

Requires Python 3.9+.

```bash
# clone the repository
git clone https://github.com/jspodnik/frozen-lake-Qlearning.git
cd frozen-lake-Qlearning

# (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# install dependencies
pip install -r requirements.txt
```

## Running the code

Each script is standalone. Run them from the repository root:

```bash
python baseline.py
python hyperparameter_test.py
python evaluate_best.py
python stochastic_test.py
```

| Script | What it does | Output |
|---|---|---|
| `baseline.py` | Trains Q-learning (α=0.1, γ=0.99, ε=0.3, 2000 episodes) and plots cumulative and rolling-average reward | Plot window + summary stats in the terminal |
| `hyperparameter_test.py` | By default, trains with ε ∈ {0.05, 0.2, 0.5} using the same seed and compares rolling-average reward. The tested hyperparameter can be changed with minor tweaks to the script. | Plot window | 
| `evaluate_best.py` | Trains with the best config (ε=0.05), evaluates the greedy policy over 30 episodes, and records the first episode | `results/frozenlake_agent.mp4` + terminal stats |
| `stochastic_test.py` | Trains and evaluates under action noise p ∈ {0, 0.10, 0.25}, then tests the p=0 policy under noise | `results/stochastic_*.png` + terminal stats |


## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.99 |
| Exploration rate (ε) | 0.3 baseline; 0.05 best |
| Training episodes | 2000 |
| Random seed | 42 |

All experiments use fixed seeds, so results should be reproducible on the same library versions.

## Results

Results are saved in the `results/` folder:

- `partC.png`: baseline training rewards
- `partD.png`: effect of α on learning speed
- `partE.png`: effect of γ on learning speed
- `partF.png`: effect of ε on learning speed
- `frozenlake_agent.mp4`: video of the trained agent reaching the goal
- `stochastic_learning_curves.png`: training curves under different noise levels
- `stochastic_evaluation.png`: evaluation success rate per noise level
- `stochastic_robustness.png`: p=0 policy vs policies trained at matching noise

