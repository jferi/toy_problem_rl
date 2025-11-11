# DQN Mean-Reverting Trading

Deep Q-Network implementation for automated trading of mean-reverting assets.

## Structure

```
trading_dqn/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration parameters
│   ├── environment.py       # Trading environment (market simulation)
│   ├── network.py           # DQN neural network architecture
│   ├── replay_buffer.py     # Experience replay buffer
│   ├── agent.py             # DQN agent with training logic
│   ├── training.py          # Training loop
│   └── evaluation.py        # Evaluation and visualization
├── main.py                  # Main entry point (full training)
├── requirements.txt         # Dependencies
├── .gitignore               # Git ignore patterns
└── README.md                # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py
```
