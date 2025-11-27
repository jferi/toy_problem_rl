import torch
from torch import Tensor

# ===== Aliases to make type-setting more expressive =====
PriceSeq = Tensor       # shape: (B, T)
Position = Tensor       # shape: (B, )
State = tuple[PriceSeq, Position]
QValues = Tensor        # shape: (B, A)
Action = Tensor         # shape: (B,)
Reward = Tensor         # shape: (B,)


def action_2_pos(action: Tensor) -> Tensor:
    """ Convert action [0, 1, 2] to position [-1.0, 0.0, 1.0] """
    return (action - 1).to(dtype=torch.float32)


def pos_2_action(pos: Tensor) -> Tensor:
    """ Convert position [-1.0, 0.0, 1.0] to action [0, 1, 2] """
    return pos.to(dtype=torch.long) + 1


class Config:
    def __init__(
            self,
            # ===== Data tensor sizes =====
            num_actions: int = 3,
            batch_size: int = 128,
            lookback: int = 10,

            # ===== Price dynamics =====
            S_mean: float = 100.0,
            vol: float = 2.0,
            kappa: float = 0.1,

            # ===== Reward specification =====
            gamma: float = 0.99,
            half_ba: float = 1.0,
            risk_av: float = 0.02,

            # ===== Network specification =====
            hidden_size: int = 128,

            # ===== Exploration parameters =====
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            epsilon_decay: float = 0.9999,

            # ===== Training control =====
            learning_rate: float = 0.0001,
            target_update: int = 10,
            buffer_capacity: int = 100000,
            max_steps: int = 300,
            num_episodes: int = 500,
            print_every: int = 20
    ):
        # Data tensor sizes
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.lookback = lookback

        # Price dynamics
        self.S_mean = S_mean
        self.vol = vol
        self.kappa = kappa

        # Reward specification
        self.gamma = gamma
        self.half_ba = half_ba
        self.risk_av = risk_av

        # Network specification
        self.hidden_size = hidden_size

        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Training control
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.buffer_capacity = buffer_capacity
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.print_every = print_every
