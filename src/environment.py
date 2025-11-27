import torch
from typing import Tuple

from src.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TradingEnvironment:
    
    def __init__(self, conf: Config):
        self.batch_size = conf.batch_size
        self.s_mean = conf.S_mean
        self.vol = conf.vol
        self.kappa = conf.kappa
        self.half_ba = conf.half_ba
        self.risk_av = conf.risk_av
        self.lookback = conf.lookback
        self.max_steps = conf.max_steps
        
        self.current_prices = None
        self.price_history = None
        self.position_history = None
        self.current_position = None
        self.step_count = None
        
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset environment to initial state"""
        self.current_prices = torch.tensor(
            self.s_mean + torch.randn(self.batch_size) * self.vol,
            dtype=torch.float32, device=device
        )
        
        self.price_history = torch.zeros(
            (self.batch_size, self.lookback), 
            dtype=torch.float32, device=device
        )
        self.position_history = torch.zeros(
            (self.batch_size, self.lookback), 
            dtype=torch.float32, device=device
        )
        
        self.price_history[:, -1] = self.current_prices
        self.current_position = torch.zeros(
            self.batch_size, dtype=torch.float32, device=device
        )
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized state"""
        normalized_prices = (self.price_history - self.s_mean) / (5 * self.vol)
        return normalized_prices, self.position_history
    
    def step(self, actions: torch.Tensor) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """Execute one timestep: action -> next_state, reward, done"""
        # Convert actions (0,1,2) to positions (-1,0,1)
        new_positions = actions.float() - 1.0
        
        # Price dynamics: mean-reverting with random shocks
        random_shocks = torch.randn(self.batch_size, device=device) * self.vol
        mean_reversion = -self.kappa * (self.current_prices - self.s_mean)
        price_changes = random_shocks + mean_reversion
        new_prices = self.current_prices + price_changes
        
        # Reward components
        pnl = new_positions * price_changes
        trading_cost = self.half_ba * torch.abs(new_positions - self.current_position)
        risk_penalty = self.risk_av * (self.vol ** 2) * (new_positions ** 2)
        rewards = pnl - trading_cost - risk_penalty
        
        # Update state
        self.price_history = torch.roll(self.price_history, shifts=-1, dims=1)
        self.price_history[:, -1] = new_prices
        
        self.position_history = torch.roll(self.position_history, shifts=-1, dims=1)
        self.position_history[:, -1] = new_positions
        
        self.current_prices = new_prices
        self.current_position = new_positions
        self.step_count += 1
        
        # Check if episode is done
        dones = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        if self.step_count >= self.max_steps:
            dones[:] = True
        
        return self._get_state(), rewards, dones
