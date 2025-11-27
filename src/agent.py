import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from typing import Optional

from src.config import Config
from .network import DQNNetwork
from .replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    
    def __init__(self, conf: Config):
        self.conf = conf
        self.lookback = conf.lookback
        self.gamma = conf.gamma
        self.epsilon = conf.epsilon_start
        self.epsilon_end = conf.epsilon_end
        self.epsilon_decay = conf.epsilon_decay
        self.target_update = conf.target_update
        self.batch_size = conf.batch_size
        
        # Q-network and target network
        self.q_network = DQNNetwork(conf).to(device)
        self.target_network = DQNNetwork(conf).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=conf.learning_rate)
        self.memory = ReplayBuffer(conf)
        self.steps_done = 0
    
    def select_action(
        self, 
        state_price: torch.Tensor, 
        state_pos: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        if training and random.random() < self.epsilon:
            # Exploration: random action
            batch_size = state_price.shape[0]
            return torch.randint(0, 3, (batch_size,), device=device)
        else:
            # Exploitation: best action based on Q-values
            with torch.no_grad():
                self.q_network.eval()
                q_values = self.q_network(state_price, state_pos)
                self.q_network.train()
                return q_values.argmax(dim=1)
    
    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        state_price, state_pos, action, reward, next_state_price, next_state_pos, done = \
            self.memory.sample(self.batch_size)
        
        # Current Q-values
        current_q_values = self.q_network(state_price, state_pos).gather(
            1, action.unsqueeze(1)
        ).squeeze(1)
        
        # Double DQN: use q_network to select action, target_network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_state_price, next_state_pos).argmax(dim=1)
            next_q_values = self.target_network(next_state_price, next_state_pos).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            
            next_q_values = next_q_values * (~done).float()
            target_q_values = reward + self.gamma * next_q_values
        
        # Huber loss for robustness
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
