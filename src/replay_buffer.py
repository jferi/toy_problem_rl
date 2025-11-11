import torch
import random
from collections import deque, namedtuple
from typing import Tuple

from src.config import AgentConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple('Experience', 
    ['state_price', 'state_pos', 'action', 'reward', 'next_state_price', 'next_state_pos', 'done']
)


class ReplayBuffer:
    
    def __init__(self, capacity: int = AgentConfig.BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state_price: torch.Tensor,
        state_pos: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state_price: torch.Tensor,
        next_state_pos: torch.Tensor,
        done: torch.Tensor
    ):
        batch_size = state_price.shape[0]
        
        for i in range(batch_size):
            exp = Experience(
                state_price[i].cpu(),
                state_pos[i].cpu(),
                action[i].cpu(),
                reward[i].cpu(),
                next_state_price[i].cpu(),
                next_state_pos[i].cpu(),
                done[i].cpu()
            )
            self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        experiences = random.sample(self.buffer, batch_size)
        batch = Experience(*zip(*experiences))
        
        state_price = torch.stack(batch.state_price).to(device)
        state_pos = torch.stack(batch.state_pos).to(device)
        action = torch.stack(batch.action).to(device)
        reward = torch.stack(batch.reward).to(device)
        next_state_price = torch.stack(batch.next_state_price).to(device)
        next_state_pos = torch.stack(batch.next_state_pos).to(device)
        done = torch.stack(batch.done).to(device)
        
        return state_price, state_pos, action, reward, next_state_price, next_state_pos, done
    
    def __len__(self):
        return len(self.buffer)

