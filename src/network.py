import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NetworkConfig


class DQNNetwork(nn.Module):
    
    def __init__(
        self, 
        lookback: int = NetworkConfig.LOOKBACK,
        hidden_size: int = NetworkConfig.HIDDEN_SIZE,
        num_actions: int = NetworkConfig.NUM_ACTIONS
    ):
        super(DQNNetwork, self).__init__()
        
        input_size = lookback * 2  # prices + positions
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_actions)
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self, 
        price_history: torch.Tensor, 
        position_history: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([price_history, position_history], dim=1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        q_values = self.fc4(x)
        
        return q_values

