# models/net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    """
    Simple neural network that takes state tensor and outputs policy logits and value.
    """
    def __init__(self, state_dim: int = 256, action_dim: int = 4096, hidden_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor):
        """
        Args:
            state: tensor of shape (batch_size, state_dim) or (state_dim,)
        
        Returns:
            policy_logits: (batch_size, action_dim) or (action_dim,)
            value: (batch_size, 1) or (1,)
        """
        # Handle single state
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single = True
        else:
            single = False
        
        # Shared layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        if single:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)
        
        return policy_logits, value
