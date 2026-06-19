# models/replay_buffer.py
import random
import collections
import torch

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state_tensor, policy_target, value_target):
        # state_tensor: torch.Tensor (state_dim,)
        # policy_target: numpy or torch vector length ACTION_DIM
        # value_target: float in [-1,1]
        self.buffer.append((state_tensor.detach().cpu(), torch.tensor(policy_target, dtype=torch.float), torch.tensor([value_target], dtype=torch.float)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        states = torch.stack(states).float()
        policies = torch.stack(policies).float()
        values = torch.stack(values).float()
        return states, policies, values

    def __len__(self):
        return len(self.buffer)
