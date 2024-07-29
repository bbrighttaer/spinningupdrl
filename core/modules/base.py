import torch
import torch.nn as nn


class TorchModel(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model_config = config

    def initial_state(self):
        return []

    def forward(self, input_obs, hidden_state, **kwargs):
        ...
