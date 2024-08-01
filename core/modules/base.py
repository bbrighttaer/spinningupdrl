import torch.nn as nn

from core.utils import DotDic


class TorchModel(nn.Module):

    def __init__(self, config: DotDic):
        super().__init__()
        self.obs_dim = config.obs_size
        self.action_dim = config.n_actions
        self.comm_dim = config.comm_size
        self.state_dim = config.state_size
        self.model_config = config

    def get_initial_state(self):
        return []

    def forward(self, input_obs, hidden_state, **kwargs):
        ...
