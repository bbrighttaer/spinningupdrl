
import torch
import torch.nn as nn

from core.modules.base import TorchModel


class SimpleCommNet(TorchModel):

    def __init__(self, config):
        super().__init__(config)
        self.linear1 = nn.Linear(self.obs_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, self.comm_dim)

    def forward(self, input_x, hidden_state, **kwargs):
        x = torch.relu(self.linear1(input_x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x, []
