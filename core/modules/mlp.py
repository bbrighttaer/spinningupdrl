
import torch
import torch.nn as nn

from core import utils
from core.modules.base import TorchModel
from core.utils import DotDic


class SimpleFCNet(TorchModel):

    def __init__(self, obs_dim: int, action_dim: int, config: DotDic):
        super().__init__(obs_dim, action_dim, config)

        layers = []
        activation = utils.get_activation_function(self.model_config["activation"])

        # create hidden layers
        input_dim = obs_dim
        for dim in self.model_config["hidden_layers"]:
            layers.extend([
                nn.Linear(input_dim, dim),
                activation(),
            ])
            input_dim = dim

        # output layer
        layers.append(nn.Linear(input_dim, action_dim))

        # create model
        self.model = nn.Sequential(*layers)

    def forward(self, input_obs, hidden_state, **kwargs):
        x = self.model(input_obs)
        return x, []
