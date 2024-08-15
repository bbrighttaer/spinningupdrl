import torch.nn as nn

from core import utils
from core.modules.base import TorchModel
from core.utils import DotDic


class SimpleFCNet(TorchModel):

    def __init__(self, config: DotDic):
        super().__init__(config)

        layers = []
        activation = utils.get_activation_function(self.model_config.activation)

        if self.model_config.obs_size == 1 and self.model_config.num_discrete_obs > 0:
            self.embedding = nn.Embedding(self.model_config.num_discrete_obs, self.model_config.embedding_dim)
            input_dim = self.model_config.embedding_dim
        else:
            self.embedding = None
            input_dim = self.obs_dim

        # add fingerprint if set
        input_dim += self.fp_dim

        # create hidden layers
        for dim in self.model_config.hidden_layers:
            layers.extend([
                nn.Linear(input_dim, dim),
                activation(),
            ])
            input_dim = dim

        # output layer
        layers.append(nn.Linear(input_dim, self.action_dim))

        # create model
        self.model = nn.Sequential(*layers)

    def forward(self, input_obs, hidden_state, **kwargs):
        if self.embedding is not None:
            input_obs = self.embedding(input_obs.long()).view(input_obs.shape[0], -1)
        x = self.model(input_obs)
        return x, []
