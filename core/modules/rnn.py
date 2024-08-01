

import torch
import torch.nn as nn

from core import utils
from core.modules.base import TorchModel
from core.utils import DotDic


class SimpleRNN(TorchModel):

    def __init__(self, config: DotDic):
        super().__init__(config)
        activation = utils.get_activation_function(self.model_config["activation"])

        # encoder
        enc_layers = []
        input_dim = self.obs_dim
        for hdim in self.model_config.encoder_layers[:-1]:
            enc_layers.extend([
                nn.Linear(input_dim, hdim),
                activation()
            ])
            input_dim = hdim
        enc_out_dim = self.model_config.encoder_layers[-1]
        enc_layers.extend([
            nn.Linear(input_dim, enc_out_dim),
            nn.BatchNorm1d(enc_out_dim),
            activation()
        ])
        self.encoder = nn.Sequential(*enc_layers)

        # recurrent layer
        self.rnn = nn.GRU(
            input_size=enc_out_dim,
            hidden_size=self.model_config.hidden_state_dim,
            num_layers=self.model_config.num_rnn_layers,
            batch_first=True,
            dropout=self.model_config.dropout_rate,
        )

        # output layers
        self.output = nn.Sequential(
            nn.BatchNorm1d(self.model_config.hidden_state_dim),
            activation(),
            nn.Linear(self.model_config.hidden_state_dim, self.action_dim)
        )

    def get_initial_state(self):
        # Place hidden states on same device as model.
        h0 = self.rnn.all_weights[0][0].new(
            1, self.model_config.hidden_state_dim
        ).zero_()
        hidden_state = [h0.detach().clone() for _ in range(self.model_config.num_rnn_layers)]
        return hidden_state

    def forward(self, input_obs, hidden_state, **kwargs):
        batch_size = input_obs.shape[0]

        # combine hidden states
        hidden_state = torch.stack(hidden_state)

        # get obs embeddings
        z = self.encoder(input_obs).unsqueeze(1)

        # apply recurrent model
        z_out, h = self.rnn(z, hidden_state)
        h = [hs.view(batch_size, -1) for hs in torch.split(h, [1] * self.model_config.num_rnn_layers)]

        # apply output model
        z_out = z_out.view(batch_size, -1)
        x = self.output(z_out)

        return x, h
