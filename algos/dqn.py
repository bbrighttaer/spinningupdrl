import torch

from algos import Policy
from core import constants, utils
from core.buffer import sample_batch
from core.exploration import EpsilonGreedy
from core.modules.mlp import SimpleFCNet


class DQNPolicy(Policy):
    """
    Single-agent DQN Policy
    """

    def __init__(self, config, summary_writer, logger):
        super().__init__(config, summary_writer, logger)
        algo_config = config[constants.ALGO_CONFIG]
        model_config = config[constants.MODEL_CONFIG]

        # create model
        # if model_config.core_arch == "mlp":
        self.model = SimpleFCNet(self.obs_size, self.act_size, model_config).to(self.device)
        self.target_model = SimpleFCNet(self.obs_size, self.act_size, model_config).to(self.device)

        # create optimizers
        if algo_config.optimizer == "rmsprop":
            from torch.optim import RMSprop
            self.optimizer = RMSprop(params=self.model.parameters(), lr=algo_config.learning_rate)
        elif algo_config.optimizer == "adam":
            from torch.optim import Adam
            self.optimizer = Adam(params=self.model.parameters(), lr=algo_config.learning_rate)
        else:
            raise RuntimeError(f"Unsupported optimizer: {algo_config.optimizer}")

        # exploration or action selection strategy
        self.exploration = EpsilonGreedy(
            initial_epsilon=algo_config.epsilon,
            final_epsilon=algo_config.final_epsilon,
            epsilon_timesteps=algo_config.epsilon_timesteps,
        )

        # trigger initial network sync
        self.update_target()

    def initial_hidden_state(self):
        return self.model.initial_state()

    def update_target(self):
        utils.soft_update(self.target_model, self.model, self.config[constants.ALGO_CONFIG].tau)

    @torch.no_grad()
    def compute_action(self, obs, prev_action, prev_hidden_state, explore, **kwargs):
        self.model.eval()

        # convert obs to tensor
        obs_tensor = utils.convert_to_tensor(obs, self.device)
        obs_tensor = obs_tensor.view(1, -1)

        # convert hidden states to tensor
        hidden_states = [utils.convert_to_tensor(h) for h in prev_hidden_state]

        # get q-values
        q_values, hidden_states = self.model(obs_tensor, hidden_states, **kwargs)

        # apply action mask

        # select action
        action = self.exploration.select_action(
            timestep=self.global_timestep,
            logits=q_values,
            explore=explore,
        )

        return action, [utils.to_numpy(h) for h in hidden_states]

    def learn(self, samples: sample_batch.SampleBatch):
        self.model.train()
