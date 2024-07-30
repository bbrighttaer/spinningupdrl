from functools import partial

import torch

from algos import Policy
from algos.policy import LearningStats
from core import constants, utils, EPS, metrics
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
        self.params = list(self.model.parameters())
        if algo_config.optimizer == "rmsprop":
            from torch.optim import RMSprop
            self.optimizer = RMSprop(params=self.params, lr=algo_config.learning_rate)
        elif algo_config.optimizer == "adam":
            from torch.optim import Adam
            self.optimizer = Adam(params=self.params, lr=algo_config.learning_rate)
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
        self._last_target_update_ts = 0

    def get_initial_hidden_state(self):
        return self.model.get_initial_state()

    def update_target(self):
        utils.soft_update(self.target_model, self.model, self.config[constants.ALGO_CONFIG].tau)

    def get_weights(self):
        return {
            "model": utils.tensor_state_dict_to_numpy_state_dict(self.model.state_dict()),
            "target_model": utils.tensor_state_dict_to_numpy_state_dict(self.target_model.state_dict())
        }

    def set_weights(self, weights):
        if "model" in weights:
            self.model.load_state_dict(
                utils.numpy_state_dict_to_tensor_state_dict(weights["model"], self.device)
            )
        if "target_model" in weights:
            self.target_model.load_state_dict(
                utils.numpy_state_dict_to_tensor_state_dict(weights["model"], self.device)
            )

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

    def learn(self, samples: sample_batch.SampleBatch) -> LearningStats:
        self.model.train()
        algo_config = self.config[constants.ALGO_CONFIG]

        # set a get interceptor to convert values to tensor on retrieval
        samples.set_get_interceptor(partial(utils.convert_to_tensor, device=self.device))

        # training data
        obs = samples[constants.OBS]
        actions = samples[constants.ACTION]
        rewards = samples[constants.REWARD]
        next_obs = samples[constants.NEXT_OBS]
        dones = samples[constants.DONE]
        seq_mask = ~samples[constants.MASK]

        # reward normalization
        if algo_config.reward_normalization:
            rewards = (rewards - rewards.mean()) / (rewards.std() + EPS)

        # put all observations together for convenience
        whole_obs = torch.cat((obs[:, 0:1], next_obs), dim=1)

        # get q-values for all experiences
        mac_out = utils.unroll_mac(self.model, whole_obs)
        target_mac_out = utils.unroll_mac(self.target_model, whole_obs)

        # main model q-values
        q_values = torch.gather(mac_out[:, :-1], dim=2, index=actions)

        # target model q-values
        target_mac_out_tp1 = target_mac_out[:, 1:]
        target_q_values = torch.max(target_mac_out_tp1, dim=2)[0]
        target_q_values = target_q_values.unsqueeze(dim=2)

        # compute targets
        targets = rewards + (1 - dones) * algo_config.gamma * target_q_values
        targets = targets.detach()

        # one step TD error
        td_error = targets - q_values
        masked_td_error = seq_mask * td_error
        loss = masked_td_error ** 2
        loss = loss.sum() / (seq_mask.sum() + EPS)

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm_clipping_ = algo_config.grad_clip
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, grad_norm_clipping_)
        self.optimizer.step()

        # target model update
        if self.global_timestep > algo_config.target_update_freq + self._last_target_update_ts:
            self.update_target()
            self._last_target_update_ts = self.global_timestep

        # metrics gathering
        mask_elems = seq_mask.sum().item()
        return {
            metrics.LearningMetrics.TRAINING_LOSS: loss.item(),
            metrics.LearningMetrics.GRAD_NORM: grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            metrics.LearningMetrics.TD_ERROR_ABS: masked_td_error.abs().sum().item() / mask_elems,
            metrics.LearningMetrics.Q_TAKEN_MEAN: (q_values * seq_mask).sum().item() / mask_elems,
            metrics.LearningMetrics.TARGET_MEAN: (targets * seq_mask).sum().item() / mask_elems,
        }







