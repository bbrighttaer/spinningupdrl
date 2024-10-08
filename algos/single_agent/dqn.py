from functools import partial

import numpy as np
import torch

from algos import Policy
from algos.policy import LearningStats
from core import constants, utils, EPS, metrics
from core.buffer import sample_batch
from core.exploration import EpsilonGreedy
from core.modules.mlp import SimpleFCNet
from core.modules.rnn import SimpleRNN


class DQNPolicy(Policy):
    """
    Single-agent DQN Policy
    """

    def __init__(self, config, summary_writer, logger, policy_id=None):
        super().__init__(config, summary_writer, logger, policy_id)
        # create model
        if self.model_config.core_arch == "mlp":
            self.model = SimpleFCNet(self.model_config).to(self.device)
            self.target_model = SimpleFCNet(self.model_config).to(self.device)
        elif self.model_config.core_arch == "rnn":
            self.model = SimpleRNN(self.model_config).to(self.device)
            self.target_model = SimpleRNN(self.model_config).to(self.device)
        else:
            raise RuntimeError("Core arch should be either gru or mlp")

        # create optimizers
        self.params = list(self.model.parameters())
        if self.algo_config.optimizer == "rmsprop":
            from torch.optim import RMSprop
            self.optimizer = RMSprop(params=self.params, lr=self.algo_config.learning_rate)
        elif self.algo_config.optimizer == "adam":
            from torch.optim import Adam
            self.optimizer = Adam(params=self.params, lr=self.algo_config.learning_rate)
        else:
            raise RuntimeError(f"Unsupported optimizer: {self.algo_config.optimizer}")

        # exploration or action selection strategy
        self.exploration = EpsilonGreedy(
            initial_epsilon=self.algo_config.epsilon,
            final_epsilon=self.algo_config.final_epsilon,
            epsilon_timesteps=self.algo_config.epsilon_timesteps,
        )

        # trigger initial network sync
        self.update_target()
        self._last_target_update = 0
        self._training_count = 0

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
                utils.numpy_state_dict_to_tensor_state_dict(weights["target_model"], self.device)
            )

    @torch.no_grad()
    def compute_action(self, obs, prev_action, prev_hidden_state, explore, state, **kwargs):
        self.model.eval()

        # convert obs to tensor
        obs_tensor = utils.convert_to_tensor(obs, self.device)
        obs_tensor = obs_tensor.view(1, -1)

        # convert hidden states to tensor
        hidden_states = [utils.convert_to_tensor(h, self.device) for h in prev_hidden_state]

        # get q-values
        q_values, hidden_states = self.model(obs_tensor, hidden_states, **kwargs)

        # apply action mask
        if self.action_mask_size > 0 and constants.ACTION_MASK in kwargs:
            avail_actions = utils.convert_to_tensor(kwargs[constants.ACTION_MASK], self.device)
            avail_actions = avail_actions.view(*q_values.shape)
            masked_q_values = q_values.clone()
            masked_q_values[avail_actions == 0.0] = -float("inf")
            final_q_values = masked_q_values
        else:
            final_q_values = q_values

        # select action
        action = self.exploration.select_action(
            timestep=self.global_timestep,
            logits=final_q_values,
            explore=explore,
        )

        return action, [utils.tensor_to_numpy(h) for h in hidden_states]

    def learn(self, samples: sample_batch.SampleBatch, **kwargs) -> LearningStats:
        self.model.train()
        self._training_count += 1
        algo_config = self.config[constants.ALGO_CONFIG]

        # set a get interceptor to convert values to tensor on retrieval
        samples.set_get_interceptor(partial(utils.convert_to_tensor, device=self.device))

        # training data
        obs = samples[constants.OBS]
        actions = samples[constants.ACTION]
        rewards = samples[constants.REWARD]
        rewards = rewards.float()
        next_obs = samples[constants.NEXT_OBS]
        dones = samples[constants.DONE].long()
        weights = samples[constants.WEIGHTS]
        seq_mask = (~samples[constants.SEQ_MASK]).long()
        if constants.NEXT_ACTION_MASK in samples:
            next_action_mask = samples[constants.NEXT_ACTION_MASK]
        else:
            next_action_mask = None
        B, T = obs.shape[:2]

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
        # if action mask is present avoid selecting these actions
        if self.action_mask_size > 0 and next_action_mask is not None:
            ignore_action_tp1 = (next_action_mask == 0) & (seq_mask == 1)
            target_mac_out_tp1[ignore_action_tp1] = -np.inf
        target_q_values = torch.max(target_mac_out_tp1, dim=2)[0]
        target_q_values = target_q_values.unsqueeze(dim=2)

        # compute targets
        targets = rewards + (1 - dones) * algo_config.gamma * target_q_values
        targets = targets.detach()

        # one step TD error
        td_error = targets - q_values
        masked_td_error = seq_mask * td_error
        weights = weights.view(B, T, -1)
        loss = weights * masked_td_error ** 2
        seq_mask_sum = seq_mask.sum()
        loss = loss.sum() / (seq_mask_sum + EPS)

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm_clipping_ = algo_config.grad_clip
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, grad_norm_clipping_)
        self.optimizer.step()

        # target model update
        if self._training_count > algo_config.target_update_freq + self._last_target_update:
            self.update_target()
            self._last_target_update = self._training_count

        # metrics gathering
        mask_elems = seq_mask_sum.item()
        return {
            metrics.LearningMetrics.TRAINING_LOSS: loss.item(),
            metrics.LearningMetrics.GRAD_NORM: grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            metrics.LearningMetrics.TD_ERROR_ABS: masked_td_error.abs().sum().item() / mask_elems,
            metrics.LearningMetrics.Q_TAKEN_MEAN: (q_values * seq_mask).sum().item() / mask_elems,
            metrics.LearningMetrics.TARGET_MEAN: (targets * seq_mask).sum().item() / mask_elems,
            constants.TD_ERRORS: utils.tensor_to_numpy(td_error)
        }
