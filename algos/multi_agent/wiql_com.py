import typing
from functools import partial

import numpy as np
import torch

from algos import Policy
from algos.policy import LearningStats
from core import constants, utils, EPS, metrics, kde, tdw
from core.buffer import sample_batch
from core.exploration import EpsilonGreedy
from core.modules.comm_net import SimpleCommNet
from core.modules.mlp import SimpleFCNet
from core.modules.rnn import SimpleRNN
from core.modules.vae import VariationalAE
from core.schedules.piecewise_schedule import PiecewiseSchedule


class WIQLCommPolicy(Policy):
    """
    Single-agent Wighted-DQN Policy with comm net
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

        # Autoencoders
        # self.vae = VariationalAE(
        #     input_dim=self.obs_size + 1,
        #     hidden_layer_dims=self.model_config.vae_hidden_layers,
        #     latent_dim=self.model_config.vae_latent_dim,
        # ).to(self.device)
        # self.target_vae = VariationalAE(
        #     input_dim=self.obs_size + 1,
        #     hidden_layer_dims=self.model_config.vae_hidden_layers,
        #     latent_dim=self.model_config.vae_latent_dim,
        # ).to(self.device)
        self.tdw_schedule = PiecewiseSchedule(
            endpoints=self.algo_config.tdw_schedule,
            outside_value=self.algo_config.tdw_schedule[-1][-1]  # use value of last schedule
        )

        # create optimizers
        self.params = list(self.model.parameters())
        if self.comm_enabled:
            # comm models
            self.comm_net = SimpleCommNet(self.model_config).to(self.device)
            self.target_comm_net = SimpleCommNet(self.model_config).to(self.device)
            self.params += list(self.comm_net.parameters())
        if self.algo_config.optimizer == "rmsprop":
            from torch.optim import RMSprop
            self.optimizer = RMSprop(params=self.params, lr=self.algo_config.learning_rate)
            # self.vae_optimizer = RMSprop(params=self.vae.parameters(), lr=self.algo_config.learning_rate)
        elif self.algo_config.optimizer == "adam":
            from torch.optim import Adam
            self.optimizer = Adam(params=self.params, lr=self.algo_config.learning_rate)
            # self.vae_optimizer = Adam(params=self.vae.parameters(), lr=self.algo_config.learning_rate)
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
        if self.comm_enabled:
            utils.soft_update(self.target_comm_net, self.comm_net, self.config[constants.ALGO_CONFIG].tau)
        # utils.soft_update(self.target_vae, self.vae, self.config[constants.ALGO_CONFIG].tau)

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
    def get_message(self, obs, state, prev_msg, **kwargs) -> typing.List | constants.NDArray:
        if self.comm_enabled:
            obs = utils.convert_to_tensor(obs, self.device)
            obs = obs.view(1, -1)
            if kwargs.get("use_target"):
                comm_model = self.target_comm_net
            else:
                comm_model = self.comm_net
            comm_model.eval()
            msg, _ = comm_model(obs, [])
            msg = utils.tensor_to_numpy(msg).squeeze()
            return msg
        else:
            return super().get_message(obs, state, prev_msg, **kwargs)

    @torch.no_grad()
    def compute_action(self, obs, prev_action, prev_hidden_state, explore, state, **kwargs):
        self.model.eval()

        # convert obs to tensor
        obs_tensor = utils.convert_to_tensor(obs, self.device)
        obs_tensor = obs_tensor.view(1, -1)

        # add shared messages
        if self.comm_enabled:
            self.comm_net.eval()
            shared_messages = kwargs["shared_messages"]
            messages = np.concatenate(shared_messages, axis=-1)
            messages = utils.convert_to_tensor(messages, self.device).view(1, -1)
            local_msg, _ = self.comm_net(obs_tensor, prev_hidden_state)
            obs_tensor = torch.cat([obs_tensor, local_msg, messages], dim=-1)

        # fingerprint
        if self.algo_config.use_timestep_fingerprint:
            fp = np.array([self.global_timestep, self.exploration.epsilon_schedule.value(self.global_timestep)])
            fp = utils.convert_to_tensor(fp, self.device).view(1, -1)
            obs_tensor = torch.cat([obs_tensor, fp], dim=-1)
        obs_tensor = obs_tensor.float()

        # convert hidden states to tensor
        hidden_states = [utils.convert_to_tensor(h, self.device).float() for h in prev_hidden_state]

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

    def learn(self, samples: sample_batch.SampleBatch) -> LearningStats:
        self.model.train()

        self._training_count += 1
        algo_config = self.config[constants.ALGO_CONFIG]

        # set a get interceptor to convert values to tensor on retrieval
        samples.set_get_interceptor(partial(utils.convert_to_tensor, device=self.device))

        # training data
        obs = samples[constants.OBS]
        actions = samples[constants.ACTION]
        rewards = samples[constants.REWARD].float()
        next_obs = samples[constants.NEXT_OBS]
        dones = samples[constants.DONE].long()
        next_sent_msgs = samples[constants.NEXT_SENT_MESSAGE]
        received_msgs = samples[constants.RECEIVED_MESSAGE]
        next_received_msgs = samples[constants.NEXT_RECEIVED_MESSAGE]
        timestep = samples[constants.TIMESTEP]
        exp_factor = samples[constants.EXPLORATION_FACTOR]
        seq_mask = (~samples[constants.SEQ_MASK]).long()
        if constants.NEXT_ACTION_MASK in samples:
            next_action_mask = samples[constants.NEXT_ACTION_MASK]
        else:
            next_action_mask = None
        B, T = obs.shape[:2]

        # reward normalization
        if algo_config.reward_normalization:
            rewards = (rewards - rewards.mean()) / (rewards.std() + EPS)

        # construct model inputs
        B, T = obs.shape[:2]
        input_x_t = [obs]
        input_x_tp1 = [next_obs]
        if self.comm_enabled:
            self.comm_net.train()
            sent_msgs, _ = self.comm_net(obs, [])
            input_x_t.extend([sent_msgs, received_msgs.view(B, T, -1)])
            input_x_tp1.extend([next_sent_msgs, next_received_msgs.view(B, T, -1)])

        # construct fingerprint
        if self.algo_config.use_timestep_fingerprint:
            fp = torch.cat((timestep, exp_factor), dim=-1).float()
            input_x_t.append(fp)
            input_x_tp1.append(fp)

        input_x_t = torch.cat(input_x_t, dim=-1)
        input_x_tp1 = torch.cat(input_x_tp1, dim=-1)

        # get q-values for all experiences
        mac_out = utils.unroll_mac(self.model, input_x_t)
        target_mac_out_tp1 = utils.unroll_mac(self.target_model, input_x_tp1)

        # main model q-values
        q_values = torch.gather(mac_out, dim=2, index=actions.unsqueeze(2))

        # target model q-values
        # if action mask is present avoid selecting these actions
        if self.action_mask_size > 0 and next_action_mask is not None:
            ignore_action_tp1 = (next_action_mask.view(B, T, -1) == 0) & (seq_mask.view(B, T, -1) == 1)
            target_mac_out_tp1[ignore_action_tp1] = -np.inf
        target_q_values = torch.max(target_mac_out_tp1, dim=2)[0]

        # compute targets
        targets = rewards + (1 - dones) * algo_config.gamma * target_q_values
        targets = targets.detach()

        weights = tdw.target_distribution_weighting(self, targets)

        # one step TD error
        td_error = targets - q_values.unsqueeze(2)
        masked_td_error = seq_mask * td_error
        loss = masked_td_error ** 2
        loss *= weights
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
        }
