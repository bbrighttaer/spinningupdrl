from collections import Counter
from functools import partial

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset, DataLoader

from algos import Policy
from algos.policy import LearningStats
from core import constants, utils, EPS, metrics, kde
from core.buffer import sample_batch
from core.exploration import EpsilonGreedy
from core.modules.mlp import SimpleFCNet
from core.modules.rnn import SimpleRNN
from core.modules.vae import VariationalAE


def calc_mse_loss(q_values, targets, seq_mask, weights=None):
    # one step TD error
    td_error = targets - q_values
    masked_td_error = seq_mask * td_error
    loss = masked_td_error ** 2
    if weights is not None:
        loss *= weights
    seq_mask_sum = seq_mask.sum()
    loss = loss.sum() / (seq_mask_sum + EPS)
    return loss, masked_td_error, seq_mask_sum


class WBQLPolicy(Policy):
    """
    Weighted Best possible QL
    """

    def __init__(self, config, summary_writer, logger, policy_id=None):
        super().__init__(config, summary_writer, logger, policy_id)
        # create model
        if self.model_config.core_arch == "mlp":
            self.model = SimpleFCNet(self.model_config).to(self.device)
            self.aux_model = SimpleFCNet(self.model_config).to(self.device)
            self.aux_target_model = SimpleFCNet(self.model_config).to(self.device)
        elif self.model_config.core_arch == "rnn":
            self.model = SimpleRNN(self.model_config).to(self.device)
            self.aux_model = SimpleRNN(self.model_config).to(self.device)
            self.aux_target_model = SimpleRNN(self.model_config).to(self.device)
        else:
            raise RuntimeError("Core arch should be either gru or mlp")

        # Autoencoder
        self.vae = VariationalAE(
            input_dim=self.obs_size + 1,
            hidden_layer_dims=self.model_config.vae_hidden_layers,
            latent_dim=self.model_config.vae_latent_dim,
        ).to(self.device)
        self.target_vae = VariationalAE(
            input_dim=self.obs_size + 1,
            hidden_layer_dims=self.model_config.vae_hidden_layers,
            latent_dim=self.model_config.vae_latent_dim,
        ).to(self.device)

        # create optimizers
        self.params = list(self.model.parameters()) + list(self.aux_model.parameters())
        if self.algo_config.optimizer == "rmsprop":
            from torch.optim import RMSprop
            self.optimizer = RMSprop(params=self.params, lr=self.algo_config.learning_rate)
            self.vae_optimizer = RMSprop(params=self.vae.parameters(), lr=self.algo_config.learning_rate)
        elif self.algo_config.optimizer == "adam":
            from torch.optim import Adam
            self.optimizer = Adam(params=self.params, lr=self.algo_config.learning_rate)
            self.vae_optimizer = Adam(params=self.vae.parameters(), lr=self.algo_config.learning_rate)
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
        utils.soft_update(self.aux_target_model, self.aux_model, self.config[constants.ALGO_CONFIG].tau)
        utils.soft_update(self.target_vae, self.vae, self.config[constants.ALGO_CONFIG].tau)

    def get_weights(self):
        return {
            "model": utils.tensor_state_dict_to_numpy_state_dict(self.model.state_dict()),
            "aux_model": utils.tensor_state_dict_to_numpy_state_dict(self.aux_model.state_dict()),
            "aux_target_model": utils.tensor_state_dict_to_numpy_state_dict(self.aux_target_model.state_dict()),
            "vae": utils.tensor_state_dict_to_numpy_state_dict(self.vae.state_dict()),
            "target_vae": utils.tensor_state_dict_to_numpy_state_dict(self.target_vae.state_dict())
        }

    def set_weights(self, weights):
        if "model" in weights:
            self.model.load_state_dict(
                utils.numpy_state_dict_to_tensor_state_dict(weights["model"], self.device)
            )
        if "aux_model" in weights:
            self.model.load_state_dict(
                utils.numpy_state_dict_to_tensor_state_dict(weights["aux_model"], self.device)
            )
        if "aux_target_model" in weights:
            self.aux_target_model.load_state_dict(
                utils.numpy_state_dict_to_tensor_state_dict(weights["aux_target_model"], self.device)
            )
        if "vae" in weights:
            self.vae.load_state_dict(
                utils.numpy_state_dict_to_tensor_state_dict(weights["vae"], self.device)
            )
        if "target_vae" in weights:
            self.target_vae.load_state_dict(
                utils.numpy_state_dict_to_tensor_state_dict(weights["target_vae"], self.device)
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

    def _get_weights(self, obs, actions, target_q_values, q_values):
        B, T = obs.shape[:2]
        target_q_values = target_q_values.view(B * T, -1)
        q_values = q_values.view(B * T, -1)
        actions = actions.view(B * T, -1)
        x = torch.cat([obs.view(B * T, -1), actions], dim=-1)

        # update vae parameters
        self.fit_vae(x)

        with torch.no_grad():
            self.vae.eval()

            # select samples
            subset_size = self.algo_config.kde_subset_size or 100

            # compute Q(x)
            # q densities
            outputs, mu, logvar = self.vae.encode(x)
            q_densities = kde.kde_density(outputs, mu, logvar, subset_size)
            q_densities = utils.apply_scaling(q_densities)

            # compute P(x)
            targets_scaled = utils.shift_and_scale(target_q_values)
            target_outputs, target_mu, target_logvar = self.target_vae.encode(x)
            p_densities = targets_scaled / kde.kde_density(target_outputs, target_mu, target_logvar, subset_size)
            p_densities /= (p_densities.max() + EPS)

            # compute weights
            weights = p_densities / q_densities
            weights = utils.apply_scaling(weights)
            weights = weights.view(B, T, 1)
            return weights

    def learn(self, samples: sample_batch.SampleBatch) -> LearningStats:
        self.model.train()
        self.aux_model.train()

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
        seq_mask = (~samples[constants.SEQ_MASK]).long()
        if constants.NEXT_ACTION_MASK in samples:
            next_action_mask = samples[constants.NEXT_ACTION_MASK]
        else:
            next_action_mask = None
        B, T = obs.shape[:2]

        if self.algo_config.show_reward_dist:
            stats = Counter(rewards.view(-1, ).numpy())
            self.summary_writer.add_scalars(
                f"training/{self.policy_id}/reward_dist", {str(k): v for k, v in stats.items()}, self.global_timestep
            )

        # reward normalization
        if algo_config.reward_normalization:
            rewards = (rewards - rewards.mean()) / (rewards.std() + EPS)

        # put all observations together for convenience
        whole_obs = torch.cat((obs[:, 0:1], next_obs), dim=1)

        # get q-values for all experiences
        mac_out = utils.unroll_mac(self.model, whole_obs)
        aux_mac_out = utils.unroll_mac(self.aux_model, whole_obs)
        aux_target_mac_out = utils.unroll_mac(self.aux_target_model, whole_obs)

        # Qe objective
        qe_q_values = torch.gather(aux_mac_out[:, :-1], dim=2, index=actions)
        qi_tp1_q_values = mac_out[:, 1:].clone()
        # if action mask is present avoid selecting unavailable actions
        if self.action_mask_size > 0 and next_action_mask is not None:
            ignore_action_tp1 = (next_action_mask == 0) & (seq_mask == 1)
            qi_tp1_q_values[ignore_action_tp1] = -np.inf
        qi_tp1_q_values = torch.max(qi_tp1_q_values, dim=2)[0]

        qi_tp1_q_values = qi_tp1_q_values.unsqueeze(dim=2)
        qe_targets = rewards + (1 - dones) * algo_config.gamma * qi_tp1_q_values
        qe_loss, qe_masked_td_error, _ = calc_mse_loss(qe_q_values, qe_targets.detach(), seq_mask)

        # Qi objective
        q_values = torch.gather(mac_out[:, :-1], dim=2, index=actions)
        qe_bar_q_values = torch.gather(aux_target_mac_out[:, :-1], dim=2, index=actions)
        weights = self._get_weights(obs, actions, qe_bar_q_values.detach(), q_values.detach())
        qi_loss, masked_td_error, seq_mask_sum = calc_mse_loss(q_values, qe_bar_q_values.detach(), seq_mask, weights)

        # aggregate the two objectives
        loss = qi_loss + qe_loss

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
            metrics.LearningMetrics.TARGET_MEAN: (qe_bar_q_values * seq_mask).sum().item() / mask_elems,
        }

    def vae_loss_function(self, recon_x, x, mu, logvar):
        mse = mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld, mse, kld

    def fit_vae(self, training_data, num_epochs=2):
        dataset = TensorDataset(training_data)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        self.vae.train()
        training_loss = []
        vae_grad_norm = []

        for epoch in range(num_epochs):
            ep_loss = 0

            for batch in data_loader:
                batch = batch[0]
                self.vae_optimizer.zero_grad()
                recon_batch, mu, logvar = self.vae(batch)
                loss, mse, kld = self.vae_loss_function(recon_batch, batch, mu, logvar)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.algo_config.grad_clip)
                ep_loss += loss.item()
                self.vae_optimizer.step()
                vae_grad_norm.append(norm)

            training_loss.append(ep_loss / len(dataset))

        # TB logging
        self.summary_writer.add_scalar(
            f"training/{self.policy_id}/vae_loss",
            np.mean(training_loss), self.global_timestep
        )
        self.summary_writer.add_scalar(
            f"training/{self.policy_id}/vae_grad_norm",
            np.mean(vae_grad_norm), self.global_timestep
        )
