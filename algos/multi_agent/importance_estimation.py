import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from core.modules.mlp import WeightNetMLP


class ImportanceEstimation:
    """
    This class implements the importance estimation algorithm.
    It is meant to be used as a mixin for policy learning algorithms.
    """

    def __init__(self):
        # Weight model
        self.weight_model = WeightNetMLP(self.model_config).to(self.device)
        self._lambda = self.algo_config.lamda
        self._update_count = 0
        self._update_freq = 5

        self.params = list(self.weight_model.parameters())
        if self.algo_config.optimizer == "rmsprop":
            from torch.optim import RMSprop
            self.wts_optimizer = RMSprop(params=self.weight_model.parameters(), lr=self.algo_config.learning_rate)
        elif self.algo_config.optimizer == "adam":
            from torch.optim import Adam
            self.wts_optimizer = Adam(params=self.weight_model.parameters(), lr=self.algo_config.learning_rate)
        else:
            raise RuntimeError(f"Unsupported optimizer: {self.algo_config.optimizer}")

    def update_weights_model(self, x_train, x_test, q_vals, seq_mask_train, seq_mask_test) -> float:
        self.weight_model.train()
        num_epochs = 1
        training_loss = []
        for epoch in range(num_epochs):
            self.wts_optimizer.zero_grad()
            objective = self._compute_objective(x_test, x_train, seq_mask_train, seq_mask_test)
            loss = -objective  # maximize the objective by minimizing the negative
            loss.backward()
            self.wts_optimizer.step()
            training_loss.append(loss.item())
        self._update_count += 1
        return np.mean(training_loss)

    def _compute_objective(self, x_test, x_train, tr_mask, te_mask):
        # Forward pass
        w_xq = self.weight_model(x_test)
        w_xp = self.weight_model(x_train)

        # Objective function components
        log_w_xq = torch.log(w_xq)
        sum_log_w_xq = log_w_xq.sum()
        sum_w_xp = w_xp.sum()

        # Lagrangian-based objective
        objective = sum_log_w_xq - self._lambda * (sum_w_xp - x_train.size(0))

        return objective

    @torch.no_grad()
    def get_estimated_weights(self, input_x):
        self.weight_model.eval()
        wts = self.weight_model(input_x)
        return wts
