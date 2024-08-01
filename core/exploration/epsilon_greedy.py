import random

import torch

from core import FLOAT_MIN
from core.proto.exploration_proto import Exploration
from core.schedules.piecewise_schedule import PiecewiseSchedule


class EpsilonGreedy(Exploration):
    """
    Implements the epsilon-greedy action selection method
    """

    def __init__(self, initial_epsilon: float, final_epsilon: float, epsilon_timesteps: int):
        self.epsilon_schedule = PiecewiseSchedule(
            endpoints=[
                (0, initial_epsilon),
                (epsilon_timesteps, final_epsilon),
            ],
            outside_value=final_epsilon,
        )

    def select_action(self, timestep, logits=None, probs=None, explore: bool = True):
        assert logits is not None or probs is not None

        # create distribution for sampling from logits or probs
        dist = torch.distributions.Categorical(probs=probs, logits=logits)

        # get current epsilon
        if explore:
            epsilon = self.epsilon_schedule.value(timestep)
        else:
            epsilon = 0.

        # exploration vs exploitation
        if random.random() < epsilon:
            q_values = logits
            # mask out actions, whose Q-values are -inf, so that we don't
            # even consider them for exploration.
            random_valid_action_logits = torch.where(
                q_values <= FLOAT_MIN, torch.ones_like(q_values) * 0.0, torch.ones_like(q_values)
            )
            # Select a random action.
            action = torch.multinomial(random_valid_action_logits, 1).item()
        else:
            action = dist.probs.argmax().item()

        return action
