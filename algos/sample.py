import random

from algos.policy import Policy
from core.buffer import sample_batch


class SamplePolicy(Policy):

    def get_initial_hidden_state(self):
        return []

    def compute_action(self, obs, prev_action, prev_hidden_state, **kwargs):
        act = random.randint(0, self.act_size - 1)
        return act, []

    def learn(self, samples: sample_batch.SampleBatch):
        return {}
