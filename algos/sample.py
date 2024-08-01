import random
import typing

from algos.policy import Policy
from core.buffer import sample_batch


class SamplePolicy(Policy):

    def get_weights(self) -> typing.Dict:
        return {}

    def set_weights(self, weights):
        pass

    def get_initial_hidden_state(self):
        return []

    def compute_action(self, obs, prev_action, prev_hidden_state, state, **kwargs):
        act = random.randint(0, self.n_actions - 1)
        return act, []

    def learn(self, samples: sample_batch.SampleBatch):
        return {}
