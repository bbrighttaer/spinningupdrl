import abc

from core import constants
from core.buffer import sample_batch


class Policy(abc.ABC):

    def __init__(self, config, summary_writer, logger):
        self.config = config
        self.obs_size = config[constants.ENV_CONFIG].obs_size
        self.act_size = config[constants.ENV_CONFIG].act_size
        self.summary_writer = summary_writer,
        self.logger = logger

    @abc.abstractmethod
    def initial_hidden_state(self):
        ...

    @abc.abstractmethod
    def compute_action(self, obs, prev_action, prev_hidden_state, **kwargs):
        ...

    @abc.abstractmethod
    def learn(self, samples: sample_batch.SampleBatch):
        ...
