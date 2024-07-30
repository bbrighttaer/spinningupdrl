import abc
import typing

from core import constants, metrics
from core.buffer import sample_batch

LearningStats = typing.Dict[metrics.LearningMetrics, typing.Any]


class Policy(abc.ABC):

    def __init__(self, config, summary_writer, logger):
        self.config = config
        self.obs_size = config[constants.ENV_CONFIG].obs_size
        self.act_size = config[constants.ENV_CONFIG].act_size
        self.device = config[constants.RUNNING_CONFIG].device
        self.summary_writer = summary_writer,
        self.logger = logger
        self.global_timestep = 0

    @abc.abstractmethod
    def get_initial_hidden_state(self):
        ...

    def on_global_timestep_update(self, t):
        self.global_timestep = t

    @abc.abstractmethod
    def compute_action(self, obs, prev_action, prev_hidden_state, explore, **kwargs):
        ...

    @abc.abstractmethod
    def learn(self, samples: sample_batch.SampleBatch) -> LearningStats:
        ...

    @abc.abstractmethod
    def get_weights(self) -> typing.Dict:
        ...

    @abc.abstractmethod
    def set_weights(self, weights):
        ...
