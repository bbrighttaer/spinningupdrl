import typing
from typing import Protocol

from algos import Policy
from core.constants import Number
from core.proto.rollout_worker_proto import RolloutWorker


class Callback(Protocol):

    def before_episode(self, *args, **kwargs):
        ...

    def on_episode_end(
            self, worker: RolloutWorker,
            episode_stats: typing.Dict[str, Number],
            is_training: bool,
            **kwargs,
    ):
        ...

    def before_episode_step(self, *args, **kwargs):
        ...

    def on_timestep_end(self, *args, **kwargs):
        ...

    def before_learn_on_batch(self, *args, **kwargs):
        ...

    def on_learn_on_batch_end(
            self, policy: Policy,
            cur_iter: int,
            timestep: int,
            learning_stats: typing.Dict[str, Number],
            **kwargs,
    ):
        ...

    def before_training(self, *args, **kwargs):
        ...

    def on_training_end(self, *args, **kwargs):
        ...
