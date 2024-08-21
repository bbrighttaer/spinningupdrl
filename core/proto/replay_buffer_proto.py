from typing import Protocol

from core.proto.episode_proto import Episode
from core.proto.sample_batch_proto import SampleBatch


class ReplayBuffer(Protocol):
    """
    Protocol for replay buffer to store and sample experiences.
    """

    def add(self, episode: Episode):
        ...

    def on_learning_completed(self, *args, **kwargs):
        ...

    def sample(self, batch_size: int, timestep: int) -> SampleBatch:
        ...

    def __len__(self):
        ...
