import typing

import numpy as np

from core.buffer.episode import Episode
from core.buffer.sample_batch import SampleBatch
from core.proto.replay_buffer_proto import ReplayBuffer as ReplayBufferProto


class ReplayBuffer(ReplayBufferProto):
    """
    Simple off-policy replay buffer to store and sample experiences.
    """

    def __init__(self, capacity: int, policy_id=None):
        self.capacity = capacity
        self._buffer: typing.List[Episode] = []
        self.policy_id = policy_id

    def add(self, episode: Episode):
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(episode)

    def on_learning_completed(self):
        ...

    def sample(self, batch_size: int) -> SampleBatch:
        batch_size = min(batch_size, len(self._buffer))
        sampled_episodes = np.random.choice(self._buffer, batch_size, replace=False)
        max_len = max([len(e) for e in sampled_episodes])
        padded_sample_episodes = []
        for episode in sampled_episodes:
            episode_padded = episode.pad_episode(max_len - len(episode))
            padded_sample_episodes.append(episode_padded)

        sample_batch = SampleBatch(batch=padded_sample_episodes)
        return sample_batch

    def __len__(self):
        return len(self._buffer)
