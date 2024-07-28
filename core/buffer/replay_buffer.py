import typing

import numpy as np

from core.buffer.episode import Episode
from core.buffer.sample_batch import SampleBatch


class ReplayBuffer:
    """
    Simple replay buffer to store and sample experiences.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: typing.List[Episode] = []

    def add(self, episode: Episode):
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(episode)

    def clear(self):
        self._buffer.clear()

    def sample(self, batch_size: int) -> SampleBatch:
        batch_size = min(batch_size, len(self._buffer))
        sampled_episodes = np.random.choice(self._buffer, batch_size, replace=False)
        max_len = max([len(e) for e in sampled_episodes])
        padded_sample_episodes = []
        for episode in sampled_episodes:
            episode_padded = episode.pad_episode(max_len - len(episode))
            padded_sample_episodes.append(episode_padded)

        sample_batch = SampleBatch(sampled_episodes)
        return sample_batch

    def __len__(self):
        return len(self._buffer)
