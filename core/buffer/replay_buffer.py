import random
import typing

import numpy as np
import torch

from core import EPS, constants
from core.buffer.episode import Episode
from core.buffer.sample_batch import SampleBatch
from core.buffer.segment_tree import SumSegmentTree, MinSegmentTree
from core.proto.episode_proto import Episode
from core.proto.replay_buffer_proto import ReplayBuffer as ReplayBufferProto
from core.schedules.piecewise_schedule import PiecewiseSchedule
from core.utils import create_sample_batch


class ReplayBuffer(ReplayBufferProto):
    """
    Simple off-policy replay buffer to store and sample experiences.
    """

    def __init__(self, capacity: int, policy_id=None):
        self.capacity = capacity
        self._buffer: typing.List[Episode] = []
        self.policy_id = policy_id
        self._num_timesteps_sampled = 0

    def add(self, episode: Episode):
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(episode)

    # def sample(self, batch_size: int, timestep: int) -> SampleBatch:
    #     batch_size = min(batch_size, len(self._buffer))
    #     sampled_episodes = np.random.choice(self._buffer, batch_size, replace=False)
    #     max_len = max([len(e) for e in sampled_episodes])
    #     padded_sample_episodes = []
    #     for episode in sampled_episodes:
    #         episode_padded = episode.pad_episode(max_len - len(episode))
    #         padded_sample_episodes.append(episode_padded)
    #
    #     sample_batch = SampleBatch(batch=padded_sample_episodes)
    #     return sample_batch

    def sample(self, batch_size: int, timestep: int) -> SampleBatch:
        batch_size = min(batch_size, len(self._buffer))
        idxes = [
            random.randint(0, len(self._buffer) - 1) for _ in range(batch_size)
        ]
        sampled_episodes = []

        batch_indexes = []
        weights = []
        max_len = 0
        for idx in idxes:
            episode = self._buffer[idx]
            sampled_episodes.append(episode)
            count = len(episode)
            max_len = max(max_len, count)
            weights.append([1.0] * count)
            batch_indexes.extend([idx] * count)
            self._num_timesteps_sampled += count

        sample_batch = create_sample_batch(sampled_episodes, weights, batch_indexes, max_len)
        return sample_batch

    def __len__(self):
        return len(self._buffer)


class PrioritizedExperienceReplayBuffer(ReplayBufferProto):
    """
    Prioritized Experience Replay buffer with stochastic eviction.
    Mainly based on RLlib's implementation of PER.
    """

    def __init__(
            self,
            capacity: int,
            alpha: float = 0.6,
            beta: float = 0.1,
            final_beta: float = 1.0,
            beta_annealing_timesteps: float = 1000000,
            policy_id=None,
            stochastic_eviction=False,
    ):
        self.capacity = capacity
        self._alpha = alpha
        assert beta >= 0.0
        self._beta_schedule = PiecewiseSchedule(
            endpoints=[(0, beta), (beta_annealing_timesteps, final_beta)],
            outside_value=final_beta,
        )
        self._buffer: typing.List[Episode] = []
        self.policy_id = policy_id
        self.stochastic_eviction = stochastic_eviction
        it_capacity = 1
        while it_capacity < self.capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._next_idx = 0
        self._num_timesteps_added = 0
        self._num_timesteps_added_wrap = 0
        self._num_timesteps_sampled = 0
        self._eviction_started = False
        self._priorities = np.zeros((capacity,)) + self._max_priority

    def add(self, episode: Episode):
        idx = self._next_idx
        self._add(episode)
        weight = self._max_priority
        self._it_sum[idx] = weight ** self._alpha
        self._it_min[idx] = weight ** self._alpha
        self._priorities[idx] = self._it_sum[idx]

    def _add(self, item):
        self._num_timesteps_added += len(item)

        if self._next_idx < self.capacity and not self._eviction_started:  # buffer is not full
            self._buffer.append(item)
            self._next_idx += 1
        elif self._eviction_started:
            self._buffer[self._next_idx] = item
            self._select_next_index()

        # if next_idx is out of range update it via stochastic eviction
        if self._next_idx >= self.capacity:
            self._eviction_started = True
            self._select_next_index()

    def sample(self, batch_size: int, timestep: int) -> SampleBatch:
        # select indexes based on existing priorities
        idxes = self._sample_proportional(batch_size)

        # gather samples and related properties
        beta = self._beta_schedule.value(timestep)
        sampled_episodes = []
        weights = []
        batch_indexes = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._buffer)) ** (-beta)
        max_len = 0
        for idx in idxes:
            episode = self._buffer[idx]
            sampled_episodes.append(episode)
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._buffer)) ** (-beta)
            count = len(episode)
            max_len = max(max_len, count)
            weights.append([weight / max_weight] * count)
            batch_indexes.extend([idx] * count)
            self._num_timesteps_sampled += count

        sample_batch = create_sample_batch(sampled_episodes, weights, batch_indexes, max_len)
        return sample_batch

    def on_learning_completed(self, *args, **kwargs):
        td_errors = kwargs[constants.TD_ERRORS]
        sample_batch = kwargs["sample_batch"]
        sample_batch.set_get_interceptor(None)
        # in the multi-agent concurrent sampling case, td_errors.ndim is 4
        ag_td_errors = np.mean(td_errors, axis=0) if td_errors.ndim > 3 else td_errors
        ag_td_errors = ag_td_errors.reshape(len(sample_batch), -1)
        seq_lens = sample_batch[constants.SEQ_LENS]
        idxes = sample_batch[constants.BATCH_INDEXES]

        priorities = []
        for i, traj_len in enumerate(seq_lens):
            priorities.extend(ag_td_errors[i, :traj_len].tolist())

        self._update_priorities(idxes, priorities)

    def _update_priorities(self, idxes: typing.List[int], priorities: typing.List[float]):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._buffer)
            priority = np.abs(priority) + EPS
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._priorities[idx] = self._it_sum[idx]
            self._max_priority = max(self._max_priority, priority)

    def _select_next_index(self):
        if self.stochastic_eviction:
            # calc eviction props
            priorities = (1. - (self._priorities / self._priorities.max())) + EPS
            eviction_probs = torch.softmax(torch.from_numpy(priorities), dim=-1).numpy()

            # weighted sampling
            self._next_idx = np.random.choice(np.arange(self.capacity), p=eviction_probs)
        else:
            self._next_idx += 1
            if self._next_idx >= self.capacity:
                self._next_idx = 0

    def _sample_proportional(self, num_items: int) -> typing.List[int]:
        res = []
        for _ in range(num_items):
            mass = random.random() * self._it_sum.sum(0, len(self._buffer))
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def __len__(self):
        return len(self._buffer)
