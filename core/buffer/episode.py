import copy
from collections import defaultdict

import numpy as np

from core import constants
from core.proto.episode_proto import Episode as EpisodeProto


class Episode(EpisodeProto):
    """
    Houses the experiences within a trajectory
    """

    def __init__(self):
        self._episode_data = defaultdict(list)

    @property
    def data(self):
        # return a copy, to avoid any potential tampering
        return copy.deepcopy(self._episode_data)

    def add(self, **experience):
        """
        Adds an episode/experience record.
        """
        def wrap(obj):
            return obj if isinstance(obj, np.ndarray) or isinstance(obj, list) else [obj]

        for key, value in experience.items():
            self._episode_data[key].append(wrap(value))

    def __len__(self):
        return len(self._episode_data[list(self._episode_data.keys())[0]])

    def pad_episode(self, size) -> "Episode":
        """
        Pads a copy of this episode by the given size (sequence padding).

        Arguments
        ----------
        :param size: padding size
        :return: Padded copy
        """
        ep_copy = copy.deepcopy(self)

        for i in range(size):
            pad_entry = {}
            for key, val_list in self._episode_data.items():
                pad = np.zeros_like(val_list[-1])
                # since the expected `seq_mask` value for actual experiences is `False`,
                # the value is negated for the padding if the `seq_mask` field is encountered.
                pad_entry[key] = ~pad if key == constants.SEQ_MASK else pad
            ep_copy.add(**pad_entry)
        return ep_copy

    def merge_time_steps(self) -> 'Episode':
        """
        Merge all episode steps into a single into a single batch.

        :return: Episode with a single batch entry for each property.
        """
        episode = Episode()
        episode.add(**{key: np.stack(value) for key, value in self._episode_data.items()})
        return episode

