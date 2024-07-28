import collections
import typing

import numpy as np

from core import constants
from core.buffer.episode import Episode


class SampleBatch:
    """
    Houses samples of episodes
    """

    def __init__(self, batch: typing.List[Episode]):
        batch_ep = Episode()
        for ep in batch:
            ep.add(
                obs=ep.obs,
                act=ep.action,
                reward=ep.reward,
                next_obs=ep.next_obs,
                done=ep.done,
                state=ep.state,
                next_state=ep.state,
                prev_action=ep.prev_action,
                mask=ep.mask
            )
        batch_ep = batch_ep.deflate()
        self._data = {
            constants.OBS: batch_ep.obs,
            constants.ACTION: batch_ep.action,
            constants.REWARD: batch_ep.reward,
            constants.NEXT_OBS: batch_ep.next_obs,
            constants.DONE: batch_ep.done,
            constants.STATE: batch_ep.state,
            constants.PREV_ACTION: batch_ep.prev_action,
            constants.MASK: batch_ep.mask,
        }
        self._count = len(batch_ep)

    def __len__(self):
        return self._count

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value
