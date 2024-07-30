from typing import Callable

import numpy as np
import torch

from core import constants
from core.buffer.episode import Episode


class SampleBatch(dict):
    """
    Houses samples of episodes
    """

    def __init__(self, *args, **kwargs):
        batch = kwargs.pop("batch", [])
        super().__init__(*args, **kwargs)
        batch_ep = Episode()
        for ep in batch:
            batch_ep.add(
                obs=ep.obs,
                act=ep.action,
                reward=ep.reward,
                next_obs=ep.next_obs,
                done=ep.done,
                state=ep.state,
                next_state=ep.next_state,
                prev_action=ep.prev_action,
                mask=ep.mask
            )
        batch_ep = batch_ep.merge_time_steps()
        self.update({
            constants.OBS: batch_ep.obs[0],
            constants.ACTION: batch_ep.action[0],
            constants.REWARD: batch_ep.reward[0],
            constants.NEXT_OBS: batch_ep.next_obs[0],
            constants.DONE: batch_ep.done[0],
            constants.STATE: batch_ep.state[0],
            constants.NEXT_STATE: batch_ep.next_state[0],
            constants.PREV_ACTION: batch_ep.prev_action[0],
            constants.MASK: batch_ep.mask[0],
        })
        self._count = len(batch)
        self._intercepted = {}
        self._interceptor = None

    def set_get_interceptor(self, interceptor: Callable[[np.ndarray], torch.Tensor]):
        self._interceptor = interceptor

    def __len__(self):
        return self._count

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if self._interceptor is not None:
            if key in self._intercepted:
                return self._intercepted[key]
            tensor = self._interceptor(value)
            self._intercepted[key] = tensor
            return tensor
        return value
