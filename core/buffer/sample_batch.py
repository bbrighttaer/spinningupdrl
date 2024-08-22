from typing import Callable

import numpy as np
import torch

from core.buffer.episode import Episode


class SampleBatch(dict):
    """
    Holds samples of episodes
    """

    def __init__(self, *args, **kwargs):
        batch = kwargs.pop("batch", [])
        super().__init__(*args, **kwargs)
        if batch:
            batch_ep = Episode()
            for ep in batch:
                batch_ep.add(**ep.data)
            batch_ep = batch_ep.merge_time_steps()
            self.update({k: v[0] for k, v in batch_ep.data.items()})
        self._intercepted = {}
        self._interceptor = None

    def set_get_interceptor(self, interceptor: Callable[[np.ndarray], torch.Tensor]):
        self._interceptor = interceptor

    def slice_multi_agent_batch(self, i):
        """
        Selects the samples of agent i along the agent axis.
        This assumes that each key in the multi-agent batch is structured as [N, T, num_agents,...].
        Keys whose values cannot be sliced (due to being common across agents) are left as is.

        :param i: agent index
        :return: sample batch for a single agent
        """
        return SampleBatch({
            k: v[:, :, i] if v.ndim > 2 else v for k, v in self.items()
        })

    def __len__(self):
        value = dict.__getitem__(self, list(self.keys())[0])
        return len(value)

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if self._interceptor is not None:
            if key in self._intercepted:
                return self._intercepted[key]
            tensor = self._interceptor(value)
            self._intercepted[key] = tensor
            return tensor
        return value
