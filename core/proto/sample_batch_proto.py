from typing import Protocol


class SampleBatch(Protocol):
    """
    Protocol for SampleBatch class
    """

    def __len__(self):
        ...

    def __getitem__(self, item):
        ...

    def __setitem__(self, key, value):
        ...
