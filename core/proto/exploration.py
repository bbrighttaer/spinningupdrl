from typing import Protocol


class Exploration(Protocol):

    def select_action(self, timestep, logits=None, probs=None, explore: bool = True):
        ...
