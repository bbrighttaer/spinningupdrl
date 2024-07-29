from typing import Protocol


class RolloutWorker(Protocol):

    @property
    def timestep(self):
        return 0

    def generate_trajectory(self):
        ...

    def evaluate_policy(self, num_episodes: int):
        ...
