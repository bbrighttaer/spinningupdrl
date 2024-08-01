from typing import Protocol


class RolloutWorker(Protocol):

    @property
    def timestep(self):
        return 0

    @property
    def cur_iter(self):
        return 0

    def create_env(self):
        ...

    def generate_trajectory(self):
        ...

    def evaluate_policy(self, num_episodes: int, render: bool = False) -> bool:
        ...
