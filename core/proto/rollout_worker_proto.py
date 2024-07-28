from typing import Protocol


class RolloutWorker(Protocol):

    def get_global_time_step(self):
        ...

    def generate_trajectory(self):
        ...

    def evaluate_policy(self, num_episodes: int):
        ...
