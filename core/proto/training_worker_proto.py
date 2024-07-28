from typing import Protocol


class TrainingWorker(Protocol):

    def train(self, time_step: int):
        ...
