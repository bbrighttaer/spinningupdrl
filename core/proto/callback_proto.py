from typing import Protocol


class Callback(Protocol):

    def before_episode(self, *args, **kwargs):
        ...

    def after_episode(self, *args, **kwargs):
        ...

    def before_episode_step(self, *args, **kwargs):
        ...

    def after_episode_step(self, *args, **kwargs):
        ...

    def before_training(self, *args, **kwargs):
        ...

    def after_training(self, *args, **kwargs):
        ...
