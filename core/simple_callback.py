from core.proto.callback_proto import Callback


class SimpleCallback(Callback):

    def __init__(self, summary_writer, logger):
        self.summary_writer = summary_writer
        self.logger = logger

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