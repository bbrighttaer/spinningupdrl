from core.proto.training_worker_proto import TrainingWorker


class SingleAgentTrainingWorker(TrainingWorker):

    def __init__(self, policy, replay_buffer, config, summary_writer, logger, callback=None):
        self.policy = policy
        self.replay = replay_buffer
        self.config = config
        self.summary_writer = summary_writer
        self.logger = logger
        self.callback = callback

    def train(self, time_step: int):
        ...
