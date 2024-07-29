from torch.utils.tensorboard import SummaryWriter

from algos import Policy
from core import constants
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.training_worker_proto import TrainingWorker


class SingleAgentTrainingWorker(TrainingWorker):

    def __init__(self,
                 policy: Policy,
                 replay_buffer: ReplayBuffer,
                 config: dict,
                 summary_writer: SummaryWriter,
                 logger, callback=None):
        self.policy = policy
        self.replay = replay_buffer
        self.config = config
        self.summary_writer = summary_writer
        self.logger = logger
        self.callback = callback

    def train(self, time_step: int):
        running_config = self.config[constants.RUNNING_CONFIG]
        algo_config = self.config[constants.ALGO_CONFIG]

        # Train after `num_steps_to_training` time steps
        if len(self.replay) < algo_config.num_steps_to_training:
            return

        # Sample a batch from the buffer
        samples = self.replay.sample(algo_config.training_batch_size)

        # Train using samples
        stats = self.policy.learn(samples)

        # Logging
