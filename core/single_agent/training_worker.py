from torch.utils.tensorboard import SummaryWriter

from algos import Policy
from core import constants
from core.metrics.sim_metrics import MetricsManager
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.training_worker_proto import TrainingWorker


class SingleAgentTrainingWorker(TrainingWorker):

    def __init__(self,
                 policy: Policy,
                 replay_buffer: ReplayBuffer,
                 config: dict,
                 summary_writer: SummaryWriter,
                 metrics_manager: MetricsManager,
                 logger, callback=None):
        self.policy = policy
        self.replay = replay_buffer
        self.config = config
        self.summary_writer = summary_writer
        self.metrics_manager = metrics_manager
        self.logger = logger
        self.callback = callback

    def train(self, timestep: int, cur_iter: int):
        algo_config = self.config[constants.ALGO_CONFIG]

        # Train after `num_steps_to_training` time steps
        if timestep < algo_config.num_steps_to_training:
            return

        # Sample a batch from the buffer
        samples = self.replay.sample(algo_config.training_batch_size)

        # Train using samples
        learning_stats = self.policy.learn(samples)

        # update metrics
        self.metrics_manager.add_learning_stats(
            cur_iter=cur_iter,
            timestep=timestep,
            data=learning_stats,
        )

        # Log stats to TB (assumes all are scalars)
        for metric in learning_stats:
            self.summary_writer.add_scalar(
                tag=f"{constants.TRAINING}/{metric}",
                scalar_value=learning_stats[metric],
                global_step=timestep
            )

        # Replay buffer callback (e.g. on-policy buffer can be cleared at this state)
        self.replay.on_learning_completed()

        # Logging
