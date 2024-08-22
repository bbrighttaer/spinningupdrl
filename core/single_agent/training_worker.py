from algos import Policy
from core import constants
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.training_worker_proto import TrainingWorker


class SingleAgentTrainingWorker(TrainingWorker):

    def __init__(self,
                 policy: Policy,
                 replay_buffer: ReplayBuffer,
                 config: dict,
                 logger, callback):
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.config = config
        self.logger = logger
        self.callback = callback

    def train(self, timestep: int, cur_iter: int, eval_episodes: list):
        algo_config = self.config[constants.ALGO_CONFIG]

        # Train after `num_steps_to_training` timesteps
        if len(self.replay_buffer) < algo_config.replay_start_size:
            return

        # Sample a batch from the buffer
        samples = self.replay_buffer.sample(algo_config.training_batch_size, timestep)

        # Train using samples
        learning_stats = self.policy.learn(samples)
        td_error = learning_stats.pop(constants.TD_ERRORS)

        # update metrics
        self.callback.on_learn_on_batch_end(
            policy=self.policy,
            cur_iter=cur_iter,
            timestep=timestep,
            learning_stats=learning_stats,
        )

        kwargs = {
            constants.TD_ERRORS: td_error,
            "sample_batch": samples,
        }

        # Replay buffer callback (e.g. on-policy buffer can be cleared at this stage)
        self.replay_buffer.on_learning_completed(**kwargs)
