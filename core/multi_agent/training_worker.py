import typing

from algos import Policy
from core import constants
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.training_worker_proto import TrainingWorker


class MultiAgentIndependentTrainingWorker(TrainingWorker):

    def __init__(self,
                 policies: typing.Dict[constants.PolicyID, Policy],
                 replay_buffer: ReplayBuffer,
                 policy_mapping_fn: typing.Callable[[constants.PolicyID], constants.AgentID],
                 config: dict,
                 logger, callback):
        self.policies = policies
        self.replay_buffer = replay_buffer
        self.policy_mapping_fn = policy_mapping_fn
        self.config = config
        self.logger = logger
        self.callback = callback

    def train(self, timestep: int, cur_iter: int):
        algo_config = self.config[constants.ALGO_CONFIG]

        # Train after `num_steps_to_training` timesteps
        if len(self.replay_buffer) < algo_config.replay_start_size:
            return

        # Sample a batch from the buffer
        multi_agent_samples = self.replay_buffer.sample(algo_config.training_batch_size)

        # Perform independent training
        for i, policy_id in enumerate(self.policies):
            policy = self.policies[policy_id]
            samples = multi_agent_samples.slice_multi_agent_batch(i)

            # Train using samples
            learning_stats = policy.learn(samples)

            # update metrics
            self.callback.on_learn_on_batch_end(
                policy=policy,
                cur_iter=cur_iter,
                timestep=timestep,
                learning_stats=learning_stats,
                label_suffix=policy_id
            )

            # Replay buffer callback (e.g. on-policy buffer can be cleared at this stage)
            self.replay_buffer.on_learning_completed()
