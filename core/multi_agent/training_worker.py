import typing

import numpy as np

from algos import Policy
from core import constants
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.training_worker_proto import TrainingWorker


class MultiAgentIndependentTrainingWorker(TrainingWorker):
    """
    This trainer assumes that the replay buffer supports concurrent sampling of experiences for agents.
    """

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
        multi_agent_samples = self.replay_buffer.sample(algo_config.training_batch_size, timestep)

        # pop out keys that cannot be sliced for each agent
        if constants.WEIGHTS in multi_agent_samples:
            weights = multi_agent_samples.pop(constants.WEIGHTS)
        else:
            weights = []
        if constants.BATCH_INDEXES in multi_agent_samples:
            batch_indexes = multi_agent_samples.pop(constants.BATCH_INDEXES)
        else:
            batch_indexes = []
        if constants.SEQ_LENS in multi_agent_samples:
            seq_lens = multi_agent_samples.pop(constants.SEQ_LENS)
        else:
            seq_lens = []

        # gather returned td errors
        ma_td_errors = []

        # Perform independent training
        for i, policy_id in enumerate(self.policies):
            policy = self.policies[policy_id]
            samples = multi_agent_samples.slice_multi_agent_batch(i)
            samples[constants.WEIGHTS] = weights

            # Train using samples
            learning_stats = policy.learn(samples)
            td_errors = learning_stats.pop(constants.TD_ERRORS)
            ma_td_errors.append(td_errors)

            # update metrics
            self.callback.on_learn_on_batch_end(
                policy=policy,
                cur_iter=cur_iter,
                timestep=timestep,
                learning_stats=learning_stats,
                label_suffix=policy_id
            )
        ma_td_errors = np.stack(ma_td_errors)
        multi_agent_samples[constants.BATCH_INDEXES] = batch_indexes
        multi_agent_samples[constants.SEQ_LENS] = seq_lens
        kwargs = {
            constants.TD_ERRORS: ma_td_errors,
            "sample_batch": multi_agent_samples,
        }

        # Replay buffer callback (e.g. on-policy buffer can be cleared at this stage)
        self.replay_buffer.on_learning_completed(**kwargs)
