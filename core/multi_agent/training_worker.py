import typing

from torch.utils.tensorboard import SummaryWriter

from algos import Policy
from core import constants
from core.metrics.sim_metrics import MetricsManager
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.training_worker_proto import TrainingWorker


class MultiAgentIndependentTrainingWorker(TrainingWorker):

    def __init__(self,
                 policies: typing.Dict[constants.PolicyID, Policy],
                 replay_buffers: typing.Dict[constants.PolicyID, ReplayBuffer],
                 policy_mapping_fn: typing.Callable[[constants.PolicyID], constants.AgentID],
                 config: dict,
                 logger, callback):
        self.policies = policies
        self.replay_buffers = replay_buffers
        self.policy_mapping_fn = policy_mapping_fn
        self.config = config
        self.logger = logger
        self.callback = callback

    def train(self, timestep: int, cur_iter: int):
        algo_config = self.config[constants.ALGO_CONFIG]

        # Train after `num_steps_to_training` timesteps
        if min([len(b) for b in self.replay_buffers.values()]) < algo_config.replay_start_size:
            return

        # Sample batches for each policy
        multi_agent_batches = {
            p: self.replay_buffers[p].sample(algo_config.training_batch_size) for p in self.policies
        }

        # Perform independent training (todo: can this be paralleled using mp?)
        for policy_id in self.policies:
            policy = self.policies[policy_id]
            replay_buffer = self.replay_buffers[policy_id]
            samples = multi_agent_batches[policy_id]

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
            replay_buffer.on_learning_completed()
