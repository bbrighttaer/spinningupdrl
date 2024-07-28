import gymnasium as gym

from core import constants
from core.proto.rollout_worker_proto import RolloutWorker


def RolloutWorkerCreator(policy, replay_buffer, config, summary_writer, logger, callback=None) -> RolloutWorker:
    return SimpleRolloutWorker(policy, replay_buffer, config, summary_writer, logger, callback)


class SimpleRolloutWorker(RolloutWorker):

    def __init__(self, policy, replay_buffer, config, summary_writer, logger, callback):
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.config = config
        self.summary_writer = summary_writer
        self.logger = logger
        self.callback = callback
        self._time_step = 0
        self.env = gym.make(**self.config[constants.ENV_CONFIG])

    def get_global_time_step(self):
        return self._time_step

    def generate_trajectory(self):
        obs, info = self.env.reset()
        done = False
        prev_act = 0
        prev_hidden_state = self.policy.initial_hidden_state()

        while not done:
            action = self.policy.compute_action(
                obs=obs,
                prev_action=prev_act,
                prev_hidden_state=prev_hidden_state,
            )
            next_obs, reward, done, truncated, info = self.env.step(action)

            obs = next_obs
            prev_act = action

    def evaluate_policy(self, num_episodes: int):
        pass
