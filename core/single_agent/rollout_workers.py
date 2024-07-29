import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from algos import Policy
from core import constants
from core.buffer.episode import Episode
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.rollout_worker_proto import RolloutWorker


def RolloutWorkerCreator(
        policy: Policy,
        replay_buffer: ReplayBuffer,
        config: dict,
        summary_writer: SummaryWriter,
        logger, callback=None) -> RolloutWorker:
    return SimpleRolloutWorker(policy, replay_buffer, config, summary_writer, logger, callback)


class SimpleRolloutWorker(RolloutWorker):

    def __init__(self,
                 policy: Policy,
                 replay_buffer: ReplayBuffer,
                 config: dict,
                 summary_writer: SummaryWriter,
                 logger, callback):
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
        state = info.get("state") or obs
        done = False
        prev_act = 0
        prev_hidden_state = self.policy.initial_hidden_state()
        episode = Episode()

        while not done:
            action, hidden_state = self.policy.compute_action(
                obs=obs.reshape(-1, 1),
                prev_action=prev_act,
                prev_hidden_state=prev_hidden_state,
            )
            next_obs, reward, done, truncated, info = self.env.step(action)

            # add time step record to episode/trajectory
            next_state = info.get("next_state") or next_obs
            episode.add(
                obs=obs,
                act=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                state=state,
                next_state=next_state,
                prev_action=prev_act,
            )

            # time step props update
            obs = next_obs
            state = next_state
            prev_act = action
            prev_hidden_state = hidden_state

        self.replay_buffer.add(episode)

    def evaluate_policy(self, num_episodes: int):
        pass
