import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from algos import Policy
from core import constants, metrics
from core.buffer.episode import Episode
from core.metrics.sim_metrics import MetricsManager
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.rollout_worker_proto import RolloutWorker


def RolloutWorkerCreator(
        policy: Policy,
        replay_buffer: ReplayBuffer,
        config: dict,
        summary_writer: SummaryWriter,
         metrics_manager: MetricsManager,
        logger, callback=None) -> RolloutWorker:
    return SimpleRolloutWorker(policy, replay_buffer, config, summary_writer, metrics_manager, logger, callback)


class SimpleRolloutWorker(RolloutWorker):

    def __init__(self,
                 policy: Policy,
                 replay_buffer: ReplayBuffer,
                 config: dict,
                 summary_writer: SummaryWriter,
                 metrics_manager: MetricsManager,
                 logger, callback):
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.config = config
        self.summary_writer = summary_writer
        self.metrics_manager = metrics_manager
        self.logger = logger
        self.callback = callback
        self._timestep = 0
        self.cur_iter = 0
        self.env = gym.make(**self.config[constants.ENV_CONFIG])

    @property
    def timestep(self):
        return self._timestep

    def _increment_timestep(self):
        self._timestep += 1
        self.policy.on_global_timestep_update(self._timestep)

    def generate_trajectory(self):
        self.cur_iter += 1
        obs, info = self.env.reset()
        state = info.get("state") or obs
        done = False
        prev_act = 0
        prev_hidden_state = self.policy.get_initial_hidden_state()
        episode = Episode()
        episode_len = 0
        episode_reward = 0

        while not done and self.config[constants.CMD_LINE_ARGS].total_timesteps:
            action, hidden_state = self.policy.compute_action(
                obs=obs.reshape(-1, 1),
                prev_action=prev_act,
                prev_hidden_state=prev_hidden_state,
                explore=True,
            )
            next_obs, reward, done, truncated, info = self.env.step(action)

            # add timestep record to episode/trajectory
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

            # timestep props update
            obs = next_obs
            state = next_state
            prev_act = action
            prev_hidden_state = hidden_state

            # update global time step and trigger related callbacks
            self._increment_timestep()

            # update episode metrics
            episode_len += 1
            episode_reward += reward

        self.replay_buffer.add(episode)

        self._log_metrics(episode_len, episode_reward, constants.TRAINING)

    def evaluate_policy(self, num_episodes: int):
        eval_episode_lens = []
        eval_episode_rewards = []

        for i in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            prev_act = 0
            prev_hidden_state = self.policy.get_initial_hidden_state()
            episode_len = 0
            episode_reward = 0

            while not done:
                action, hidden_state = self.policy.compute_action(
                    obs=obs.reshape(-1, 1),
                    prev_action=prev_act,
                    prev_hidden_state=prev_hidden_state,
                    explore=False,
                )
                next_obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward

                # timestep props update
                obs = next_obs
                prev_act = action
                prev_hidden_state = hidden_state

                episode_len += 1

            eval_episode_lens.append(episode_len)
            eval_episode_rewards.append(episode_reward)

        self._log_metrics(
            episode_len=np.mean(eval_episode_lens),
            episode_reward=np.mean(eval_episode_rewards),
            mode=constants.EVALUATION,
        )

    def _log_metrics(self, episode_len, episode_reward, mode):
        # logging and metrics
        self.summary_writer.add_scalar(
            tag=f"{mode}/{metrics.PerformanceMetrics.EPISODE_REWARD}",
            scalar_value=episode_reward,
            global_step=self.timestep,
        )
        self.summary_writer.add_scalar(
            tag=f"{mode}/{metrics.PerformanceMetrics.EPISODE_LENGTH}",
            scalar_value=episode_len,
            global_step=self.timestep,
        )
        self.metrics_manager.add_performance_metric(
            data={
                metrics.PerformanceMetrics.EPISODE_REWARD: episode_reward,
                metrics.PerformanceMetrics.EPISODE_LENGTH: episode_len
            },
            cur_iter=self.cur_iter,
            timestep=self.timestep,
            training=mode == constants.TRAINING,
        )
