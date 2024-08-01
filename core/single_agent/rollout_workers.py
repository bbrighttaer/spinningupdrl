import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from algos import Policy
from core import constants, metrics, utils
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
        self._cur_iter = 0
        self.create_env()

    def create_env(self):
        self.env = gym.make(**self.config[constants.ENV_CONFIG])

    @property
    def timestep(self):
        return self._timestep

    @property
    def cur_iter(self):
        return self._cur_iter

    def _increment_timestep(self):
        self._timestep += 1
        self.policy.on_global_timestep_update(self._timestep)

    def generate_trajectory(self):
        self._cur_iter += 1
        obs, info = self.env.reset()
        state = info.get(constants.STATE)
        done = False
        prev_act = 0
        prev_hidden_state = self.policy.get_initial_hidden_state()
        episode = Episode()
        episode_len = 0
        episode_reward = 0

        while not done and episode_len < self.config[constants.RUNNING_CONFIG].max_timesteps_per_episode:
            action, hidden_state = self.policy.compute_action(
                obs=utils.to_numpy_array(obs),
                prev_action=prev_act,
                prev_hidden_state=prev_hidden_state,
                explore=True,
                state=utils.to_numpy_array(state),
            )
            next_obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated

            # add timestep record to episode/trajectory
            next_state = info.get(constants.NEXT_STATE)
            experience = {
                constants.OBS: obs,
                constants.ACTION: action,
                constants.REWARD: reward,
                constants.NEXT_OBS: next_obs,
                constants.DONE: done,
                constants.PREV_ACTION: prev_act,
                constants.SEQ_MASK: False,
            }
            if state is not None and next_state is not None:
                experience.update({
                    constants.STATE: state,
                    constants.NEXT_STATE: next_state,
                })
            episode.add(**experience)

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

    def evaluate_policy(self, num_episodes: int, render=False):
        eval_episode_lens = []
        eval_episode_rewards = []

        for i in range(num_episodes):
            obs, info = self.env.reset()
            state = info.get("state")
            done = False
            prev_act = 0
            prev_hidden_state = self.policy.get_initial_hidden_state()
            episode_len = 0
            episode_reward = 0

            while not done and episode_len < self.config[constants.RUNNING_CONFIG].max_timesteps_per_episode:
                if render:
                    self.env.render()

                action, hidden_state = self.policy.compute_action(
                    obs=utils.to_numpy_array(obs),
                    prev_action=prev_act,
                    prev_hidden_state=prev_hidden_state,
                    explore=False,
                    state=state,
                )
                next_obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                episode_reward += reward

                # timestep props update
                obs = next_obs
                state = info.get("state")
                prev_act = action
                prev_hidden_state = hidden_state

                episode_len += 1

            eval_episode_lens.append(episode_len)
            eval_episode_rewards.append(episode_reward)

        episode_reward_mean = np.mean(eval_episode_rewards)
        self._log_metrics(
            episode_len=np.mean(eval_episode_lens),
            episode_reward=episode_reward_mean,
            mode=constants.EVALUATION,
        )

        # return flag for terminating the trial if target has been reached
        return self.config[constants.RUNNING_CONFIG].episode_reward_mean_goal <= episode_reward_mean

    def _log_metrics(self, episode_len, episode_reward, mode):
        self.metrics_manager.add_performance_metric(
            data={
                metrics.PerformanceMetrics.EPISODE_REWARD: episode_reward,
                metrics.PerformanceMetrics.EPISODE_LENGTH: episode_len
            },
            cur_iter=self.cur_iter,
            timestep=self.timestep,
            training=mode == constants.TRAINING,
        )
