import gymnasium as gym
import numpy as np

from algos import Policy
from core import constants, metrics, utils
from core.buffer.episode import Episode
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.rollout_worker_proto import RolloutWorker


def RolloutWorkerCreator(
        policy: Policy,
        replay_buffer: ReplayBuffer,
        config: dict,
        logger, callback) -> RolloutWorker:
    return SimpleRolloutWorker(policy, replay_buffer, config, logger, callback)


class SimpleRolloutWorker(RolloutWorker):

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

        # record metrics
        self.callback.on_episode_end(
            worker=self,
            is_training=True,
            episode_stats={
                metrics.PerformanceMetrics.EPISODE_LENGTH: episode_len,
                metrics.PerformanceMetrics.EPISODE_REWARD: episode_reward,
            }
        )

    def evaluate_policy(self, num_episodes: int, render=False):
        eval_episode_lens = []
        eval_episode_rewards = []
        eval_episodes = []

        for i in range(num_episodes):
            episode = Episode()
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
                state = info.get("state")
                prev_act = action
                prev_hidden_state = hidden_state

                episode_len += 1

            eval_episode_lens.append(episode_len)
            eval_episode_rewards.append(episode_reward)
            eval_episodes.append(episode)

        episode_reward_mean = np.mean(eval_episode_rewards)

        # record metrics
        self.callback.on_episode_end(
            worker=self,
            is_training=False,
            episode_stats={
                metrics.PerformanceMetrics.EPISODE_LENGTH: np.mean(eval_episode_lens),
                metrics.PerformanceMetrics.EPISODE_REWARD: episode_reward_mean,
            }
        )

        # return flag for terminating the trial if target has been reached
        return self.config[constants.RUNNING_CONFIG].episode_reward_mean_goal <= episode_reward_mean, eval_episodes
