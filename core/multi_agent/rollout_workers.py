import collections
import typing

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from algos import Policy
from core import constants, metrics, utils
from core.buffer.episode import Episode
from core.metrics.sim_metrics import MetricsManager
from core.multi_agent import execution_strategies
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.rollout_worker_proto import RolloutWorker


def RolloutWorkerCreator(
        policies: typing.Dict[constants.PolicyID, Policy],
        replay_buffers: typing.Dict[constants.PolicyID, ReplayBuffer],
        policy_mapping_fn: typing.Callable[[constants.PolicyID], constants.AgentID],
        config: dict,
        summary_writer: SummaryWriter,
        metrics_manager: MetricsManager,
        logger, callback=None) -> RolloutWorker:
    return SimpleMultiAgentRolloutWorker(
        policies, replay_buffers, policy_mapping_fn, config, summary_writer, metrics_manager, logger, callback,
    )


class SimpleMultiAgentRolloutWorker(RolloutWorker):

    def __init__(self,
                 policies: typing.Dict[constants.PolicyID, Policy],
                 replay_buffers: typing.Dict[constants.PolicyID, ReplayBuffer],
                 policy_mapping_fn: typing.Callable[[constants.PolicyID], constants.AgentID],
                 config: dict,
                 summary_writer: SummaryWriter,
                 metrics_manager: MetricsManager,
                 logger, callback):
        self.policies = policies
        self.replay_buffers = replay_buffers
        self.policy_mapping_fn = policy_mapping_fn
        self.config = config
        self.summary_writer = summary_writer
        self.metrics_manager = metrics_manager
        self.logger = logger
        self.callback = callback
        self._timestep = 0
        self._cur_iter = 0
        self.create_env()

        # execution strategy (basically, an extension of this worker)
        exec_strategy = config[constants.ALGO_CONFIG].time_step_exec_strategy
        # if exec_strategy == constants.COMM_BEFORE_ACTION_SELECTION_EXEC_STRATEGY:
        #     strategy_type = execution_strategies.CommBeforeActionSelectionExecStrategy
        # else:
        strategy_type = execution_strategies.DefaultExecutionStrategy
        self.timestep_execution_strategy = strategy_type(self)

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
        for policy_id in self.policies:
            policy = self.policies[policy_id]
            policy.on_global_timestep_update(self._timestep)

    def unpack_env_data(self, env_data, data_key):
        data = {}
        for policy_id in self.policies.keys():
            agent_id = self.policy_mapping_fn(policy_id)
            if env_data is not None:
                agent_data = utils.to_numpy_array(env_data[agent_id][data_key])
            else:
                agent_data = None
            data[agent_id] = agent_data
        return data

    def generate_trajectory(self):
        self._cur_iter += 1
        obs, info = self.env.reset()
        obs = self.unpack_env_data(obs, constants.OBS)
        state = self.unpack_env_data(info.get(constants.STATE), constants.STATE)
        done = False
        episode_len = 0
        episode_reward = 0
        prev_act = collections.defaultdict(int)
        prev_hidden_states = {}
        prev_messages = {}
        policies_episode = {}
        for policy_id in self.policies:
            policy = self.policies[policy_id]
            prev_hidden_states[policy_id] = policy.get_initial_hidden_state()
            prev_messages[policy_id] = policy.get_initial_message()
            policies_episode[policy_id] = Episode()

        # Run for an episode
        while not done and episode_len < self.config[constants.RUNNING_CONFIG].max_timesteps_per_episode:
            results = self.timestep_execution_strategy(
                env=self.env,
                obs=obs,
                state=state,
                prev_actions=prev_act,
                prev_hidden_states=prev_hidden_states,
                prev_messages=prev_messages,
                explore=True
            )

            # add timestep record to episode/trajectory of each agent/policy
            for policy_id in self.policies:
                policy_episode = policies_episode[policy_id]
                agent_id = self.policy_mapping_fn(policy_id)
                experience = {
                    constants.OBS: results[constants.OBS][agent_id],
                    constants.ACTION: results[constants.ACTION][policy_id],
                    constants.REWARD: results[constants.REWARD][agent_id],
                    constants.NEXT_OBS: results[constants.NEXT_OBS][agent_id],
                    constants.DONE: results[constants.DONE][policy_id],
                    constants.PREV_ACTION: prev_act[policy_id],
                    constants.SENT_MESSAGE: results[constants.SENT_MESSAGE][policy_id],
                    constants.RECEIVED_MESSAGE: results[constants.RECEIVED_MESSAGE][policy_id],
                    constants.SEQ_MASK: False,
                }
                state = results[constants.STATE]
                next_state = results[constants.NEXT_STATE]
                if state and next_state:
                    experience.update({
                        constants.STATE: state[agent_id],
                        constants.NEXT_STATE: next_state[agent_id],
                    })
                policy_episode.add(**experience)

            # timestep props update
            obs = results[constants.NEXT_OBS]
            state = results[constants.NEXT_STATE]
            prev_act = results[constants.ACTION]
            prev_hidden_states = results[constants.HIDDEN_STATE]
            done = bool(sum(results[constants.DONE].values()))
            prev_messages = results[constants.SENT_MESSAGE]

            # update global time step and trigger related callbacks
            self._increment_timestep()

            # update episode metrics
            episode_len += 1
            rewards_list = list(results[constants.REWARD].values())
            episode_reward += sum(rewards_list)

        # add episodes the buffer of policies
        for policy_id in self.replay_buffers:
            replay_buffer = self.replay_buffers[policy_id]
            replay_buffer.add(policies_episode[policy_id])

        # record stats
        self._log_metrics(episode_len, episode_reward, constants.TRAINING)

    def evaluate_policy(self, num_episodes: int, render=False):
        eval_episode_lens = []
        eval_episode_rewards = []

        for i in range(num_episodes):
            obs, info = self.env.reset()
            obs = self.unpack_env_data(obs, constants.OBS)
            state = self.unpack_env_data(info.get(constants.STATE), constants.STATE)
            done = False
            episode_len = 0
            episode_reward = 0
            prev_act = collections.defaultdict(int)
            prev_hidden_states = {}
            prev_messages = {}
            policies_episode = {}
            for policy_id in self.policies:
                policy = self.policies[policy_id]
                prev_hidden_states[policy_id] = policy.get_initial_hidden_state()
                prev_messages[policy_id] = policy.get_initial_message()
                policies_episode[policy_id] = Episode()

            while not done and episode_len < self.config[constants.RUNNING_CONFIG].max_timesteps_per_episode:
                if render:
                    self.env.render()

                results = self.timestep_execution_strategy(
                    env=self.env,
                    obs=obs,
                    state=state,
                    prev_actions=prev_act,
                    prev_hidden_states=prev_hidden_states,
                    prev_messages=prev_messages,
                    explore=False
                )

                # timestep props update
                obs = results[constants.NEXT_OBS]
                state = results[constants.NEXT_STATE]
                prev_act = results[constants.ACTION]
                prev_hidden_states = results[constants.HIDDEN_STATE]
                done = bool(sum(results[constants.DONE].values()))
                prev_messages = results[constants.SENT_MESSAGE]

                # update global time step and trigger related callbacks
                self._increment_timestep()

                # update episode metrics
                episode_len += 1
                rewards_list = list(results[constants.REWARD].values())
                episode_reward += sum(rewards_list)

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
