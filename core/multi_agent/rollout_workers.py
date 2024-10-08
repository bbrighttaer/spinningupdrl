import collections
import typing

import numpy as np

from algos import Policy
from core import constants, metrics, utils
from core.buffer.episode import Episode
from core.multi_agent import execution_strategies
from core.proto.replay_buffer_proto import ReplayBuffer
from core.proto.rollout_worker_proto import RolloutWorker


def RolloutWorkerCreator(
        policies: typing.Dict[constants.PolicyID, Policy],
        replay_buffer: ReplayBuffer,
        policy_mapping_fn: typing.Callable[[constants.PolicyID], constants.AgentID],
        config: dict,
        logger, callback=None) -> RolloutWorker:
    return SimpleMultiAgentRolloutWorker(
        policies, replay_buffer, policy_mapping_fn, config, logger, callback,
    )


def _get_experience_attributes():
    # list of experience elements
    experience_attributes = [
        constants.OBS,
        constants.STATE,
        constants.ACTION_MASK,
        constants.ACTION,
        constants.REWARD,
        constants.NEXT_OBS,
        constants.NEXT_STATE,
        constants.NEXT_ACTION_MASK,
        constants.DONE,
        constants.PREV_ACTION,
        constants.SENT_MESSAGE,
        constants.NEXT_SENT_MESSAGE,
        constants.RECEIVED_MESSAGE,
        constants.NEXT_RECEIVED_MESSAGE,
        constants.TIMESTEP,
        constants.EXPLORATION_FACTOR,
        constants.SEQ_MASK,
    ]
    return experience_attributes


class SimpleMultiAgentRolloutWorker(RolloutWorker):

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
        self.env = utils.make_multi_agent_env(**self.config[constants.ENV_CONFIG])

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
                agent_data = utils.to_numpy_array(env_data[agent_id].get(data_key))
                if agent_data is None:
                    agent_data = []
            else:
                agent_data = None
            data[policy_id] = agent_data
        return data
    def generate_trajectory(self):
        self._cur_iter += 1
        raw_obs, info = self.env.reset()
        obs = self.unpack_env_data(raw_obs, constants.OBS)
        state = self.unpack_env_data(raw_obs, constants.STATE)
        action_mask = self.unpack_env_data(raw_obs, constants.ACTION_MASK)
        done = False
        episode_len = 0
        episode_reward = 0
        episode = Episode()
        prev_act = collections.defaultdict(int)
        prev_hidden_states = {}
        prev_messages = {}
        for policy_id in self.policies:
            policy = self.policies[policy_id]
            prev_hidden_states[policy_id] = policy.get_initial_hidden_state()
            prev_messages[policy_id] = policy.get_initial_message()

        experience_attributes = _get_experience_attributes()

        # Run for an episode
        while not done and episode_len < self.config[constants.RUNNING_CONFIG].max_timesteps_per_episode:
            results = self.timestep_execution_strategy(
                env=self.env,
                obs=obs,
                state=state,
                action_mask=action_mask,
                prev_actions=prev_act,
                prev_hidden_states=prev_hidden_states,
                prev_messages=prev_messages,
                explore=True
            )

            # add timestep record to episode/trajectory of each agent/policy
            experience = {
               exp_attr: [
                   results[exp_attr][policy_id] for policy_id in self.policies
               ] for exp_attr in experience_attributes
            }
            episode.add(**experience)

            # timestep props update
            obs = results[constants.NEXT_OBS]
            state = results[constants.NEXT_STATE]
            action_mask = results[constants.NEXT_ACTION_MASK]
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

        # add episode the buffer (co-incident on the policy/agent axis)
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
        experience_attributes = _get_experience_attributes()
        eval_episodes = []

        for i in range(num_episodes):
            raw_obs, info = self.env.reset()
            obs = self.unpack_env_data(raw_obs, constants.OBS)
            state = self.unpack_env_data(raw_obs, constants.STATE)
            action_mask = self.unpack_env_data(raw_obs, constants.ACTION_MASK)
            done = False
            episode_len = 0
            episode = Episode()
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
                    action_mask=action_mask,
                    state=state,
                    prev_actions=prev_act,
                    prev_hidden_states=prev_hidden_states,
                    prev_messages=prev_messages,
                    explore=False
                )

                # add timestep record to episode/trajectory of each agent/policy
                experience = {
                    exp_attr: [
                        results[exp_attr][policy_id] for policy_id in self.policies
                    ] for exp_attr in experience_attributes
                }
                episode.add(**experience)

                # timestep props update
                obs = results[constants.NEXT_OBS]
                state = results[constants.NEXT_STATE]
                action_mask = results[constants.NEXT_ACTION_MASK]
                prev_act = results[constants.ACTION]
                prev_hidden_states = results[constants.HIDDEN_STATE]
                done = bool(sum(results[constants.DONE].values()))
                prev_messages = results[constants.SENT_MESSAGE]

                # update episode metrics
                episode_len += 1
                rewards_list = list(results[constants.REWARD].values())
                eval_episode_rewards.append(sum(rewards_list))

            eval_episode_lens.append(episode_len)
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
