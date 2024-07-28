import copy
import typing

import gymnasium as gym

import algos
from algos import SamplePolicy
from core import constants
from core.buffer.replay_buffer import ReplayBuffer


def SingleAgentPolicyCreator(config, summary_writer, logger) -> typing.Tuple[SamplePolicy, ReplayBuffer]:
    config = copy.deepcopy(config)
    env_config = config[constants.ENV_CONFIG]
    env = gym.make(**env_config)
    env_config["obs_size"] = env.observation_space.shape[0]
    env_config["act_size"] = env.action_space.n
    replay_buffer = ReplayBuffer(config[constants.ALGO_CONFIG].buffer_size)
    return algos.SamplePolicy(config, summary_writer, logger), replay_buffer
