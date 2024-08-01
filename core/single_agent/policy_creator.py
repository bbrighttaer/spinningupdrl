import copy
import typing

import gymnasium as gym

from algos import ALGO_REGISTRY, Policy
from core import constants
from core.buffer.replay_buffer import ReplayBuffer


def SingleAgentPolicyCreator(config, summary_writer, logger) -> typing.Tuple[Policy, ReplayBuffer]:
    config = copy.deepcopy(config)
    env_config = config[constants.ENV_CONFIG]
    env = gym.make(**env_config)
    env_config[constants.OBS] = copy.deepcopy(env.observation_space)
    env_config[constants.ENV_ACT_SPACE] = copy.deepcopy(env.action_space)
    replay_buffer = ReplayBuffer(config[constants.ALGO_CONFIG].buffer_size)
    policy_class = ALGO_REGISTRY[config[constants.ALGO_CONFIG].algo]
    return policy_class(config, summary_writer, logger), replay_buffer
