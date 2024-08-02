import copy
import typing
from typing import Optional

import gymnasium as gym

from algos import ALGO_REGISTRY, Policy
from core import constants
from core.buffer.replay_buffer import ReplayBuffer
from core.constants import AgentID, PolicyID
from core.envs.base_env import MultiAgentEnv
from core.proto.multi_agent_policy_mapping_proto import MultiAgentPolicyMapping


def MultiAgentIndependentPolicyCreator(
        config, summary_writer, logger
) -> typing.Tuple[typing.Dict[str, Policy], typing.Dict[str, ReplayBuffer], typing.Callable[[AgentID], PolicyID]]:
    config = copy.deepcopy(config)
    env_config = config[constants.ENV_CONFIG]
    env = gym.make(**env_config)
    assert isinstance(env.unwrapped, MultiAgentEnv), "A MultiAgentEnv environment type is required"
    assert isinstance(env.observation_space, gym.spaces.Dict), ("Observation space of multi-agent env should "
                                                                "be gym.spaces.Dict, with agent IDs as the keys")
    env_info = env.unwrapped.get_env_info()
    policy_mapping = {}
    replay_buffer_mapping = {}

    n_agents = env_info[constants.ENV_NUM_AGENTS]
    for i in range(n_agents):
        policy_id = f"policy_{i}"
        env_config[constants.OBS] = copy.deepcopy(env.observation_space[f"agent_{i}"])
        env_config[constants.ENV_ACT_SPACE] = copy.deepcopy(env.action_space)
        buffer_size = config[constants.ALGO_CONFIG].buffer_size
        replay_buffer_mapping[policy_id] = ReplayBuffer(buffer_size, policy_id=policy_id)
        policy_class = ALGO_REGISTRY[config[constants.ALGO_CONFIG].algo]
        policy_mapping[policy_id] = policy_class(config, summary_writer, logger, policy_id=policy_id)

    return policy_mapping, replay_buffer_mapping, IndividualAgentPolicyMapping(n_agents)


class IndividualAgentPolicyMapping(MultiAgentPolicyMapping):

    def __init__(self, num_agents: int):
        self._policy_to_agent = {
            f"policy_{i}": f"agent_{i}" for i in range(num_agents)
        }
        self._reverse = {v: k for k, v in self._policy_to_agent.items()}

    def find(self, key, reverse: bool = False) -> Optional[str]:
        if reverse:
            return self._reverse.get(key)
        else:
            return self._policy_to_agent.get(key)

    def __call__(self, key, reverse=False):
        return self.find(key, reverse)
