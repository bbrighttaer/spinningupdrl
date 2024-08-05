"""
Adapted from Marllib
"""
import typing

from smac.env.starcraft2.starcraft2 import StarCraft2Env
import numpy as np
from gymnasium.spaces import Dict as GymDict, Discrete, Box

from core import constants
from core.envs.base_env import MultiAgentEnv


class SMACv0(MultiAgentEnv):

    def __init__(self, map_name):
        map_name = map_name if isinstance(map_name, str) else map_name["map_name"]
        self.env = StarCraft2Env(map_name)

        env_info = self.env.get_env_info()
        self.num_agents = self.env.n_agents
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        obs_shape = env_info['obs_shape']
        n_actions = env_info['n_actions']
        state_shape = env_info['state_shape']
        obs_space = {
            "obs": Box(-2.0, 2.0, shape=(obs_shape,)),
            "state": Box(-2.0, 2.0, shape=(state_shape,)),
            "action_mask": Box(-2.0, 2.0, shape=(n_actions,))
        }
        self.observation_space = GymDict({a: GymDict(obs_space) for a in self.agents})
        self.action_space = Discrete(n_actions)

    @property
    def death_tracker_ally(self):
        return self.env.death_tracker_ally

    @property
    def death_tracker_enemy(self):
        return self.env.death_tracker_enemy

    def reset(self, **kwargs):
        self.env.reset()
        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()
        obs_dict = {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs_smac[agent_index]
            state_one_agent = state_smac
            action_mask_one_agent = np.array(self.env.get_avail_agent_actions(agent_index)).astype(np.float32)
            agent_index = "agent_{}".format(agent_index)
            obs_dict[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
                "action_mask": action_mask_one_agent,
            }

        return obs_dict, {}

    def step(self, actions: typing.Dict[str, int]):

        actions_ls = [int(actions[agent_id]) for agent_id in actions.keys()]

        reward, terminated, info = self.env.step(actions_ls)

        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()

        obs_dict = {}
        reward_dict = {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs_smac[agent_index]
            state_one_agent = state_smac
            action_mask_one_agent = np.array(self.env.get_avail_agent_actions(agent_index)).astype(np.float32)
            agent_index = "agent_{}".format(agent_index)
            obs_dict[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
                "action_mask": action_mask_one_agent
            }
            reward_dict[agent_index] = reward

        dones = {"__all__": terminated}

        return obs_dict, reward_dict, dones, dones, {}

    def get_env_info(self):
        env_info = {
            constants.OBS: self.observation_space,
            constants.ENV_ACT_SPACE: self.action_space,
            constants.ENV_NUM_AGENTS: self.num_agents,
            constants.EPISODE_LIMIT: self.env.episode_limit,
        }
        return env_info

    def close(self):
        self.env.close()
