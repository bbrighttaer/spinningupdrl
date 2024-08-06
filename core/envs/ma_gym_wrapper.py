import gymnasium as gym
import gym as old_gym

from core import constants, utils


def _create_gymnasium_space(space_obj):
    if isinstance(space_obj, old_gym.spaces.Box):
        return gym.spaces.Box(space_obj.low, space_obj.high, space_obj.shape)
    elif isinstance(space_obj, old_gym.spaces.Discrete):
        return gym.spaces.Discrete(space_obj.n)


class MAGymEnvWrapper(old_gym.Wrapper):
    """
    Wraps environments from the ma_gym library
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = _create_gymnasium_space(env.action_space[0])
        obs_space = env.observation_space
        self.observation_space = gym.spaces.Dict({
            f"agent_{i}": gym.spaces.Dict({
                constants.OBS: _create_gymnasium_space(ob)
            }) for i, ob in enumerate(obs_space)
        })
        self.num_agents = env.n_agents
        setattr(env.unwrapped, "get_env_info", self.get_env_info)

    def reset(self, **kwargs):
        multi_agent_obs = super().reset()
        obs_dict = {
            f"agent_{i}": {"obs": utils.to_numpy_array(multi_agent_obs[i])} for i in range(self.num_agents)
        }
        return obs_dict, {}

    def step(self, actions_dict):
        actions = list(actions_dict.values())
        obs_n, reward_n, done_n, info = super().step(actions)
        info_wrapped = {}
        obs_wrapped = {}
        rewards_wrapped = {}
        dones_wrapped = {}
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            info_wrapped[agent_id] = info
            obs_wrapped[agent_id] = {"obs": utils.to_numpy_array(obs_n[i])}
            rewards_wrapped[agent_id] = reward_n[i]
            dones_wrapped[agent_id] = done_n[i]
        return obs_wrapped, rewards_wrapped, dones_wrapped, dones_wrapped, info_wrapped

    def get_env_info(self):
        env_info = {
            constants.OBS: self.observation_space,
            constants.ENV_ACT_SPACE: self.action_space,
            constants.ENV_NUM_AGENTS: self.num_agents,
        }
        return env_info
