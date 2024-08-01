import numpy as np
from gymnasium import spaces

from core import constants
from core.envs.base_env import MultiAgentEnv


class OneStepMultiAgentCoopMatrixGame(MultiAgentEnv):
    """
    Implements a simple one-step cooperative matrix game following QTran paper
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.agents = ["agent_0", "agent_1"]
        self.observation_space = spaces.Dict({a: spaces.Dict({
            constants.OBS: spaces.Discrete(2),
        }) for a in self.agents})
        self.num_agents = len(self.agents)
        self.step_count = 0
        self.max_steps = 1
        # Payoff matrix for game 1
        self.payoff = np.array([
            [8, -12, -12],
            [-12, 0, 0],
            [-12, 0, 0],
        ])

    def reset(self, **kwargs):
        self.step_count = 0
        obs = {agent: {constants.OBS: 0} for agent in self.agents}
        return obs, {}

    def step(self, actions):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise ValueError("All steps already taken")

        if len(actions) != len(self.agents):
            raise ValueError("Number of actions must match the number of agents")

        # Calculate payoff based on the chosen game and actions of both agents
        payoffs = [
            self.payoff[actions["agent_0"]][actions["agent_1"]],
            self.payoff[actions["agent_0"]][actions["agent_1"]],
        ]

        # Observations after taking actions
        obs = {agent: {constants.OBS: 1} for agent in self.agents}

        # Return observations, global payoff, done and truncated flags, and info dict
        rewards = {agent: reward for agent, reward in zip(self.agents, payoffs)}
        info = {agent: {} for agent in self.agents}
        return obs, rewards, {"__all__": True}, {"__all__": True}, info

    def get_env_info(self):
        env_info = {
            constants.OBS: self.observation_space,
            constants.ENV_ACT_SPACE: self.action_space,
            constants.ENV_NUM_AGENTS: self.num_agents,
        }
        return env_info
