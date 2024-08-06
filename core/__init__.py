import sys

import gymnasium as gym
import ma_gym  # noqa

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38
INT_MAX = sys.maxsize
INT_MIN = -sys.maxsize
EPS = 1e-7

# register environments
gym.register(
    id="OneStepMatrixGame-v0",
    entry_point="core.envs:OneStepMultiAgentCoopMatrixGame",
)
gym.register(
    id="SMAC-v0",
    entry_point="core.envs:SMACv0",
)
