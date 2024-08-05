
import gymnasium as gym

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38
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
