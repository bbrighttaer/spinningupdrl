from .policy import Policy
from .sample import SamplePolicy
from .dqn import DQNPolicy

ALGO_REGISTRY = {
    "sample": SamplePolicy,
    "dqn": DQNPolicy,
}
