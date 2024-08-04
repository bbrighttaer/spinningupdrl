from .policy import Policy
from .sample import SamplePolicy
from .dqn import DQNPolicy
from .bql import BQLPolicy
from .hiql import HIQLPolicy

ALGO_REGISTRY = {
    "sample": SamplePolicy,
    "dqn": DQNPolicy,
    "bql": BQLPolicy,
    "hiql": HIQLPolicy,
    "iql": DQNPolicy,
}
