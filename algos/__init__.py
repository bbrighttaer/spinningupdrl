from .policy import Policy
from .sample import SamplePolicy
from .dqn import DQNPolicy
from .bql import BQLPolicy
from .hiql import HIQLPolicy
from .wbql import WBQLPolicy
from .iql_com import IQLCommPolicy
from .wiql_com import WIQLCommPolicy

ALGO_REGISTRY = {
    "sample": SamplePolicy,
    "dqn": DQNPolicy,
    "bql": BQLPolicy,
    "hiql": HIQLPolicy,
    "iql": DQNPolicy,
    "wbql": WBQLPolicy,
    "iql_comm": IQLCommPolicy,
    "wiql_comm": WIQLCommPolicy,
}
