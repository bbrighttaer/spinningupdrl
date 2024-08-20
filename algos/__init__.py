from .policy import Policy
from .sample import SamplePolicy
from .single_agent.dqn import DQNPolicy
from .multi_agent.iql import IQLPolicy
from .multi_agent.bql import BQLPolicy
from .multi_agent.hiql import HIQLPolicy
from .multi_agent.wbql import WBQLPolicy
from .multi_agent.iql_com import IQLCommPolicy
from .multi_agent.wiql_com import WIQLCommPolicy
from .multi_agent.wiql import WIQLPolicy

ALGO_REGISTRY = {
    "sample": SamplePolicy,
    "dqn": DQNPolicy,
    "bql": BQLPolicy,
    "hiql": HIQLPolicy,
    "iql": IQLPolicy,
    "wiql": WIQLPolicy,
    "wbql": WBQLPolicy,
    "iql_comm": IQLCommPolicy,
    "wiql_comm": WIQLCommPolicy,
}
