from typing import Protocol, Optional


class MultiAgentPolicyMapping(Protocol):
    """
    Manages how policy IDs map to agent IDs (from the environment) and vice-versa.
    """

    def find(self, key, reverse: bool = False) -> Optional[str]:
        """

        Arguments
        :param key: The key for the search. It could be an agent ID or a policy ID
        :param reverse: The default behaviour of this function is to map from agent policy ID to agent ID.
                        When reverse is set to `True`, the function maps from agent ID to policy ID.
        :return: corresponding agent ID or policy ID
        """
        ...


