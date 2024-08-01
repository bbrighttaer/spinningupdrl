import abc
import typing

import gymnasium as gym


class MultiAgentEnv(abc.ABC, gym.Env):
    """
    Base class for multi-agent environments.
    Subclasses are meant to wrap around concrete multi-agent environments (e.g. MPE, ma_gym)
    """

    @abc.abstractmethod
    def get_env_info(self) -> typing.Dict[str, typing.Any]:
        ...
