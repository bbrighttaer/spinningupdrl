import abc
import typing

import numpy as np
from gymnasium import spaces

from core import constants, metrics
from core.buffer import sample_batch

LearningStats = typing.Dict[metrics.LearningMetrics, typing.Any]


def get_size(gym_space: spaces.Space, obs_check: bool) -> int | typing.Dict | typing.Tuple[int]:
    if gym_space is None:
        return 0
    elif isinstance(gym_space, spaces.Discrete):
        return (1, int(gym_space.n)) if obs_check else int(gym_space.n)
    elif isinstance(gym_space, spaces.Box):
        return int(np.prod(gym_space.shape))
    elif isinstance(gym_space, spaces.Dict):
        return {
            env_space: get_size(gym_space[env_space], obs_check) for env_space in gym_space
        }
    else:
        raise RuntimeError(f"Action/Observation space should either be discrete or box, not {type(gym_space)}")


class Policy(abc.ABC):

    def __init__(self, config, summary_writer, logger, policy_id):
        self.config = config
        self.algo_config = config[constants.ALGO_CONFIG]
        self.model_config = config[constants.MODEL_CONFIG]
        self.device = config[constants.RUNNING_CONFIG].device
        self.summary_writer = summary_writer
        self.logger = logger
        self.policy_id = policy_id  # only relevant in the multi-agent case
        self.global_timestep = 0

        # action space should either be discrete or box
        self.act_space = config[constants.ENV_CONFIG][constants.ENV_ACT_SPACE]
        self.n_actions = get_size(self.act_space, False)
        if not isinstance(self.n_actions, int):
            raise RuntimeError("Number of actions should be an integer")

        # observation space
        # Supported types are discrete, box and dict.
        # If type is spaces.Dict, the expected keys are obs, action mask, and state.
        self.action_mask_size = 0
        self.state_size = 0
        self.obs_size = 0
        self.num_discrete_obs = None
        self.obs_space = config[constants.ENV_CONFIG][constants.OBS]
        res = get_size(self.obs_space, True)
        if isinstance(res, int):
            self.obs_size = res
        elif isinstance(res, tuple):
            self.obs_size, self.num_discrete_obs = res
        elif isinstance(res, dict):
            self.obs_size = res[constants.OBS]
            if isinstance(self.obs_size, tuple):
                self.obs_size, self.num_discrete_obs = self.obs_size
            self.state_size = res.get(constants.ENV_STATE_SPACE) or 0
            self.action_mask_size = res.get(constants.ACTION_MASK) or 0

        # update configs
        self.model_config["state_size"] = self.state_size
        self.model_config["obs_size"] = self.obs_size
        self.model_config["n_actions"] = self.n_actions
        self.model_config["num_discrete_obs"] = self.num_discrete_obs
        self.model_config["comm_size"] = self.algo_config.comm_size
        self.model_config["n_agents"] = config[constants.ENV_CONFIG][constants.ENV_NUM_AGENTS]
        self.model_config["discrete_comm_space_size"] = self.algo_config.discrete_comm_space_size
        self.model_config["msg_aggregation_type"] = self.algo_config.msg_aggregation_type

    @abc.abstractmethod
    def get_initial_hidden_state(self):
        ...

    def on_global_timestep_update(self, t):
        self.global_timestep = t

    @abc.abstractmethod
    def compute_action(self, obs, prev_action, prev_hidden_state, explore, state, **kwargs):
        ...

    @abc.abstractmethod
    def learn(self, samples: sample_batch.SampleBatch) -> LearningStats:
        ...

    @abc.abstractmethod
    def get_weights(self) -> typing.Dict:
        ...

    @abc.abstractmethod
    def set_weights(self, weights):
        ...

    def get_initial_message(self) -> typing.List | constants.NDArray:
        comm_size = self.algo_config.comm_size
        msg = np.zeros((comm_size,)) if comm_size else []
        return msg

    def get_message(
            self, obs, state, prev_msg, **kwargs
    ) -> typing.List | constants.NDArray:
        return self.get_initial_message()
