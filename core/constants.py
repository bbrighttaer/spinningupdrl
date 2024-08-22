import typing

# config-related constants
RUNNING_CONFIG = "running_config"
ALGO_CONFIG = "algo_config"
MODEL_CONFIG = "model_config"
ENV_CONFIG = "env_config"
CMD_LINE_ARGS = "cmd_line_args"

# GENERAL
TIMESTEP = "timestep"
ITER = "iter"
TOTAL_TIME = "total time (s)"
TRIAL_NAME = "trial name"

# experience-related constants
OBS = "obs"
ACTION = "action"
PREV_ACTION = "prev_action"
NEXT_OBS = "next_obs"
REWARD = "reward"
STATE = "state"
NEXT_STATE = "next_state"
DONE = "done"
SEQ_MASK = "seq_mask"
ACTION_MASK = "action_mask"
NEXT_ACTION_MASK = "next_action_mask"
RECEIVED_MESSAGE = "received_msg"
NEXT_RECEIVED_MESSAGE = "next_received_msg"
SENT_MESSAGE = "sent_msg"
NEXT_SENT_MESSAGE = "next_sent_msg"
HIDDEN_STATE = "hidden_state"
INFO = "info"
EXPLORATION_FACTOR = "exploration_factor"
TD_ERRORS = "td_errors"
WEIGHTS = "weights"
BATCH_INDEXES = "batch_indexes"
SEQ_LENS = "seq_lens"
EVAL_SAMPLE_BATCH = "eval_samples"

# Experiment modes
SINGLE_AGENT = "sa"
MULTI_AGENT = "ma"
MULTI_AGENT_WITH_PARAMETER_SHARING = "ma-ps"
TRAINING = "training"
EVALUATION = "evaluation"
LEARNING = "learning"

# Environment-related values
ENV_ACT_SPACE = "action"
ENV_NUM_AGENTS = "number_of_agents"
ENV_STATE_SPACE = "state"
EPISODE_LIMIT = "episode_limit"

# execution strategies
COMM_BEFORE_ACTION_SELECTION_EXEC_STRATEGY = "comm_before_action"
ACTION_AND_MESSAGE_SELECTION_EXEC_STRATEGY = "simultaneous_action_and_message"
DEFAULT_EXEC_STRATEGY = ACTION_AND_MESSAGE_SELECTION_EXEC_STRATEGY


# Types
NDArray = typing.TypeVar("NDArray")
Number = typing.TypeVar("Number")  # int, float, long
AgentID = str
PolicyID = str

# communication
CONCATENATE_MSGS = "concatenate_received_msgs"
PROJECT_MSGS = "project_shared_msgs"
