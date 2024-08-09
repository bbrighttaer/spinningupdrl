from core import constants

RUNNING_CONFIG = {
    "total_timesteps": 1000000,
    "logging_steps": 10000,
    "checkpoint_freq": 10000,
    "episode_reward_mean_goal": 1000,
    "max_timesteps_per_episode": 1000,
    "evaluation_interval": 10000,
    "evaluation_num_episodes": 20,
}
ALGO_CONFIG = {
    "buffer_size": 5000,
    "epsilon": 1.0,
    "final_epsilon": 0.05,
    "epsilon_timesteps": 50000,
    "training_batch_size": 32,
    "replay_start_size": 1000,
    "gamma": 0.99,
    "target_update_freq": 200,
    "optimizer": "rmsprop",
    "learning_rate": 0.0005,
    "grad_clip": 10,
    "tau": 0.7,
    "reward_normalization": False,
    "comm_size": 10,  # size or dimension of a message (if communication is enabled)
    "discrete_comm_space_size": 5,  # the size of the message space when using discrete communication
    "msg_aggregation_type": constants.CONCATENATE_MSGS
}
MODEL_CONFIG = {
    "core_arch": "rnn",
    "encoder_layers": [128],
    "hidden_state_dim": 128,
    "hidden_layers": [64],
    "embedding_dim": 8,
    "num_rnn_layers": 1,
    "activation": "relu",
    "dropout_rate": 0.,
}
ENV_CONFIG = {
    "id": "PredatorPrey7x7-v0",
    "penalty": -0.75,
}
