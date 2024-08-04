
RUNNING_CONFIG = {
    "total_timesteps": 10000,
    "logging_steps": 1000,
    "checkpoint_freq": 1000,
    "episode_reward_mean_goal": 1000,
    "max_timesteps_per_episode": 1000,
}
ALGO_CONFIG = {
    "buffer_size": 5000,
    "epsilon": 1.0,
    "final_epsilon": 0.05,
    "epsilon_timesteps": 2000,
    "training_batch_size": 64,
    "replay_start_size": 100,
    "gamma": 0.99,
    "target_update_freq": 100,
    "optimizer": "adam",
    "learning_rate": 0.0005,
    "grad_clip": 10,
    "tau": 0.7,
    "alpha": 0.9,
    "beta": 0.1,
    "reward_normalization": True,
    "show_reward_dist": True,
    "comm_size": 0,  # size or dimension of a message (if communication is enabled)
    "discrete_comm_space_size": 5,  # the size of the message space when using discrete communication
}
MODEL_CONFIG = {
    "core_arch": "mlp",
    "encoder_layers": [64],
    "hidden_state_dim": 64,
    "hidden_layers": [64],
    "embedding_dim": 8,
    "num_rnn_layers": 1,
    "activation": "relu",
    "dropout_rate": 0.,
}
ENV_CONFIG = {
}
