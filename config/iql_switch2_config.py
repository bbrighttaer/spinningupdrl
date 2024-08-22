
RUNNING_CONFIG = {
    "total_timesteps": 1000000,
    "logging_steps": 10,
    "checkpoint_freq": 1000,
    "episode_reward_mean_goal": 1000,
    "max_timesteps_per_episode": 1000,
    "evaluation_interval": 1000,
    "evaluation_num_episodes": 5,
}
ALGO_CONFIG = {
    "buffer_size": 50000,
    "epsilon": 1.0,
    "final_epsilon": 0.1,
    "epsilon_timesteps": 30000,
    "training_batch_size": 32,
    "replay_start_size": 100,
    "gamma": 0.99,
    "target_update_freq": 20,
    "optimizer": "adam",
    "learning_rate": 0.0005,
    "grad_clip": 40,
    "tau": 1.0,
    "reward_normalization": False,
    "comm_size": 0,  # size or dimension of a message (if communication is enabled)
    "discrete_comm_space_size": 5,  # the size of the message space when using discrete communication
}
MODEL_CONFIG = {
    "core_arch": "mlp",
    "encoder_layers": [128],
    "hidden_state_dim": 128,
    "hidden_layers": [128, 128, 64],
    "embedding_dim": 8,
    "num_rnn_layers": 1,
    "activation": "relu",
    "dropout_rate": 0.,
}
ENV_CONFIG = {
    "id": "Switch2-v1"
}
