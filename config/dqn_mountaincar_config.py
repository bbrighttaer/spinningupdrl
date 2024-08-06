
RUNNING_CONFIG = {
    "total_timesteps": 100000,
    "logging_steps": 10000,
    "checkpoint_freq": 10000,
    "episode_reward_mean_goal": 1000,
    "max_timesteps_per_episode": 1000,
    "evaluation_interval": 1000,
    "evaluation_num_episodes": 20,
}
ALGO_CONFIG = {
    "buffer_size": 5000,
    "epsilon": 1.0,
    "final_epsilon": 0.05,
    "epsilon_timesteps": 100000,
    "training_batch_size": 32,
    "replay_start_size": 1000,
    "gamma": 0.99,
    "target_update_freq": 100,
    "optimizer": "rmsprop",
    "learning_rate": 0.0005,
    "grad_clip": 10,
    "tau": 0.5,
    "reward_normalization": False,
}
MODEL_CONFIG = {
    "core_arch": "rnn",
    "encoder_layers": [128],
    "hidden_state_dim": 128,
    "hidden_layers": [64, 64],
    "num_rnn_layers": 2,
    "activation": "relu",
    "dropout_rate": 0.2,
}
ENV_CONFIG = {
    "id": "MountainCar-v0",
}
