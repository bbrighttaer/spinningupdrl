
RUNNING_CONFIG = {
    "total_timesteps": 1000000,
    "logging_steps": 1000,
    "checkpoint_freq": 10000,
    "episode_reward_mean_goal": 1000,
    "max_timesteps_per_episode": 1000,
    "evaluation_interval": 1000,
    "evaluation_num_episodes": 20,
}
ALGO_CONFIG = {
    "buffer_size": 1000,
    "epsilon": 1.0,
    "final_epsilon": 1e-8,
    "epsilon_timesteps": 50000,
    "training_batch_size": 32,
    "replay_start_size": 100,
    "gamma": 0.99,
    "target_update_freq": 100,
    "optimizer": "rmsprop",
    "learning_rate": 0.0005,
    "grad_clip": 10,
    "tau": 0.6,
    "reward_normalization": False,
}
MODEL_CONFIG = {
    "core_arch": "mlp",
    "encoder_layers": [64],
    "hidden_state_dim": 64,
    "hidden_layers": [64, 64],
    "num_rnn_layers": 1,
    "activation": "relu",
    "dropout_rate": 0.,
}
ENV_CONFIG = {
    "id": "CartPole-v1",
}