
RUNNING_CONFIG = {
    "total_timesteps": 100000,
    "logging_steps": 1000,
    "checkpoint_freq": 10000,
    "episode_reward_mean_goal": 1000,
    "max_timesteps_per_episode": 1000,
}
ALGO_CONFIG = {
    "buffer_size": 5000,
    "epsilon": 1.0,
    "final_epsilon": 1e-8,
    "epsilon_timesteps": 50000,
    "training_batch_size": 32,
    "num_steps_to_training": 1000,
    "gamma": 0.99,
    "target_update_freq": 100,
    "optimizer": "rmsprop",
    "learning_rate": 0.0005,
    "grad_clip": 10,
    "tau": 0.5,
    "reward_normalization": True,
}
MODEL_CONFIG = {
    "core_arch": "rnn",
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
