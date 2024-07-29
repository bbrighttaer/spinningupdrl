
RUNNING_CONFIG = {
    "total_time_steps": 100000,
}
ALGO_CONFIG = {
    "buffer_size": 5000,
    "epsilon": 1.0,
    "final_epsilon": 0.05,
    "epsilon_timesteps": 50000,
    "training_batch_size": 32,
    "num_steps_to_training": 1000,
    "gamma": 0.99,
    "target_update_freq": 200,
    "optimizer": "rmsprop",
    "learning_rate": 0.0005,
    "grad_clip": 10,
    "tau": 0.5,
}
MODEL_CONFIG = {
    "core_arch": "mlp",
    "hidden_layers": [64],
    "activation": "relu"
}
ENV_CONFIG = {
    "id": "CartPole-v1"
}
