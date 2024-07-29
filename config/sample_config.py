
RUNNING_CONFIG = {
    "total_time_steps": 100,
}
ALGO_CONFIG = {
    "buffer_size": 500,
    "epsilon": 1.0,
    "epsilon_time_steps": 50,
    "final_epsilon": 0.01,
    "training_batch_size": 10,
    "num_steps_to_training": 10,
}
MODEL_CONFIG = {
    "core_arch": "mlp",
    "hidden_layers": [64],
    "activation": "relu"
}
ENV_CONFIG = {
    "id": "CartPole-v1"
}
