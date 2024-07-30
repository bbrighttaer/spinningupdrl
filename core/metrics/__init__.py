from enum import Enum


class LearningMetrics(Enum):
    TRAINING_LOSS = "loss"
    GRAD_NORM = "grad_norm"
    TD_ERROR_ABS = "td_error_abs"
    Q_TAKEN_MEAN = "q_taken_mean"
    TARGET_MEAN = "target_mean"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class PerformanceMetrics(Enum):
    EPISODE_REWARD = "episode_reward"
    EPISODE_LENGTH = "episode_length"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
