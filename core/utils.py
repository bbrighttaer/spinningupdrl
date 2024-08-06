import copy
import datetime
import os
import random
import string

import gymnasium
import gym

import joblib
import numpy as np
import torch.nn


class DotDic(dict):
    """dot.notation for dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))


def update_dict(target_dict: dict, ref_dict: dict, create=False):
    """
    Updates `target_dict` with values of `ref_dict` for where keys are common between the two dicts.

    Arguments
    -----------
    :param target_dict: The dictionary to be updated
    :param ref_dict: The dictionary containing records for the update
    :param create: If True, keys that are absent in `target_dict` will be created.
    :return: updated dictionary (copy update of `target_dict`)
    """
    target_dict = copy.deepcopy(target_dict)
    for k in ref_dict:
        ref_val = ref_dict[k]
        if create and ref_val is not None:
            target_dict[k] = ref_val
            continue

        if k in target_dict and ref_val is not None:
            target_dict[k] = ref_val

    return target_dict


def generate_random_label(length=4):
    rand_lbl = "".join(random.choices(string.ascii_uppercase + string.digits, k=length)).lower()
    now = datetime.datetime.now()
    rand_lbl += "_" + "".join([str(v) for v in [now.year, now.month, now.day, now.hour, now.minute, now.second]])
    return rand_lbl


def get_activation_function(name: str):
    return {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "leaky_relu": torch.nn.LeakyReLU,
    }[name.lower()]


def soft_update(target_net, source_net, tau):
    """
    Soft update the parameters of the target network with those of the source network.

    Args:
    - target_net: Target network.
    - source_net: Source network.
    - tau: Soft update parameter (0 < tau <= 1).

    Returns:
    - target_net: Updated target network.
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    return target_net


def linear_interpolation(left_value: float, right_value: float, alpha: float) -> float:
    """
    Linearly interpolates between two values.

    Arguments
    ------------
    :param left_value: The value at the left endpoint (float).
    :param right_value: The value at the right endpoint (float).
    :param alpha: The interpolation parameter, where 0.0 corresponds to the left value
                  and 1.0 corresponds to the right value (float).
    :return: The interpolated value (float).
    """
    return left_value + alpha * (right_value - left_value)


def convert_to_tensor(x, device=None):
    if torch.is_tensor(x):
        return x.to(device) if device else x

    if isinstance(x, np.ndarray):
        if x.dtype == np.object_:
            return x
        return torch.from_numpy(x).to(device)


def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def to_numpy_array(x, dtype=np.float32):
    if x is None:
        return

    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif type(x) in [list, set, tuple]:
        return np.array(x, dtype=dtype)
    else:
        return np.array([x], dtype=dtype)


def unroll_mac(model, obs_batch, **kwargs):
    B, T = obs_batch.shape[:2]
    h = [s.expand([B, -1]) for s in model.get_initial_state()]

    outputs = []
    for t in range(T):
        obs_t = obs_batch[:, t]
        out, h = model(obs_t, h, **kwargs)
        outputs.append(out.unsqueeze(1))

    model_out = torch.cat(outputs, dim=1)
    return model_out


def tensor_state_dict_to_numpy_state_dict(state_dict):
    return {k: tensor_to_numpy(v) for k, v in state_dict.items()}


def numpy_state_dict_to_tensor_state_dict(state_dict, device):
    return {
        k: torch.as_tensor(v).to(device) for k, v in state_dict.items()
    }


def save_policy_weights(policy, base_dir, checkpoint_count):
    if isinstance(policy, dict):
        weights = {policy_id: policy[policy_id].get_weights() for policy_id in policy}
    else:
        weights = policy.get_weights()
    chkpt_dir_name = "checkpoints"
    os.makedirs(os.path.join(base_dir, chkpt_dir_name), exist_ok=True)
    joblib.dump(weights, os.path.join(base_dir, chkpt_dir_name, f"checkpoint-{checkpoint_count}"))


def load_policy_weights(weights_file_path):
    weights = joblib.load(weights_file_path)
    return weights


def make_multi_agent_env(**kwargs):
    try:
        env = gymnasium.make(**kwargs, disable_env_checker=True)
    except gymnasium.error.NameNotFound:  # check if env is in ma_gym envs
        from core.envs.ma_gym_wrapper import MAGymEnvWrapper
        env = gym.make(**kwargs)
        env = MAGymEnvWrapper(env)
    return env


def get_smac_stats(
        death_tracker_ally,
        death_tracker_enemy,
        battle_win_queue,
        ally_survive_queue,
        enemy_killing_queue) -> dict:
    # SMAC metrics (from https://github.com/Replicable-MARL/MARLlib/blob/mq_dev/SMAC/metric/smac_callback.py)
    smac_stats = {}
    ally_state = death_tracker_ally
    enemy_state = death_tracker_enemy

    # count battle win rate in recent 100 games
    if battle_win_queue.full():
        battle_win_queue.get()  # pop FIFO

    # compute win rate
    battle_win_this_episode = int(all(enemy_state == 1))  # all enemy died / win
    battle_win_queue.put(battle_win_this_episode)
    smac_stats["battle_win_rate"] = sum(battle_win_queue.queue) / battle_win_queue.qsize()

    # count ally survive in recent 100 games
    if ally_survive_queue.full():
        ally_survive_queue.get()  # pop FIFO

    # compute ally survive rate
    ally_survive_this_episode = sum(ally_state == 0) / ally_state.shape[0]  # all enemy died / win
    ally_survive_queue.put(ally_survive_this_episode)
    smac_stats["ally_survive_rate"] = sum(ally_survive_queue.queue) / ally_survive_queue.qsize()

    # count enemy killing rate in recent 100 games
    if enemy_killing_queue.full():
        enemy_killing_queue.get()  # pop FIFO

    # compute enemy kill rate
    enemy_killing_this_episode = sum(enemy_state == 1) / enemy_state.shape[0]  # all enemy died / win
    enemy_killing_queue.put(enemy_killing_this_episode)
    smac_stats["enemy_kill_rate"] = sum(enemy_killing_queue.queue) / enemy_killing_queue.qsize()

    return smac_stats
