import copy
import datetime
import os
import random
import string

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
        return torch.from_numpy(x)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


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
    return {k: to_numpy(v) for k, v in state_dict.items()}


def numpy_state_dict_to_tensor_state_dict(state_dict, device):
    return {
        k: torch.as_tensor(v).to(device) for k, v in state_dict.items()
    }


def save_policy_weights(policy, base_dir, checkpoint_count):
    weights = policy.get_weights()
    chkpt_dir_name = "checkpoints"
    os.makedirs(os.path.join(base_dir, chkpt_dir_name), exist_ok=True)
    joblib.dump(weights, os.path.join(base_dir, chkpt_dir_name, f"checkpoint-{checkpoint_count}"))


def load_policy_weights(weights_file_path):
    weights = joblib.load(weights_file_path)
    return weights
