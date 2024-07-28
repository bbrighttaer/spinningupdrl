import copy
import random
import string
import datetime


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
        if create:
            target_dict[k] = ref_dict[k]
            continue

        if k in target_dict:
            target_dict[k] = ref_dict[k]

    return target_dict


def generate_random_label(length=4):
    rand_lbl = "".join(random.choices(string.ascii_uppercase + string.digits, k=length)).lower()
    now = datetime.datetime.now()
    rand_lbl += "_" + "".join([str(v) for v in [now.year, now.month, now.day, now.hour, now.minute, now.second]])
    return rand_lbl
