import collections
import random

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from core import utils


def standardize(r):
    return (r - r.mean()) / (r.std() + 1e-5)


def tb_add_scalar(policy, label, value):
    if hasattr(policy, "summary_writer") and hasattr(policy, "policy_id"):
        policy.summary_writer.add_scalar(policy.policy_id + "/" + label, value, policy.global_timestep)


def tb_add_scalars(policy, label, values_dict):
    if hasattr(policy, "summary_writer") and hasattr(policy, "policy_id"):
        policy.summary_writer.add_scalars(
            policy.policy_id + "/" + label, {str(k): v for k, v in values_dict.items()}, policy.global_timestep
        )


def cluster_labels(data, min_samples_in_cluster=2, eps=.1):
    # data = standardize(data)
    # data = data / np.max(data)
    clustering = DBSCAN(min_samples=min_samples_in_cluster, eps=eps).fit(data)
    bin_index_per_label = clustering.labels_
    return bin_index_per_label


def get_target_dist_weights_cl(eval_data) -> np.array:
    data = utils.tensor_to_numpy(eval_data)

    # clustering
    bin_index_per_label = cluster_labels(data)
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(collections.Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # Use re-weighting based on empirical cluster distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [emp_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [1. / (x + 1e-6) for x in eff_num_per_label]
    tdw_weights = np.array(weights).reshape(len(data), -1)
    return tdw_weights


def target_distribution_weighting(policy, targets):
    targets_flat = targets.reshape(-1, 1)
    if random.random() < policy.tdw_schedule.value(policy.global_timestep):
        lds_weights = get_target_dist_weights_cl(targets_flat)
        scaling = len(lds_weights) / (lds_weights.sum() + 1e-7)
        lds_weights *= scaling
        lds_weights = utils.convert_to_tensor(lds_weights, policy.device).reshape(*targets.shape)
        min_w = max(1e-2, lds_weights.min())
        lds_weights = torch.clamp(torch.log(lds_weights), min_w, max=2 * min_w)

        tb_add_scalars(policy, "tdw_stats", {
            # "scaling": scaling,
            "max_weight": lds_weights.max(),
            "min_weight": lds_weights.min(),
            "mean_weight": lds_weights.mean(),
        })
    else:
        lds_weights = torch.ones_like(targets)
    return lds_weights
