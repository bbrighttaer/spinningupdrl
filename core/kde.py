import torch
import numpy as np
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset, DataLoader

from core import EPS, utils


def get_weights(policy, obs, actions, q_values, target_q_values, mask):
    B, T = obs.shape[:2]
    N = B * T
    target_q_values = target_q_values.view(B * T, -1)
    q_values = q_values.view(B * T, -1)
    actions = actions.view(B * T, -1)
    obs = obs.view(B * T, -1)
    mask = mask.view(B * T, -1)
    x1 = torch.cat([obs, target_q_values], dim=-1)
    # x2 = torch.cat([obs, target_q_values], dim=-1)
    # samples = torch.cat([x1, x2], dim=0)
    # samples_mask = torch.cat([mask, mask], dim=0)

    # update vae parameters
    fit_res = fit_vae(policy, x1, mask)

    with torch.no_grad():
        policy.vae.eval()
        policy.target_vae.eval()

        # select samples
        subset_size = policy.algo_config.kde_subset_size or 100

        # vae output
        # joint_z, joint_mu, joint_logvar = policy.vae.encode(samples)

        # compute Q(x)
        outputs_q, mu_q, logvar_q = policy.vae.encode(x1)
        q_densities = kde_density(outputs_q, mu_q, logvar_q, subset_size, mask)
        # q_densities = utils.apply_scaling(q_densities)

        # compute P(x)
        outputs_p, mu_p, logvar_p = policy.target_vae.encode(x1)
        targets_scaled = utils.shift_and_scale(target_q_values)
        p_densities = kde_density(outputs_q, mu_p, logvar_p, subset_size, mask)
        p_densities = targets_scaled / p_densities

        # compute weights
        weights = p_densities / q_densities
        weights = wts_scaling(weights, mask)
        weights = weights.view(B, T, 1)
        return weights


def fit_vae(policy, training_data, mask, num_epochs=2):
    dataset = TensorDataset(training_data, mask)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    policy.vae.train()
    training_loss = []
    vae_grad_norm = []

    for epoch in range(num_epochs):
        ep_loss = 0

        for batch in data_loader:
            batch_samples, batch_mask = batch
            policy.vae_optimizer.zero_grad()
            recon_batch, mu, logvar = policy.vae(batch_samples)
            loss, mse, kld = vae_loss_function(recon_batch, batch_samples, mu, logvar, batch_mask)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.vae.parameters(), policy.algo_config.grad_clip)
            ep_loss += loss.item()
            policy.vae_optimizer.step()
            vae_grad_norm.append(norm)

        training_loss.append(ep_loss / len(dataset))

    return np.mean(training_loss), np.mean(vae_grad_norm)


def vae_loss_function(recon_x, x, mu, logvar, mask):
    mse = (recon_x - x) ** 2
    mse *= mask
    mse = torch.sum(mse)
    kld = 1 + logvar - mu.pow(2) - logvar.exp()
    kld *= mask
    kld = -0.5 * torch.sum(kld)
    return mse + kld, mse, kld


def wts_scaling(vector, mask):
    # Compute scaling factor
    scaling = mask.sum() / vector.sum()

    # Scale the vector
    vector *= scaling

    # Normalise to [0, 1]
    vector /= (vector.max() + 1e-7)
    return vector

def gaussian_density(z, mu, logvar):
    """
    Compute the Gaussian density of z given a Gaussian defined by mu and logvar.

    Parameters:
    z (tensor): Input tensor of shape (N, D).
    mu (tensor): Mean tensor of shape (N, D).
    logvar (tensor): Log variance tensor of shape (N, D).

    Returns:
    tensor: Gaussian density of shape (N, D).
    """
    std = torch.exp(0.5 * logvar)
    var = std ** 2
    normalization = torch.sqrt(2 * np.pi * var)

    # Compute exponent
    x = -0.5 * ((z - mu) ** 2 / var)

    # Compute density
    exponent = torch.exp(x)
    density = exponent / normalization

    return density


def compute_gaussian_densities(Z, logvars, mus, mask, weights=None):
    # Expand dimensions for broadcasting
    Z_expanded = Z.detach().unsqueeze(1)
    mus_expanded = mus.detach().unsqueeze(0)
    logvars_expanded = logvars.detach().unsqueeze(0)

    # Compute pairwise Gaussian densities
    pairwise_densities = gaussian_density(Z_expanded, mus_expanded, logvars_expanded)

    # Compute product of densities across dimensions
    pairwise_densities_prod = pairwise_densities.prod(dim=2)

    # apply weights if present
    if weights is not None:
        pairwise_densities_prod *= weights.T.expand_as(pairwise_densities_prod)
    N = torch.sum(mask)
    mask = mask.T.expand_as(pairwise_densities_prod)
    pairwise_densities_prod *= mask

    # Compute densities
    densities = pairwise_densities_prod.sum(dim=1) / N
    densities = torch.clamp(densities, min=EPS)

    return densities


def nystroem_gaussian_density(z, mu, log_var, num_samples, mask, weights=None):
    """
    Compute the Gaussian densities of z given a Gaussian defined by mu and logvar.

    Parameters:
    z (tensor): Input tensor of shape (N, D).
    mu (tensor): Mean tensor of shape (N, D).
    log_var (tensor): Log variance tensor of shape (N, D).
    num_samples (int): Number of samples for the Nystroem approximation.
    mask: sequence mask (N, 1)
    weights: KDE weights (N, 1)

    Returns:
    tensor: Gaussian densities of shape (N, 1).
    """
    N, D = z.shape
    std = torch.exp(0.5 * log_var)
    var = std ** 2

    # Sample selection
    indices = torch.randperm(N)[:num_samples]
    z_sampled = z[indices]
    mu_sampled = mu[indices]
    var_sampled = var[indices]

    # Compute normalization factors
    normalization = torch.sqrt(2 * np.pi * var_sampled)

    # Compute kernel sub-matrix K_m
    diff = z_sampled.unsqueeze(1) - mu_sampled.unsqueeze(0)
    K_m = torch.exp(-0.5 * (diff ** 2 / var_sampled.unsqueeze(0))) / normalization.unsqueeze(0)
    K_m = K_m.prod(dim=2)

    # Compute cross-kernel sub-matrix K_Nm
    diff = z.unsqueeze(1) - mu_sampled.unsqueeze(0)
    K_Nm = torch.exp(-0.5 * (diff ** 2 / var_sampled.unsqueeze(0))) / normalization.unsqueeze(0)
    K_Nm = K_Nm.prod(dim=2)

    # Compute the approximate kernel matrix
    K_m_inv = torch.linalg.pinv(K_m)
    K_approx = K_Nm @ K_m_inv @ K_Nm.T

    # Compute densities
    if weights is not None:
        K_approx *= weights.T.expand_as(K_approx)
    N = torch.sum(mask)
    mask = mask.T.expand_as(K_approx)
    K_approx_mask = K_approx * mask
    densities = K_approx_mask.sum(dim=1) / N
    densities = torch.clamp(densities.view(-1, 1), min=EPS)

    return densities


def kde_density(samples, mus, logvars, num_samples, mask, weights=None, approx=True):
    """
    Compute the density of each sample z_i in Z by merging all individual Gaussian distributions.

    Parameters:
    Z (tensor): NxD tensor of samples.
    mus (tensor): NxD tensor of means.
    logvars (tensor): NxD tensor of log variances.
    approx: whether to use an approximation method for KDE
    num_samples: only applicable when approx is set to True
    weights: weights of datapoints. Must be same dimensions as Z

    Returns:
    tensor: Nx1 tensor of densities for each sample.
    """
    if approx:
        densities = nystroem_gaussian_density(samples, mus, logvars, num_samples, mask, weights=weights)
    else:
        densities = compute_gaussian_densities(samples, logvars, mus, mask, weights=weights)

    return densities
