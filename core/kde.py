import torch
import numpy as np

from core import EPS


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


def compute_gaussian_densities(Z, logvars, mus, weights=None):
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

    # Compute densities
    densities = pairwise_densities_prod.mean(dim=1, keepdims=True)
    densities = torch.clamp(densities, min=EPS)

    return densities


def nystroem_gaussian_density(z, mu, log_var, num_samples, weights=None):
    """
    Compute the Gaussian density of z given a Gaussian defined by mu and logvar.

    Parameters:
    z (tensor): Input tensor of shape (N, D).
    mu (tensor): Mean tensor of shape (N, D).
    log_var (tensor): Log variance tensor of shape (N, D).
    num_samples (int): Number of samples for the Nystroem approximation.

    Returns:
    tensor: Gaussian density of shape (N, 1).
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

    # Compute density
    mask = 1 - torch.eye(N, device=z.device)
    K_approx_mask = K_approx * mask
    if weights is not None:
        K_approx_mask *= weights.T.expand_as(K_approx_mask)
    densities_sum = K_approx_mask.sum(dim=1) / (N - 1)
    density = torch.clamp(densities_sum.view(-1, 1), min=1e-7)

    return density


def kde_density(samples, mus, logvars, num_samples, weights=None, approx=True):
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
        densities = nystroem_gaussian_density(samples, mus, logvars, num_samples, weights=weights)
    else:
        densities = compute_gaussian_densities(samples, logvars, mus, weights=weights)

    return densities

