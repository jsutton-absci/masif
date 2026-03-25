"""
masif_layers.py: Shared PyTorch layers for MaSIF models.

The core operation is a Gaussian basis convolution on geodesic polar
coordinates (rho, theta). Each vertex in a patch is assigned a weight
based on its proximity to a grid of Gaussian kernels placed in polar
space. Features are aggregated by those weights, then linearly
transformed.

References:
    Gainza et al. (2020) Nature Methods 17, 184–192.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_initial_grid(max_rho, n_rhos, n_thetas):
    """Return initial Gaussian center coordinates shaped [1, n_rhos*n_thetas].

    The grid matches the Matlab/TF convention used in the original code:
    rho grid excludes 0, theta grid excludes 2π, then the meshgrid is
    transposed before flattening.
    """
    grid_rho = np.linspace(0.0, max_rho, num=n_rhos + 1)[1:]          # skip 0
    grid_theta = np.linspace(0, 2 * np.pi, num=n_thetas + 1)[:-1]     # skip 2π

    grid_rho_, grid_theta_ = np.meshgrid(grid_rho, grid_theta, sparse=False)
    # Transpose matches the original MATLAB/TF convention
    grid_rho_ = grid_rho_.T.flatten()
    grid_theta_ = grid_theta_.T.flatten()

    mu_rho = grid_rho_.reshape(1, -1).astype(np.float32)
    mu_theta = grid_theta_.reshape(1, -1).astype(np.float32)
    return mu_rho, mu_theta


class GaussianBasisConvLayer(nn.Module):
    """Gaussian basis convolution for *in_feat* feature channels.

    For layer 1, call with in_feat=1 once per input feature so that each
    feature has its own learnable Gaussian parameters.

    For layers 2+, call with in_feat=n_feat to process all features at
    once with a single shared set of Gaussian parameters.

    Args:
        n_thetas:    number of angular Gaussian kernels.
        n_rhos:      number of radial Gaussian kernels.
        in_feat:     number of input feature channels.
        max_rho:     maximum geodesic radius of a patch (Å).
        sigma_rho_init:   initial width of radial Gaussians.
        sigma_theta_init: initial width of angular Gaussians.
        n_rotations: number of rotational augmentations applied at
                     inference time (max-pooled).
        normalize:   if True, normalise Gaussian weights over patch
                     vertices so they sum to 1.
    """

    def __init__(
        self,
        n_thetas: int,
        n_rhos: int,
        in_feat: int,
        max_rho: float,
        sigma_rho_init: float,
        sigma_theta_init: float,
        n_rotations: int = 16,
        normalize: bool = True,
    ):
        super().__init__()
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_rotations = n_rotations
        self.normalize = normalize
        n_gauss = n_thetas * n_rhos

        mu_rho_init, mu_theta_init = _compute_initial_grid(max_rho, n_rhos, n_thetas)

        self.mu_rho = nn.Parameter(torch.tensor(mu_rho_init))                        # [1, n_gauss]
        self.sigma_rho = nn.Parameter(torch.full((1, n_gauss), sigma_rho_init))      # [1, n_gauss]
        self.mu_theta = nn.Parameter(torch.tensor(mu_theta_init))                    # [1, n_gauss]
        self.sigma_theta = nn.Parameter(torch.full((1, n_gauss), sigma_theta_init))  # [1, n_gauss]

        n_hidden = n_gauss * in_feat
        self.W_conv = nn.Parameter(torch.empty(n_hidden, n_hidden))
        self.b_conv = nn.Parameter(torch.zeros(n_hidden))
        nn.init.xavier_uniform_(self.W_conv)

    def forward(
        self,
        feat: torch.Tensor,
        rho: torch.Tensor,
        theta: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        Args:
            feat:  [B, V, in_feat]  — per-vertex features.
            rho:   [B, V, 1]        — geodesic radial distance from patch centre.
            theta: [B, V, 1]        — geodesic angular coordinate.
            mask:  [B, V, 1]        — 1 for valid vertices, 0 for padding.
            eps:   numerical stability constant.

        Returns:
            Tensor [B, n_gauss * in_feat] after ReLU.
        """
        B, V, in_feat = feat.shape
        n_gauss = self.n_thetas * self.n_rhos

        rho_flat = rho.reshape(B * V, 1)    # [B*V, 1]
        theta_flat = theta.reshape(B * V, 1)  # [B*V, 1]

        all_conv_feat = []
        for k in range(self.n_rotations):
            theta_rot = (theta_flat + k * 2 * math.pi / self.n_rotations) % (2 * math.pi)

            rho_g = torch.exp(
                -((rho_flat - self.mu_rho) ** 2) / (self.sigma_rho ** 2 + eps)
            )  # [B*V, n_gauss]
            theta_g = torch.exp(
                -((theta_rot - self.mu_theta) ** 2) / (self.sigma_theta ** 2 + eps)
            )  # [B*V, n_gauss]

            gauss_act = (rho_g * theta_g).reshape(B, V, n_gauss)  # [B, V, n_gauss]
            gauss_act = gauss_act * mask                            # zero out padding

            if self.normalize:
                gauss_act = gauss_act / (gauss_act.sum(dim=1, keepdim=True) + eps)

            # Weighted aggregation over patch vertices:
            # [B, V, 1, n_gauss] × [B, V, in_feat, 1] → [B, V, in_feat, n_gauss]
            # sum over V → [B, in_feat, n_gauss]
            gauss_desc = (gauss_act.unsqueeze(2) * feat.unsqueeze(3)).sum(dim=1)
            gauss_desc = gauss_desc.reshape(B, in_feat * n_gauss)  # [B, in_feat*n_gauss]

            conv_feat = gauss_desc @ self.W_conv + self.b_conv      # [B, in_feat*n_gauss]
            all_conv_feat.append(conv_feat)

        # Max-pool over rotations then apply ReLU
        return F.relu(torch.stack(all_conv_feat).amax(dim=0))       # [B, in_feat*n_gauss]
