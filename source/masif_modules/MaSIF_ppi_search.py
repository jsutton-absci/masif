"""
MaSIF_ppi_search.py: PyTorch model for MaSIF-search (PPI complex structure
prediction via surface fingerprint matching).

Ported from the original TensorFlow 1.x implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from masif_modules.masif_layers import GaussianBasisConvLayer


class MaSIF_ppi_search(nn.Module):
    """Produce a surface descriptor for each patch that can be compared
    across proteins to find complementary binding interfaces.

    Architecture:
        1. One GaussianBasisConvLayer per input feature  →  [B, n_gauss] each
        2. Stack + reshape                               →  [B, n_feat * n_gauss]
        3. FC (linear, no activation)                   →  [B, n_gauss]
           → global descriptor

    Training uses a metric-learning loss: positive pairs (matching
    interface patches) should have small descriptor distance, negative
    pairs should exceed a margin.

    Note: binder features/coords are *flipped* before being fed (see
    construct_batch in train_ppi_search.py). This is handled externally.

    Args:
        max_rho:     geodesic radius of patches (Å).
        n_thetas:    number of angular Gaussian kernels.
        n_rhos:      number of radial Gaussian kernels.
        n_rotations: rotational augmentations to max-pool.
        feat_mask:   list of 0/1 feature selection flags (length 5).
    """

    def __init__(
        self,
        max_rho: float,
        n_thetas: int = 16,
        n_rhos: int = 5,
        n_rotations: int = 16,
        feat_mask=None,
        # legacy kwargs
        n_gamma: float = 1.0,
        learning_rate: float = 1e-3,
        idx_gpu: str = "/device:GPU:0",
    ):
        super().__init__()
        if feat_mask is None:
            feat_mask = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))

        n_gauss = n_thetas * n_rhos
        sigma_rho_init = max_rho / 8
        sigma_theta_init = 1.0

        # Layer 1: one conv per feature
        self.conv_l1 = nn.ModuleList([
            GaussianBasisConvLayer(
                n_thetas, n_rhos, 1, max_rho,
                sigma_rho_init, sigma_theta_init, n_rotations,
            )
            for _ in range(self.n_feat)
        ])

        # Linear projection (no activation) to form the descriptor
        self.fc_desc = nn.Linear(self.n_feat * n_gauss, n_gauss, bias=True)
        # Initialise as identity-like (matches TF identity activation)
        nn.init.xavier_uniform_(self.fc_desc.weight)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_feat: torch.Tensor,
        rho: torch.Tensor,
        theta: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_feat: [B, V, n_feat]
            rho:        [B, V, 1]
            theta:      [B, V, 1]
            mask:       [B, V, 1]

        Returns:
            descriptor: [B, n_gauss]
        """
        if rho.dim() == 2:
            rho = rho.unsqueeze(-1)
        if theta.dim() == 2:
            theta = theta.unsqueeze(-1)

        B = input_feat.shape[0]

        descs = [
            conv(input_feat[:, :, i : i + 1], rho, theta, mask)
            for i, conv in enumerate(self.conv_l1)
        ]
        x = torch.stack(descs, dim=1).reshape(B, -1)   # [B, n_feat * n_gauss]
        return self.fc_desc(x)                          # [B, n_gauss]

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def compute_loss(
        desc: torch.Tensor,
        pos_thresh: float = 0.0,
        neg_thresh: float = 10.0,
    ):
        """Metric-learning loss on a batch structured as 4 equal blocks:
        [positives | binders | negatives_1 | negatives_2].

        Minimises distance between pos and binder descriptors; maximises
        distance between the two negative descriptors.

        Args:
            desc:       [4*n, n_gauss]  stacked batch.
            pos_thresh: margin for positive pairs.
            neg_thresh: margin for negative pairs.

        Returns:
            loss:          scalar.
            pos_distances: [n] squared L2 distances for positive pairs.
            neg_distances: [n] squared L2 distances for negative pairs.
        """
        n = desc.shape[0] // 4
        desc_pos    = desc[:n]
        desc_binder = desc[n : 2 * n]
        desc_neg    = desc[2 * n : 3 * n]
        desc_neg_2  = desc[3 * n :]

        sq = lambda a, b: (a - b).pow(2).sum(dim=1)

        pos_d = sq(desc_binder, desc_pos)
        neg_d = sq(desc_neg, desc_neg_2)

        pos_loss = F.relu(pos_d - pos_thresh)
        neg_loss = F.relu(-neg_d + neg_thresh)

        pos_mean, pos_std = pos_loss.mean(), pos_loss.std()
        neg_mean, neg_std = neg_loss.mean(), neg_loss.std()
        loss = pos_std + neg_std + pos_mean + neg_mean

        return loss, pos_d.detach(), neg_d.detach()

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total:,}")
        return total
