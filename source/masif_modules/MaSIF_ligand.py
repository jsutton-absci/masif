"""
MaSIF_ligand.py: PyTorch model for MaSIF-ligand (ligand-binding pocket
classification).

Ported from the original TensorFlow 1.x implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from masif_modules.masif_layers import GaussianBasisConvLayer


class MaSIF_ligand(nn.Module):
    """Classify a protein surface patch as binding a specific ligand type.

    Architecture:
        1. One GaussianBasisConvLayer per input feature  →  [B, n_gauss] each
        2. Stack + reshape                               →  [B, n_feat * n_gauss]
        3. FC + ReLU                                     →  [B, n_gauss]
        4. Gram matrix: desc.T @ desc / B               →  [1, n_gauss²]
        5. Dropout
        6. FC + ReLU                                     →  [1, 64]
        7. FC (linear)                                   →  [1, n_ligands]

    The Gram matrix step aggregates a whole protein's patch descriptors
    into a single fixed-size representation, so the entire protein is
    treated as one sample.

    Args:
        n_ligands:   number of ligand classes.
        max_rho:     geodesic radius of patches (Å).
        n_thetas:    number of angular Gaussian kernels.
        n_rhos:      number of radial Gaussian kernels.
        n_rotations: rotational augmentations to max-pool.
        feat_mask:   list of 0/1 feature selection flags (length 4).
                     MaSIF-ligand uses 4 features (no shape_index).
        dropout:     dropout probability before the final FC layers.
    """

    def __init__(
        self,
        n_ligands: int,
        max_rho: float,
        n_thetas: int = 16,
        n_rhos: int = 5,
        n_rotations: int = 16,
        feat_mask=None,
        dropout: float = 0.5,
        # legacy kwargs
        n_gamma: float = 1.0,
        learning_rate: float = 1e-4,
        idx_gpu: str = "/gpu:0",
        costfun: str = "dprime",
        session=None,
    ):
        super().__init__()
        if feat_mask is None:
            feat_mask = [1.0, 1.0, 1.0, 1.0]

        self.n_ligands = n_ligands
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

        self.fc1 = nn.Linear(self.n_feat * n_gauss, n_gauss)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(n_gauss * n_gauss, 64)
        self.fc_out = nn.Linear(64, n_ligands)

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
            input_feat: [B, V, n_feat]  — all patches for one protein.
            rho:        [B, V, 1]
            theta:      [B, V, 1]
            mask:       [B, V, 1]

        Returns:
            logits: [1, n_ligands]
        """
        if rho.dim() == 2:
            rho = rho.unsqueeze(-1)
        if theta.dim() == 2:
            theta = theta.unsqueeze(-1)

        B = input_feat.shape[0]
        n_gauss = self.n_thetas * self.n_rhos

        descs = [
            conv(input_feat[:, :, i : i + 1], rho, theta, mask)
            for i, conv in enumerate(self.conv_l1)
        ]
        x = torch.stack(descs, dim=1).reshape(B, -1)   # [B, n_feat * n_gauss]
        x = F.relu(self.fc1(x))                         # [B, n_gauss]

        # Gram matrix over all patches → single protein-level representation
        gram = (x.t() @ x) / B                          # [n_gauss, n_gauss]
        gram = gram.reshape(1, -1)                       # [1, n_gauss²]

        gram = self.dropout(gram)
        gram = F.relu(self.fc2(gram))                    # [1, 64]
        return self.fc_out(gram)                         # [1, n_ligands]

    # ------------------------------------------------------------------

    @staticmethod
    def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Softmax cross-entropy loss.

        Args:
            logits: [1, n_ligands]
            labels: [1, n_ligands] one-hot float tensor.
        """
        return F.cross_entropy(logits, labels.argmax(dim=1))

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total:,}")
        return total
