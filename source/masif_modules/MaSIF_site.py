"""
MaSIF_site.py: PyTorch model for MaSIF-site (protein-protein interaction
site prediction).

Ported from the original TensorFlow 1.x implementation.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from masif_modules.masif_layers import GaussianBasisConvLayer


class MaSIF_site(nn.Module):
    """Predict per-vertex PPI site probability on a protein surface.

    Architecture (n_conv_layers == 1, the default):
        1. One GaussianBasisConvLayer per input feature  →  [B, n_gauss] each
        2. Stack + reshape                               →  [B, n_feat * n_gauss]
        3. FC + ReLU                                     →  [B, n_gauss]
        4. FC + ReLU                                     →  [B, n_feat]
        5. FC + ReLU                                     →  [B, n_thetas]
        6. FC (linear)                                   →  [B, 2]  (logits)

    For n_conv_layers > 1, additional GaussianBasisConvLayers are applied
    after rebuilding per-vertex patches from the neighbour index list.

    Args:
        max_rho:        geodesic radius of patches (Å).
        n_thetas:       number of angular Gaussian kernels.
        n_rhos:         number of radial Gaussian kernels.
        n_rotations:    rotational augmentations to max-pool.
        feat_mask:      list of 0/1 flags selecting active features.
                        len == 5 → [shape_index, ddc, hbond, charge, hphob].
        n_conv_layers:  number of stacked geodesic convolution layers (1–4).
    """

    def __init__(
        self,
        max_rho: float,
        n_thetas: int = 16,
        n_rhos: int = 5,
        n_rotations: int = 16,
        feat_mask=None,
        n_conv_layers: int = 1,
        # legacy kwargs kept for API compatibility
        n_gamma: float = 1.0,
        learning_rate: float = 1e-3,
        idx_gpu: str = "/device:GPU:0",
        optimizer_method: str = "Adam",
    ):
        super().__init__()
        if feat_mask is None:
            feat_mask = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))
        self.n_conv_layers = n_conv_layers

        n_gauss = n_thetas * n_rhos
        sigma_rho_init = max_rho / 8
        sigma_theta_init = 1.0

        def _make_conv(in_feat):
            return GaussianBasisConvLayer(
                n_thetas, n_rhos, in_feat, max_rho,
                sigma_rho_init, sigma_theta_init, n_rotations,
            )

        # Layer 1: one conv per feature (separate Gaussian parameters)
        self.conv_l1 = nn.ModuleList([_make_conv(1) for _ in range(self.n_feat)])

        # FC reduction after layer 1
        self.fc1 = nn.Linear(self.n_feat * n_gauss, n_gauss)
        self.fc2 = nn.Linear(n_gauss, self.n_feat)

        # Optional deeper convolutional layers
        if n_conv_layers > 1:
            self.conv_l2 = _make_conv(self.n_feat)
        if n_conv_layers > 2:
            self.conv_l3 = _make_conv(self.n_feat)
        if n_conv_layers > 3:
            # Layer 4 processes n_feat features but expects n_gauss-wide output
            self.conv_l4 = _make_conv(self.n_feat)

        # Final MLP head
        self.fc_final = nn.Linear(self.n_feat, n_thetas)
        self.fc_out = nn.Linear(n_thetas, 2)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_feat: torch.Tensor,
        rho: torch.Tensor,
        theta: torch.Tensor,
        mask: torch.Tensor,
        indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feat: [B, V, n_feat]  surface feature vectors.
            rho:        [B, V] or [B, V, 1]  geodesic radial coordinates.
            theta:      [B, V] or [B, V, 1]  geodesic angular coordinates.
            mask:       [B, V, 1]  validity mask (1 = valid, 0 = padding).
            indices:    [B, max_verts]  neighbour indices for layers 2+.
                        Required when n_conv_layers > 1.

        Returns:
            logits: [B, 2]
        """
        if rho.dim() == 2:
            rho = rho.unsqueeze(-1)
        if theta.dim() == 2:
            theta = theta.unsqueeze(-1)

        B = input_feat.shape[0]
        n_gauss = self.n_thetas * self.n_rhos

        # ---- Layer 1: per-feature Gaussian basis convolutions ----
        descs = [
            conv(input_feat[:, :, i : i + 1], rho, theta, mask)
            for i, conv in enumerate(self.conv_l1)
        ]  # each [B, n_gauss]
        x = torch.stack(descs, dim=1).reshape(B, -1)   # [B, n_feat * n_gauss]
        x = F.relu(self.fc1(x))                         # [B, n_gauss]
        x = F.relu(self.fc2(x))                         # [B, n_feat]

        # ---- Optional deeper layers ----
        if self.n_conv_layers > 1:
            assert indices is not None, "indices required for n_conv_layers > 1"
            x = x[indices]                              # [B, max_verts, n_feat]
            x = self.conv_l2(x, rho, theta, mask)       # [B, n_feat * n_gauss]
            x = x.reshape(B, self.n_feat, n_gauss).mean(dim=2)  # [B, n_feat]

        if self.n_conv_layers > 2:
            x = x[indices]
            x = self.conv_l3(x, rho, theta, mask)
            x = x.reshape(B, self.n_feat, n_gauss).mean(dim=2)  # [B, n_feat]

        if self.n_conv_layers > 3:
            x = x[indices]
            x = self.conv_l4(x, rho, theta, mask)       # [B, n_feat * n_gauss]
            x = x.reshape(B, self.n_feat, n_gauss).amax(dim=2)  # [B, n_feat]

        # ---- Classification head ----
        x = F.relu(self.fc_final(x))                    # [B, n_thetas]
        logits = self.fc_out(x)                          # [B, 2]
        return logits

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def predict_score(
        self,
        input_feat: torch.Tensor,
        rho: torch.Tensor,
        theta: torch.Tensor,
        mask: torch.Tensor,
        indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return per-vertex interface probability [B], values in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(input_feat, rho, theta, mask, indices)
        return torch.sigmoid(logits)[:, 0]

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pos_idx: torch.Tensor,
        neg_idx: torch.Tensor,
    ):
        """Binary cross-entropy loss on balanced pos/neg subsets.

        Args:
            logits:  [B, 2]   raw model output.
            labels:  [B, 2]   one-hot labels (col 0 = positive class).
            pos_idx: [n_pos]  indices of positive samples.
            neg_idx: [n_neg]  indices of negative samples.

        Returns:
            loss:        scalar Tensor.
            eval_score:  [n_pos + n_neg] sigmoid probability for class 0.
            eval_labels: [n_pos + n_neg, 2] corresponding labels.
        """
        eval_idx = torch.cat([pos_idx, neg_idx])
        eval_logits = logits[eval_idx]
        eval_labels = labels[eval_idx].float()
        loss = F.binary_cross_entropy_with_logits(eval_logits, eval_labels)
        eval_score = torch.sigmoid(eval_logits)[:, 0]
        return loss, eval_score, eval_labels

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total:,}")
        return total
