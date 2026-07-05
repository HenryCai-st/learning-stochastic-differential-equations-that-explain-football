"""
src/models/lorenz_models.py
===========================
Neural-network components for Lorenz SDE parameter inference.

Architecture overview
---------------------
TrajectoryEncoder1D   — shared 1-D CNN backbone that maps (batch, 2, T) → (batch, H)
LorenzRegimeClassifier— downstream head: binary regime (fixed-point vs chaos)
LorenzParameterRegressor — downstream head: direct parameter regression
LorenzRatioEstimator  — contrastive SBI model: joint (trajectory, params) → logit
ContrastiveRatioNet   — same as LorenzRatioEstimator but with a projection MLP
                         and NT-Xent / InfoNCE contrastive loss helper
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared backbone
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryEncoder1D(nn.Module):
    """
    1-D CNN encoder for coordinate time-series.

    Input  : (batch, in_channels, T)  — channels-first, standard PyTorch Conv1d format
    Output : (batch, hidden_dim)      — pooled embedding vector

    Three strided Conv1d blocks reduce the temporal dimension, then
    AdaptiveAvgPool1d(1) produces a fixed-size embedding regardless of T.
    """

    def __init__(self, in_channels: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.net(trajectory)


# ─────────────────────────────────────────────────────────────────────────────
# Downstream heads
# ─────────────────────────────────────────────────────────────────────────────

class LorenzRegimeClassifier(nn.Module):
    """
    Multi-class classifier for the three rho regimes: fixed-point, curve, and chaotic/repulsor.

    Loss: CrossEntropyLoss
    """

    def __init__(self, in_channels: int = 2, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.encoder = TrajectoryEncoder1D(in_channels, hidden_dim)
        self.head    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(trajectory))


class LorenzParameterRegressor(nn.Module):
    """
    Direct regression: trajectory → normalised parameter vector.

    Output is sigmoid-activated so it lives in [0, 1]^param_dim.
    Loss: L1Loss  (robust to outliers in parameter space)
    """

    def __init__(self, in_channels: int = 2, hidden_dim: int = 128, param_dim: int = 4):
        super().__init__()
        self.encoder = TrajectoryEncoder1D(in_channels, hidden_dim)
        self.head    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, param_dim),
            nn.Sigmoid(),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(trajectory))


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive SBI models
# ─────────────────────────────────────────────────────────────────────────────

class LorenzRatioEstimator(nn.Module):
    """
    Neural ratio estimator (NRE) for likelihood-free inference.

    Architecture
    ------------
    trajectory_encoder : 1-D CNN → (batch, H)
    param_encoder      : MLP     → (batch, H)
    head               : MLP     → (batch, 1)  — scalar logit

    Training
    --------
    Use BCEWithLogitsLoss on matched (label=1) and mismatched (label=0) pairs.
    The trained logit approximates  log p(x | θ) / p(x)  (the likelihood ratio),
    which is the sufficient statistic for MCMC posterior sampling.
    """

    def __init__(
        self,
        trajectory_channels: int = 2,
        param_dim:           int = 4,
        hidden_dim:          int = 128,
    ):
        super().__init__()
        self.trajectory_encoder = TrajectoryEncoder1D(trajectory_channels, hidden_dim)
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trajectory: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        z_traj  = self.trajectory_encoder(trajectory)   # (B, H)
        z_param = self.param_encoder(params)             # (B, H)
        return self.head(torch.cat([z_traj, z_param], dim=1)).squeeze(1)  # (B,)


class ContrastiveRatioNet(nn.Module):
    """
    Extended contrastive SBI model with:
      - L2-normalised projection heads (for InfoNCE / NT-Xent style training)
      - A ``contrastive_loss`` class-method for convenience

    Architecture
    ------------
    trajectory_encoder : 1-D CNN    → (B, H)
    traj_projector     : MLP        → (B, proj_dim)  L2-normalised
    param_encoder      : MLP        → (B, H)
    param_projector    : Linear     → (B, proj_dim)  L2-normalised
    ratio_head         : MLP        → (B, 1)  logit (same as LorenzRatioEstimator)

    The ratio head is used for MCMC inference (BCEWithLogitsLoss).
    The projectors can be used for representation learning (InfoNCE).
    """

    def __init__(
        self,
        trajectory_channels: int = 2,
        param_dim:           int = 4,
        hidden_dim:          int = 128,
        proj_dim:            int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── shared encoders ───────────────────────────────────────────────────
        self.trajectory_encoder = TrajectoryEncoder1D(trajectory_channels, hidden_dim)
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── projection heads (for NT-Xent contrastive learning) ───────────────
        self.traj_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.param_projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
        )

        # ── binary ratio head (for MCMC) ──────────────────────────────────────
        self.ratio_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def encode_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """(B, 2, T)  (B, H)"""
        return self.trajectory_encoder(trajectory)

    def encode_params(self, params: torch.Tensor) -> torch.Tensor:
        """(B, param_dim)  (B, H)"""
        return self.param_encoder(params)

    def forward(self, trajectory: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Returns scalar logit for each pair.  Shape (B,)."""
        z_t = self.encode_trajectory(trajectory)
        z_p = self.encode_params(params)
        return self.ratio_head(torch.cat([z_t, z_p], dim=1)).squeeze(1)

    def project(
        self, trajectory: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return L2-normalised projections for contrastive learning.

        Returns
        -------
        z_traj  : (B, proj_dim)
        z_param : (B, proj_dim)
        """
        z_t = F.normalize(self.traj_projector(self.encode_trajectory(trajectory)), dim=1)
        z_p = F.normalize(self.param_projector(self.encode_params(params)), dim=1)
        return z_t, z_p

    @staticmethod
    def nt_xent_loss(
        z_traj:      torch.Tensor,   # (B, D)  L2-normalised
        z_param:     torch.Tensor,   # (B, D)  L2-normalised
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        NT-Xent (Normalised Temperature-scaled Cross Entropy) contrastive loss.

        For each anchor trajectory z_traj[i], the positive is z_param[i] (matching
        parameters).  All other z_param[j≠i] are negatives within the batch.

        This is the contrastive learning objective from SimCLR, adapted to the
        (trajectory, parameter) pair structure of SBI.

        Loss = −(1/B) Σ_i  log [
            exp(z_t[i]·z_p[i] / τ)  /
            Σ_{j} exp(z_t[i]·z_p[j] / τ)
        ]
        """
        B = z_traj.shape[0]
        # Similarity matrix  (B, B)
        sim = torch.mm(z_traj, z_param.T) / temperature
        # Positive is on the diagonal
        labels = torch.arange(B, device=z_traj.device)
        return F.cross_entropy(sim, labels)
