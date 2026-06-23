from __future__ import annotations

import torch
from torch import nn


class TrajectoryEncoder1D(nn.Module):
    """Compact 1D CNN encoder for coordinate sequences."""

    def __init__(self, in_channels: int = 2, hidden_dim: int = 128):
        super().__init__()
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


class LorenzRegimeClassifier(nn.Module):
    """Predicts fixed-point vs chaotic regime from a trajectory."""

    def __init__(self, in_channels: int = 2, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.encoder = TrajectoryEncoder1D(in_channels, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(trajectory))


class LorenzParameterRegressor(nn.Module):
    """Predicts normalized Lorenz parameters from a trajectory."""

    def __init__(self, in_channels: int = 2, hidden_dim: int = 128, param_dim: int = 4):
        super().__init__()
        self.encoder = TrajectoryEncoder1D(in_channels, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, param_dim),
            nn.Sigmoid(),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(trajectory))


class LorenzRatioEstimator(nn.Module):
    """
    Binary classifier for matched parameter/trajectory pairs.

    Its logit can be interpreted as a learned compatibility score. This is the
    Lorenz demo version of the classifier-based SBI idea from the project PDF.
    """

    def __init__(
        self,
        trajectory_channels: int = 2,
        param_dim: int = 4,
        hidden_dim: int = 128,
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
        z_traj = self.trajectory_encoder(trajectory)
        z_param = self.param_encoder(params)
        return self.head(torch.cat([z_traj, z_param], dim=1)).squeeze(1)

