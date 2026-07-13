"""
encoder.py
==========
1D CNN encoder and probabilistic regressor for SDE trajectory processing.

Architecture overview:
  TrajectoryEncoder      : (batch, 3, steps) → (batch, feature_dim)
  ProjectionHead         : (batch, feature_dim) → (batch, proj_dim)
                           Used only during contrastive training; discarded afterwards.
  ProbabilisticRegressor : encoder + MLP head → (mean, logvar) over θ
                           Trained after contrastive pretraining.
"""

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    """
    Maps a trajectory (batch, 3, steps) → (batch, feature_dim).

    5 strided Conv1d blocks progressively halve the time dimension while
    expanding channels. AdaptiveAvgPool1d collapses the remaining time
    dimension to a fixed-size vector regardless of input length — useful
    since real tracks may have different lengths than simulated ones.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv1d(in_channels, 32,  kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # Block 2: 32 → 64
            nn.Conv1d(32,  64,  kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Block 3: 64 → 128
            nn.Conv1d(64,  128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # Block 4: 128 → 256
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # Block 5: 256 → feature_dim
            nn.Conv1d(256, feature_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Collapses time dimension: (batch, feature_dim, T') → (batch, feature_dim, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, steps) → (batch, feature_dim)"""
        x = self.conv(x)
        x = self.pool(x)
        return x.squeeze(-1)


class ProjectionHead(nn.Module):
    """
    MLP projection head used only during contrastive training (SimCLR / InfoNCE).

    Following the SimCLR finding: contrastive loss is applied in the projected
    space (proj_dim), but the encoder backbone (feature_dim) is what gets kept
    for downstream tasks. This head is discarded after pretraining.
    """

    def __init__(self, input_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, proj_dim)"""
        return self.net(x)


class ProbabilisticRegressor(nn.Module):
    """
    Encoder + probabilistic MLP head → Gaussian distribution over θ.

    Outputs (mean, logvar) where:
      mean   ∈ [-1, 1]  via Tanh — matches ParameterNormalizer's [-1, 1] output
      logvar ∈ [-8, 2]  clamped — prevents variance collapse or explosion

    Training objective: Gaussian NLL (see gaussian_nll_loss).
    Encourages calibrated uncertainty: high variance where the model is wrong,
    low variance where it's confident.

    Freeze behaviour
    ----------------
    When freeze_encoder=True, encoder parameters have requires_grad set to False
    at init time. This means:
      - The optimizer filter `[p for p in model.parameters() if p.requires_grad]`
        in train_regressor.py correctly excludes them.
      - PyTorch's autograd skips gradient computation through the encoder
        automatically (no requires_grad tensor → no grad tracked).
      - No torch.no_grad() context manager needed in forward.
    """

    def __init__(
        self,
        encoder: TrajectoryEncoder,
        feature_dim: int = 256,
        out_dim: int = 4,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.shared_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Tanh → [-1, 1] matches ParameterNormalizer output range
        # (Sigmoid would give [0, 1] which is mismatched and silently trains wrong)
        self.mean_head   = nn.Sequential(nn.Linear(64, out_dim), nn.Tanh())
        self.logvar_head = nn.Linear(64, out_dim)

    @property
    def encoder_frozen(self) -> bool:
        """Reads actual parameter state rather than a stored flag."""
        return not next(self.encoder.parameters()).requires_grad

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, 3, steps)
        returns: mean (batch, out_dim), logvar (batch, out_dim)
        """
        features = self.encoder(x)            # grad flows if encoder not frozen
        h        = self.shared_mlp(features)
        mean     = self.mean_head(h)
        logvar   = self.logvar_head(h).clamp(min=-8.0, max=2.0)
        return mean, logvar

    @staticmethod
    def gaussian_nll_loss(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood loss.

        L = 0.5 * mean_over_batch( logvar + (pred - target)^2 / exp(logvar) )

        Working in log-space (logvar) rather than converting to var first avoids
        a redundant exp→log round-trip and is more numerically stable.

        When logvar is fixed this reduces to MSE. The model is rewarded for
        expressing high uncertainty (large logvar) in regions where it's wrong,
        and penalised for being overconfident.
        """
        return 0.5 * (logvar + (mean - target) ** 2 / logvar.exp()).mean()