from __future__ import annotations

import torch
from torch import nn

from src.models.encoder import TrajectoryEncoder
from src.sde.model_voting import CONDITION_DIM, MAX_PARAM_DIM, MODEL_NAMES


class ModelVotingRatioClassifier(nn.Module):
    """
    Ratio classifier C_phi(track, model_id, theta, condition).

    The logit is used as the learned likelihood-ratio surrogate for SBI.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        param_dim: int = MAX_PARAM_DIM,
        condition_dim: int = CONDITION_DIM,
        n_models: int = len(MODEL_NAMES),
        model_emb_dim: int = 16,
    ):
        super().__init__()
        self.encoder = TrajectoryEncoder(in_channels=2, feature_dim=feature_dim)
        self.model_embedding = nn.Embedding(n_models, model_emb_dim)
        self.theta_encoder = nn.Sequential(
            nn.Linear(param_dim * 2, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + model_emb_dim + 64 + 32, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(160, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, 1),
        )

    def forward(
        self,
        track: torch.Tensor,
        params: torch.Tensor,
        param_mask: torch.Tensor,
        model_id: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        z_track = self.encoder(track)
        z_model = self.model_embedding(model_id)
        # Include the mask so the network knows which padded theta dimensions
        # are meaningful for each candidate model.
        z_theta = self.theta_encoder(torch.cat([params * param_mask, param_mask], dim=1))
        z_condition = self.condition_encoder(condition)
        features = torch.cat([z_track, z_model, z_theta, z_condition], dim=1)
        return self.classifier(features).squeeze(-1)
