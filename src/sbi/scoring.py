"""Checkpoint loading, track normalization, and ratio-classifier scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.sbi.ratio_model import ModelVotingRatioClassifier
from src.simulators.model_voting import (
    MODEL_TO_ID,
    normalize_padded_parameters,
    pad_parameters,
)


def load_checkpoint(path: Path, device: torch.device) -> tuple[ModelVotingRatioClassifier, dict]:
    """Load the trained model-voting ratio classifier and checkpoint payload."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run train_model_voting_ratio.py first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ModelVotingRatioClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def checkpoint_array(checkpoint: dict, key: str) -> np.ndarray:
    """Read checkpoint arrays saved as either tensors or NumPy arrays."""
    value = checkpoint[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_track(track: np.ndarray, checkpoint: dict) -> np.ndarray:
    """Normalize an observed track with training-set checkpoint statistics."""
    track_mean = checkpoint_array(checkpoint, "track_mean").astype(np.float32)
    track_std = checkpoint_array(checkpoint, "track_std").astype(np.float32)
    track_std = np.where(track_std < 1e-8, 1.0, track_std)
    return ((track - track_mean) / track_std).astype(np.float32)


@torch.no_grad()
def score_params(
    model: ModelVotingRatioClassifier,
    track_t: torch.Tensor,
    condition_t: torch.Tensor,
    model_name: str,
    params: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Evaluate classifier logits for many candidate parameters of one model."""
    padded, mask = pad_parameters(model_name, params.astype(np.float32))
    params_norm = normalize_padded_parameters(model_name, padded)
    model_id = MODEL_TO_ID[model_name]
    outputs: list[np.ndarray] = []
    for start in range(0, len(params), batch_size):
        end = start + batch_size
        n = len(params_norm[start:end])
        logits = model(
            track_t.repeat(n, 1, 1),
            torch.from_numpy(params_norm[start:end]).to(device),
            torch.from_numpy(mask[start:end]).to(device),
            torch.full((n,), model_id, dtype=torch.long, device=device),
            condition_t.repeat(n, 1),
        )
        outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs)
