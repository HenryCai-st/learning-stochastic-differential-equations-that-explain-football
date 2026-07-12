from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.simulators.ou import PARAMETER_HIGH, PARAMETER_LOW, pitch_normalize_xy


class OUParameterNormalizer:
    """Normalize football OU baseline parameters to [-1, 1]."""

    def __init__(self, low: np.ndarray = PARAMETER_LOW, high: np.ndarray = PARAMETER_HIGH):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)

    def normalize(self, params: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = torch.is_tensor(params)
        if is_tensor:
            low = torch.as_tensor(self.low, dtype=params.dtype, device=params.device)
            high = torch.as_tensor(self.high, dtype=params.dtype, device=params.device)
            p = params.clone()
            p[..., 1] = torch.log(torch.clamp(p[..., 1], min=float(self.low[1])))
            low_log = low.clone()
            high_log = high.clone()
            low_log[1] = torch.log(low_log[1])
            high_log[1] = torch.log(high_log[1])
            return ((p - low_log) / (high_log - low_log)) * 2.0 - 1.0

        p = np.asarray(params, dtype=np.float32).copy()
        low = self.low.copy()
        high = self.high.copy()
        p[..., 1] = np.log(np.clip(p[..., 1], self.low[1], None))
        low[1] = np.log(low[1])
        high[1] = np.log(high[1])
        return ((p - low) / (high - low)) * 2.0 - 1.0

    def denormalize(self, params_norm: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = torch.is_tensor(params_norm)
        if is_tensor:
            low = torch.as_tensor(self.low, dtype=params_norm.dtype, device=params_norm.device)
            high = torch.as_tensor(self.high, dtype=params_norm.dtype, device=params_norm.device)
            low_log = low.clone()
            high_log = high.clone()
            low_log[1] = torch.log(low_log[1])
            high_log[1] = torch.log(high_log[1])
            p = (params_norm + 1.0) / 2.0
            raw = p * (high_log - low_log) + low_log
            raw[..., 1] = torch.exp(raw[..., 1])
            return raw

        low = self.low.copy()
        high = self.high.copy()
        low[1] = np.log(low[1])
        high[1] = np.log(high[1])
        p = (np.asarray(params_norm, dtype=np.float32) + 1.0) / 2.0
        raw = p * (high - low) + low
        raw[..., 1] = np.exp(raw[..., 1])
        return raw


class FootballOUDataset(Dataset):
    """
    Dataset for synthetic OU football tracks.

    Expected .npz keys:
        tracks: (N, steps, 2)
        parameters: (N, 2)
        y0: (N, 2)
        target: (N, 2)
    """

    def __init__(self, data_dir: str | Path, normalize_params: bool = True):
        self.data_dir = Path(data_dir)
        loaded = np.load(self.data_dir / "dataset.npz", allow_pickle=True)
        self.tracks = loaded["tracks"].astype(np.float32)
        self.parameters = loaded["parameters"].astype(np.float32)
        self.y0 = loaded["y0"].astype(np.float32)
        self.target = loaded["target"].astype(np.float32)
        self.normalizer = OUParameterNormalizer()
        self.normalize_params = normalize_params

        self.track_mean = self.tracks.mean(axis=(0, 1))
        self.track_std = self.tracks.std(axis=(0, 1))
        self.track_std = np.where(self.track_std < 1e-8, 1.0, self.track_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.tracks)

    def _normalize_track(self, track: np.ndarray) -> np.ndarray:
        return ((track - self.track_mean) / self.track_std).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return one synthetic training item.

        `condition` is the football-specific addition compared with Lorenz:
        it contains y0 and target in pitch-normalized coordinates.
        """
        track = self._normalize_track(self.tracks[idx])
        params = self.parameters[idx]
        if self.normalize_params:
            params = self.normalizer.normalize(params)

        condition = np.concatenate([
            pitch_normalize_xy(self.y0[idx]),
            pitch_normalize_xy(self.target[idx]),
        ]).astype(np.float32)

        return {
            "track": torch.from_numpy(track.T),
            "params": torch.tensor(params, dtype=torch.float32),
            "condition": torch.from_numpy(condition),
            "y0": torch.from_numpy(self.y0[idx]),
            "target": torch.from_numpy(self.target[idx]),
        }


class RealFootballWindows(Dataset):
    """Loads extracted real windows for inference/evaluation."""

    def __init__(self, path: str | Path, track_mean: np.ndarray | None = None, track_std: np.ndarray | None = None):
        loaded = np.load(path, allow_pickle=True)
        self.tracks = loaded["tracks"].astype(np.float32)
        self.y0 = loaded["y0"].astype(np.float32)
        self.target = loaded["target"].astype(np.float32)
        self.meta = loaded["meta"] if "meta" in loaded.files else np.arange(len(self.tracks))
        self.track_mean = track_mean if track_mean is not None else self.tracks.mean(axis=(0, 1))
        self.track_std = track_std if track_std is not None else self.tracks.std(axis=(0, 1))
        self.track_std = np.where(self.track_std < 1e-8, 1.0, self.track_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one real observed window in the same shape as synthetic data."""
        track = ((self.tracks[idx] - self.track_mean) / self.track_std).astype(np.float32)
        condition = np.concatenate([
            pitch_normalize_xy(self.y0[idx]),
            pitch_normalize_xy(self.target[idx]),
        ]).astype(np.float32)
        return {
            "track": torch.from_numpy(track.T),
            "condition": torch.from_numpy(condition),
            "y0": torch.from_numpy(self.y0[idx]),
            "target": torch.from_numpy(self.target[idx]),
        }
