from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ModelVotingDataset(Dataset):
    """Loads mixed-model synthetic tracks for model-voting ratio training."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        loaded = np.load(self.data_dir / "dataset.npz", allow_pickle=True)
        self.tracks = loaded["tracks"].astype(np.float32)
        self.parameters_norm = loaded["parameters_norm"].astype(np.float32)
        self.parameter_mask = loaded["parameter_mask"].astype(np.float32)
        self.model_id = loaded["model_id"].astype(np.int64)
        self.conditions = loaded["conditions"].astype(np.float32)
        self.model_names = loaded["model_names"].tolist()

        self.track_mean = self.tracks.mean(axis=(0, 1)).astype(np.float32)
        self.track_std = self.tracks.std(axis=(0, 1)).astype(np.float32)
        self.track_std = np.where(self.track_std < 1e-8, 1.0, self.track_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        track = ((self.tracks[idx] - self.track_mean) / self.track_std).astype(np.float32)
        return {
            "track": torch.from_numpy(track.T),
            "params": torch.from_numpy(self.parameters_norm[idx]),
            "param_mask": torch.from_numpy(self.parameter_mask[idx]),
            "model_id": torch.tensor(self.model_id[idx], dtype=torch.long),
            "condition": torch.from_numpy(self.conditions[idx]),
        }
