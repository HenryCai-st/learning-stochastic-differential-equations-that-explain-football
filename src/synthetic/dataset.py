from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ModelVotingDataset(Dataset):
    """Loads mixed-model synthetic tracks for model-voting ratio training."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        track_mean: np.ndarray | None = None,
        track_std: np.ndarray | None = None,
    ):
        self.data_path = Path(data_path)
        if self.data_path.is_dir():
            self.data_path = self.data_path / "dataset.npz"
        self.data_dir = self.data_path.parent
        loaded = np.load(self.data_path, allow_pickle=True)
        self.tracks = loaded["tracks"].astype(np.float32)
        self.parameters_norm = loaded["parameters_norm"].astype(np.float32)
        self.parameter_mask = loaded["parameter_mask"].astype(np.float32)
        self.model_id = loaded["model_id"].astype(np.int64)
        self.conditions = loaded["conditions"].astype(np.float32)
        self.model_names = loaded["model_names"].tolist()
        self.steps = int(loaded["steps"]) if "steps" in loaded.files else int(self.tracks.shape[1])
        self.dt = float(loaded["dt"]) if "dt" in loaded.files else None
        self.T = float(loaded["T"]) if "T" in loaded.files else None
        self.seed = int(loaded["seed"]) if "seed" in loaded.files else None
        self.condition_sources = (
            str(loaded["condition_sources"].item())
            if "condition_sources" in loaded.files
            else None
        )

        computed_mean = self.tracks.mean(axis=(0, 1)).astype(np.float32)
        computed_std = self.tracks.std(axis=(0, 1)).astype(np.float32)
        self.track_mean = computed_mean if track_mean is None else np.asarray(track_mean, dtype=np.float32)
        self.track_std = computed_std if track_std is None else np.asarray(track_std, dtype=np.float32)
        if self.track_mean.shape != (self.tracks.shape[2],) or self.track_std.shape != (self.tracks.shape[2],):
            raise ValueError("track_mean and track_std must match the track channel dimension.")
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
