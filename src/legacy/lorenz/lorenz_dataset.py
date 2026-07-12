from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


PARAM_NAMES = ("sigma", "rho", "beta", "epsilon")


@dataclass(frozen=True)
class ParamScaler:
    """Min-max scaler for Lorenz parameters."""

    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def from_params(cls, params: np.ndarray) -> "ParamScaler":
        return cls(params.min(axis=0), params.max(axis=0))

    def transform(self, params: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.maximum - self.minimum, 1e-8)
        return (params - self.minimum) / denom

    def inverse_transform(self, params_scaled: np.ndarray) -> np.ndarray:
        return params_scaled * (self.maximum - self.minimum) + self.minimum


class LorenzTrajectoryDataset(Dataset):
    """
    Loads trajectories, parameters, and regime labels from lorenz_dataset.npz.

    Returned trajectory tensors have shape (channels, time), which is the
    standard format for a 1D CNN in PyTorch.
    """

    def __init__(
        self,
        path: str | Path,
        split: Literal["all", "train", "val", "test"] = "all",
        split_seed: int = 0,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        max_points: int = 512,
        normalize_trajectory: bool = True,
        param_scaler: ParamScaler | None = None,
    ):
        self.path = Path(path)
        raw = np.load(self.path, allow_pickle=True)

        trajectories = raw["trajectories"]
        if trajectories.dtype == object:
            trajectories = np.stack([np.asarray(t, dtype=np.float32) for t in trajectories])
        else:
            trajectories = trajectories.astype(np.float32)

        self.trajectories = trajectories
        self.params = raw["params"].astype(np.float32)
        self.labels = raw["labels"].astype(np.int64)
        self.max_points = max_points
        self.normalize_trajectory = normalize_trajectory
        self.param_scaler = param_scaler or ParamScaler.from_params(self.params)

        indices = self._split_indices(split, split_seed, train_fraction, val_fraction)
        self.indices = indices

    def _split_indices(
        self,
        split: str,
        split_seed: int,
        train_fraction: float,
        val_fraction: float,
    ) -> np.ndarray:
        n = len(self.labels)
        rng = np.random.default_rng(split_seed)
        indices = rng.permutation(n)

        train_end = int(n * train_fraction)
        val_end = train_end + int(n * val_fraction)

        if split == "all":
            return np.arange(n)
        if split == "train":
            return indices[:train_end]
        if split == "val":
            return indices[train_end:val_end]
        if split == "test":
            return indices[val_end:]
        raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        trajectory = self._prepare_trajectory(self.trajectories[idx])
        params = self.param_scaler.transform(self.params[idx]).astype(np.float32)
        label = self.labels[idx]

        return {
            "trajectory": torch.from_numpy(trajectory),
            "params": torch.from_numpy(params),
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }

    def _prepare_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        trajectory = np.asarray(trajectory, dtype=np.float32)
        if self.max_points and len(trajectory) > self.max_points:
            pick = np.linspace(0, len(trajectory) - 1, self.max_points).astype(np.int64)
            trajectory = trajectory[pick]

        if self.normalize_trajectory:
            minimum = trajectory.min(axis=0, keepdims=True)
            maximum = trajectory.max(axis=0, keepdims=True)
            trajectory = (trajectory - minimum) / np.maximum(maximum - minimum, 1e-8)
            trajectory = trajectory * 2.0 - 1.0

        return trajectory.T.astype(np.float32)


class LorenzPairDataset(Dataset):
    """
    Matched/mismatched pairs for classifier-based simulation-based inference.

    A positive item contains a parameter vector and its own trajectory.
    A negative item contains a parameter vector and a trajectory from another
    sample. The classifier learns whether params and trajectory belong together.
    """

    def __init__(self, base: LorenzTrajectoryDataset, seed: int = 0):
        self.base = base
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        anchor = self.base[item]
        is_match = self.rng.random() < 0.5

        if is_match:
            trajectory = anchor["trajectory"]
            label = 1.0
        else:
            other_item = int(self.rng.integers(0, len(self.base) - 1))
            if other_item >= item:
                other_item += 1
            trajectory = self.base[other_item]["trajectory"]
            label = 0.0

        return {
            "trajectory": trajectory,
            "params": anchor["params"],
            "label": torch.tensor(label, dtype=torch.float32),
        }


def make_subset(dataset: Dataset, size: int | None, seed: int = 0) -> Dataset:
    if size is None or size >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    return Subset(dataset, rng.choice(len(dataset), size=size, replace=False).tolist())

