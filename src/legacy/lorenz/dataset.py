"""
dataset.py
==========
PyTorch Dataset for SDE trajectory data.
- Global mean/std normalization for tracks (preserves cross-track scale)
- Log-scale normalization for parameters via ParameterNormalizer
- Hard negative sampling via rho-sorted ranking with tunable window
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ParameterNormalizer:
    """Normalizes SDE parameters to [-1, 1] using appropriate scale per param."""

    def __init__(self):
        self.sigma_min, self.sigma_max = 1.0, 20.0
        self.rho_min,   self.rho_max   = 0.5, 50.0
        self.beta_min,  self.beta_max  = 0.5, 5.0
        self.noise_min, self.noise_max = 0.01, 0.5

    def normalize(self, params: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Input shape: (..., 4) — [sigma, rho, beta, noise_scale] → [-1, 1]"""
        is_tensor = torch.is_tensor(params)

        if is_tensor:
            d, dt = params.device, params.dtype
            def t(v): return torch.tensor(v, device=d, dtype=dt)
            sigma = (params[..., 0] - t(self.sigma_min)) / t(self.sigma_max - self.sigma_min)
            rho   = (torch.log(torch.clamp(params[..., 1], min=self.rho_min)) - t(np.log(self.rho_min))) \
                    / t(np.log(self.rho_max) - np.log(self.rho_min))
            beta  = (params[..., 2] - t(self.beta_min)) / t(self.beta_max - self.beta_min)
            noise = (torch.log(torch.clamp(params[..., 3], min=self.noise_min)) - t(np.log(self.noise_min))) \
                    / t(np.log(self.noise_max) - np.log(self.noise_min))
            return torch.stack([sigma, rho, beta, noise], dim=-1) * 2 - 1
        else:
            sigma = (params[..., 0] - self.sigma_min) / (self.sigma_max - self.sigma_min)
            rho   = (np.log(np.clip(params[..., 1], self.rho_min, None)) - np.log(self.rho_min)) \
                    / (np.log(self.rho_max) - np.log(self.rho_min))
            beta  = (params[..., 2] - self.beta_min) / (self.beta_max - self.beta_min)
            noise = (np.log(np.clip(params[..., 3], self.noise_min, None)) - np.log(self.noise_min)) \
                    / (np.log(self.noise_max) - np.log(self.noise_min))
            return np.stack([sigma, rho, beta, noise], axis=-1) * 2 - 1

    def denormalize(self, norm_params: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Input shape: (..., 4) — [-1, 1] → physical ranges"""
        is_tensor = torch.is_tensor(norm_params)

        if is_tensor:
            d, dt = norm_params.device, norm_params.dtype
            def t(v): return torch.tensor(v, device=d, dtype=dt)
            p = (norm_params + 1) / 2  # shift to [0, 1]
            sigma = p[..., 0] * t(self.sigma_max - self.sigma_min) + t(self.sigma_min)
            rho   = torch.exp(p[..., 1] * t(np.log(self.rho_max) - np.log(self.rho_min)) + t(np.log(self.rho_min)))
            beta  = p[..., 2] * t(self.beta_max  - self.beta_min)  + t(self.beta_min)
            noise = torch.exp(p[..., 3] * t(np.log(self.noise_max) - np.log(self.noise_min)) + t(np.log(self.noise_min)))
            return torch.stack([sigma, rho, beta, noise], dim=-1)
        else:
            p = (norm_params + 1) / 2
            sigma = p[..., 0] * (self.sigma_max - self.sigma_min) + self.sigma_min
            rho   = np.exp(p[..., 1] * (np.log(self.rho_max) - np.log(self.rho_min)) + np.log(self.rho_min))
            beta  = p[..., 2] * (self.beta_max  - self.beta_min)  + self.beta_min
            noise = np.exp(p[..., 3] * (np.log(self.noise_max) - np.log(self.noise_min)) + np.log(self.noise_min))
            return np.stack([sigma, rho, beta, noise], axis=-1)


class SDEDataset(Dataset):
    """
    Loads SDE trajectories and returns contrastive triplets.

    Each __getitem__ returns:
        query     : (3, steps) float32  — anchor track
        positive  : (3, steps) float32  — different track, same θ
        negative  : (3, steps) float32  — track from nearby-rho group (hard negative)
        params    : (4,)       float32  — normalized parameters of query
    """

    def __init__(
        self,
        data_path: str | Path,
        normalize_params: bool = True,
        hard_neg_window: int = 5,
    ):
        self.data_path = Path(data_path)
        self.normalize_params = normalize_params
        self.hard_neg_window = hard_neg_window
        self.normalizer = ParameterNormalizer()

        # ── Load data ──────────────────────────────────────────────────────────
        loaded = np.load(self.data_path / "dataset.npz")
        self.tracks     = loaded["tracks"]      # (total_tracks, steps, 3)
        self.parameters = loaded["parameters"]  # (total_tracks, 4)
        self.group_ids  = loaded["group_ids"]   # (total_tracks,)

        # ── Global track statistics (computed once, over all timesteps) ────────
        # axis=(-1,1): average over tracks and time, keep per-dimension (x,y,z)
        self.track_mean = self.tracks.mean(axis=(0, 1))  # (3,)
        self.track_std  = self.tracks.std(axis=(0, 1))   # (3,)

        # ── Group index mapping ────────────────────────────────────────────────
        self.group_to_indices = defaultdict(list)
        for idx, gid in enumerate(self.group_ids):
            self.group_to_indices[gid].append(idx)

        # ── Sort groups by rho for hard negative sampling ──────────────────────
        # Each group shares the same parameters, so just read rho from first member
        unique_groups = list(self.group_to_indices.keys())
        self.groups_by_rho = sorted(
            unique_groups,
            key=lambda g: self.parameters[self.group_to_indices[g][0], 1]  # col 1 = rho
        )
        # Rank lookup: group_id → position in rho-sorted list
        self.group_rho_rank = {g: i for i, g in enumerate(self.groups_by_rho)}

        print(f"Loaded {len(self.tracks)} tracks "
              f"from {len(self.group_to_indices)} parameter sets. "
              f"Track mean: {self.track_mean}, std: {self.track_std}")

    def _normalize_track(self, track: np.ndarray) -> np.ndarray:
        """Apply global mean/std normalization. Input: (steps, 3)."""
        return ((track - self.track_mean) / (self.track_std + 1e-8)).astype(np.float32)

    def _sample_hard_negative_group(self, group_id: int) -> int:
        """Return a group_id whose rho rank is within hard_neg_window of query."""
        rank = self.group_rho_rank[group_id]
        n    = len(self.groups_by_rho)
        lo   = max(0, rank - self.hard_neg_window)
        hi   = min(n - 1, rank + self.hard_neg_window)
        candidates = [
            self.groups_by_rho[r]
            for r in range(lo, hi + 1)
            if self.groups_by_rho[r] != group_id
        ]
        # Fall back to any other group if window is empty (tiny datasets)
        if not candidates:
            candidates = [g for g in self.groups_by_rho if g != group_id]
        return np.random.choice(candidates)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int):
        group_id = self.group_ids[idx]

        # ── Query ──────────────────────────────────────────────────────────────
        query = self._normalize_track(self.tracks[idx])

        # ── Positive: same group, different track ──────────────────────────────
        same = self.group_to_indices[group_id]
        pos_idx = np.random.choice([i for i in same if i != idx]) if len(same) > 1 else idx
        positive = self._normalize_track(self.tracks[pos_idx])

        # ── Hard negative: nearby rho, different group ─────────────────────────
        neg_group = self._sample_hard_negative_group(group_id)
        neg_idx   = np.random.choice(self.group_to_indices[neg_group])
        negative  = self._normalize_track(self.tracks[neg_idx])

        # ── To tensors — transpose to (3, steps) for Conv1d ───────────────────
        query_t    = torch.from_numpy(query.T)
        positive_t = torch.from_numpy(positive.T)
        negative_t = torch.from_numpy(negative.T)

        # ── Parameters ────────────────────────────────────────────────────────
        params = self.parameters[idx]
        if self.normalize_params:
            params = self.normalizer.normalize(params)
        params_t = torch.tensor(params, dtype=torch.float32)

        return query_t, positive_t, negative_t, params_t