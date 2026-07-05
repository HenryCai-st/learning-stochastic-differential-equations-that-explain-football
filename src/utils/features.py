"""
src/utils/features.py
=====================
Trajectory summary features and parameter transforms used by both
the logistic-regression contrastive classifier and the MCMC log-target.

Public API
----------
summarize_trajectory(traj, max_points) -> np.ndarray (16,)
transform_params(params)               -> np.ndarray  (N, 4)  log-scaled
standardize_fit(x)                     -> (mean, std)
standardize(x, mean, std)              -> np.ndarray
pair_design_matrix(pf, tf)             -> np.ndarray  (N, F)
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory summary statistics  (16-dimensional hand-crafted features)
# ─────────────────────────────────────────────────────────────────────────────

def summarize_trajectory(traj: np.ndarray, max_points: int = 512) -> np.ndarray:
    """
    Compress a (T, 2) trajectory into a 16-dimensional feature vector.

    Features:
        [0-1]   mean_x, mean_y
        [2-3]   std_x, std_y
        [4-5]   min_x, min_y
        [6-7]   max_x, max_y
        [8-9]   displacement_x, displacement_y  (last - first)
        [10]    total path length
        [11]    mean step size
        [12]    std  step size
        [13]    max  step size
        [14-15] eigenvalues of position covariance (spread shape)
    """
    if len(traj) > max_points:
        pick = np.linspace(0, len(traj) - 1, max_points).astype(np.int64)
        traj = traj[pick]

    steps     = np.diff(traj, axis=0)
    step_norm = np.linalg.norm(steps, axis=1)
    displ     = traj[-1] - traj[0]
    centred   = traj - traj.mean(axis=0, keepdims=True)
    cov       = np.cov(centred.T)
    eigvals   = np.linalg.eigvalsh(cov)

    return np.array([
        traj[:, 0].mean(), traj[:, 1].mean(),
        traj[:, 0].std(),  traj[:, 1].std(),
        traj[:, 0].min(),  traj[:, 1].min(),
        traj[:, 0].max(),  traj[:, 1].max(),
        displ[0],          displ[1],
        step_norm.sum(),   step_norm.mean(), step_norm.std(), step_norm.max(),
        eigvals[0],        eigvals[1],
    ], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter transforms
# ─────────────────────────────────────────────────────────────────────────────

def transform_params(params: np.ndarray) -> np.ndarray:
    """
    Log-transform rho (col 1) and epsilon (col 3) to de-skew their distributions,
    then return a float64 copy.  All other columns are returned as-is.

    Input  shape: (N, 4)  or (4,)
    Output shape: same as input, dtype=float64
    """
    t = np.atleast_2d(params).astype(np.float64).copy()
    t[:, 1] = np.log1p(t[:, 1])   # log(1 + rho)
    t[:, 3] = np.log1p(t[:, 3])   # log(1 + epsilon)
    return t if params.ndim == 2 else t[0]


# ─────────────────────────────────────────────────────────────────────────────
# Standardisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column mean and std from training data."""
    mean = x.mean(axis=0)
    std  = x.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalisation using pre-computed mean/std."""
    return (x - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Pair design matrix  (used by logistic-regression ratio estimator)
# ─────────────────────────────────────────────────────────────────────────────

def pair_design_matrix(pf: np.ndarray, tf: np.ndarray) -> np.ndarray:
    """
    Build the feature vector for a (parameter, trajectory) pair.

    Concatenates:
      - param features        shape (N, P)
      - traj features         shape (N, F)
      - element-wise products shape (N, P*F)   ← interaction terms

    Interactions let a *linear* classifier learn compatibility without
    a neural encoder, approximating the role of a product kernel.

    Parameters
    ----------
    pf : (N, P) parameter feature matrix
    tf : (N, F) trajectory feature matrix

    Returns
    -------
    (N, P + F + P*F) design matrix
    """
    interactions = (pf[:, :, None] * tf[:, None, :]).reshape(len(pf), -1)
    return np.concatenate([pf, tf, interactions], axis=1)
