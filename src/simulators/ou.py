from __future__ import annotations

import numpy as np


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PARAMETER_NAMES = np.array(["k", "noise_scale"])
PARAMETER_LOW = np.array([0.05, 0.05], dtype=np.float32)
PARAMETER_HIGH = np.array([2.50, 4.00], dtype=np.float32)


def sample_ou_parameters(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample football OU baseline parameters.

    k is sampled uniformly because it is a relaxation/attraction rate.
    noise_scale is sampled log-uniformly because movement noise can vary over
    orders of magnitude and small values need adequate coverage.
    """
    params = np.zeros((n_samples, 2), dtype=np.float32)
    params[:, 0] = rng.uniform(PARAMETER_LOW[0], PARAMETER_HIGH[0], size=n_samples)
    log_noise = rng.uniform(
        np.log(PARAMETER_LOW[1]),
        np.log(PARAMETER_HIGH[1]),
        size=n_samples,
    )
    params[:, 1] = np.exp(log_noise)
    return params


def simulate_position_ou_batch(
    params: np.ndarray,
    y0: np.ndarray,
    target: np.ndarray,
    steps: int,
    dt: float,
    rng: np.random.Generator,
    clip_to_pitch: bool = True,
) -> np.ndarray:
    """
    Simulate 2D target-seeking OU tracks.

    State: (x, y)
    dx = k * (target_x - x) dt + noise_scale dW_x
    dy = k * (target_y - y) dt + noise_scale dW_y
    """
    params = np.asarray(params, dtype=np.float32)
    y0 = np.asarray(y0, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    n = len(params)
    tracks = np.zeros((n, steps, 2), dtype=np.float32)
    tracks[:, 0] = y0
    current = y0.copy()
    sqrt_dt = np.sqrt(dt)

    k = params[:, 0]
    noise = params[:, 1]

    for t in range(1, steps):
        drift = k[:, None] * (target - current)
        dw = rng.normal(0.0, sqrt_dt, size=(n, 2)).astype(np.float32)
        current = current + drift * dt + noise[:, None] * dw
        if clip_to_pitch:
            current[:, 0] = np.clip(current[:, 0], 0.0, PITCH_LENGTH)
            current[:, 1] = np.clip(current[:, 1], 0.0, PITCH_WIDTH)
        tracks[:, t] = current

    return tracks


def pitch_normalize_xy(xy: np.ndarray) -> np.ndarray:
    """Map metre coordinates to roughly [-1, 1] using pitch dimensions."""
    xy = np.asarray(xy, dtype=np.float32)
    scale = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)
    return (xy / scale) * 2.0 - 1.0


def pitch_denormalize_xy(xy_norm: np.ndarray) -> np.ndarray:
    """Map [-1, 1] pitch-normalized coordinates back to metres."""
    xy_norm = np.asarray(xy_norm, dtype=np.float32)
    scale = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)
    return ((xy_norm + 1.0) / 2.0) * scale
