from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sde.football_ou import PITCH_LENGTH, PITCH_WIDTH, simulate_position_ou_batch


MODEL_NAMES = ("brownian", "constant_velocity", "ou_target", "piecewise_velocity")
MODEL_TO_ID = {name: i for i, name in enumerate(MODEL_NAMES)}
MAX_PARAM_DIM = 7
CONDITION_DIM = 8
MAX_SEGMENTS = 3


@dataclass(frozen=True)
class ModelSpec:
    name: str
    param_dim: int
    low: np.ndarray
    high: np.ndarray
    log_scale: np.ndarray


MODEL_SPECS = {
    "brownian": ModelSpec(
        name="brownian",
        param_dim=1,
        low=np.array([0.05], dtype=np.float32),
        high=np.array([5.0], dtype=np.float32),
        log_scale=np.array([True]),
    ),
    "constant_velocity": ModelSpec(
        name="constant_velocity",
        param_dim=3,
        low=np.array([-35.0, -35.0, 0.05], dtype=np.float32),
        high=np.array([35.0, 35.0, 4.0], dtype=np.float32),
        log_scale=np.array([False, False, True]),
    ),
    "ou_target": ModelSpec(
        name="ou_target",
        param_dim=2,
        low=np.array([0.05, 0.05], dtype=np.float32),
        high=np.array([2.50, 4.00], dtype=np.float32),
        log_scale=np.array([False, True]),
    ),
    "piecewise_velocity": ModelSpec(
        name="piecewise_velocity",
        param_dim=7,
        low=np.array([-35.0, -35.0, -35.0, -35.0, -35.0, -35.0, 0.05], dtype=np.float32),
        high=np.array([35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 4.0], dtype=np.float32),
        log_scale=np.array([False, False, False, False, False, False, True]),
    ),
}


def pitch_normalize_condition(y0: np.ndarray, target: np.ndarray, change_points: np.ndarray, steps: int) -> np.ndarray:
    """
    Fixed-size condition vector:
        y0_norm(2), target_norm(2), normalized change points for 3-segment model(2),
        prefix placeholders(2)

    The last two slots are reserved for future velocity-prefix conditioning.
    """
    scale = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)
    y0_norm = (y0 / scale) * 2.0 - 1.0
    target_norm = (target / scale) * 2.0 - 1.0
    cp_norm = np.zeros(2, dtype=np.float32)
    for i, cp in enumerate(change_points[:2]):
        cp_norm[i] = float(cp) / max(steps - 1, 1)
    return np.concatenate([y0_norm, target_norm, cp_norm, np.zeros(2, dtype=np.float32)]).astype(np.float32)


def sample_model_parameters(model_name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    spec = MODEL_SPECS[model_name]
    params = np.zeros((n, spec.param_dim), dtype=np.float32)
    for i in range(spec.param_dim):
        if spec.log_scale[i]:
            params[:, i] = np.exp(rng.uniform(np.log(spec.low[i]), np.log(spec.high[i]), size=n))
        else:
            params[:, i] = rng.uniform(spec.low[i], spec.high[i], size=n)
    return params


def pad_parameters(model_name: str, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = MODEL_SPECS[model_name]
    params = np.asarray(params, dtype=np.float32)
    padded = np.zeros((len(params), MAX_PARAM_DIM), dtype=np.float32)
    mask = np.zeros((len(params), MAX_PARAM_DIM), dtype=np.float32)
    padded[:, :spec.param_dim] = params[:, :spec.param_dim]
    mask[:, :spec.param_dim] = 1.0
    return padded, mask


def normalize_padded_parameters(model_name: str, padded_params: np.ndarray) -> np.ndarray:
    spec = MODEL_SPECS[model_name]
    out = np.zeros_like(padded_params, dtype=np.float32)
    for i in range(spec.param_dim):
        value = padded_params[:, i].astype(np.float32)
        low = float(spec.low[i])
        high = float(spec.high[i])
        if spec.log_scale[i]:
            value = np.log(np.clip(value, low, None))
            low = float(np.log(low))
            high = float(np.log(high))
        out[:, i] = ((value - low) / max(high - low, 1e-8)) * 2.0 - 1.0
    return out


def simulate_model_batch(
    model_name: str,
    params: np.ndarray,
    y0: np.ndarray,
    target: np.ndarray,
    change_points: np.ndarray,
    steps: int,
    dt: float,
    rng: np.random.Generator,
    clip_to_pitch: bool = True,
) -> np.ndarray:
    if model_name == "brownian":
        return simulate_brownian(params, y0, steps, dt, rng, clip_to_pitch)
    if model_name == "constant_velocity":
        return simulate_constant_velocity(params, y0, steps, dt, rng, clip_to_pitch)
    if model_name == "ou_target":
        return simulate_position_ou_batch(params, y0, target, steps, dt, rng, clip_to_pitch)
    if model_name == "piecewise_velocity":
        return simulate_piecewise_velocity(params, y0, change_points, steps, dt, rng, clip_to_pitch)
    raise ValueError(f"Unknown model_name: {model_name}")


def _clip_pitch(current: np.ndarray) -> np.ndarray:
    current[:, 0] = np.clip(current[:, 0], 0.0, PITCH_LENGTH)
    current[:, 1] = np.clip(current[:, 1], 0.0, PITCH_WIDTH)
    return current


def simulate_brownian(params, y0, steps, dt, rng, clip_to_pitch=True):
    n = len(params)
    tracks = np.zeros((n, steps, 2), dtype=np.float32)
    tracks[:, 0] = y0
    current = y0.astype(np.float32).copy()
    noise = params[:, 0].astype(np.float32)
    sqrt_dt = np.sqrt(dt)
    for t in range(1, steps):
        current = current + noise[:, None] * rng.normal(0.0, sqrt_dt, size=(n, 2)).astype(np.float32)
        if clip_to_pitch:
            current = _clip_pitch(current)
        tracks[:, t] = current
    return tracks


def simulate_constant_velocity(params, y0, steps, dt, rng, clip_to_pitch=True):
    n = len(params)
    tracks = np.zeros((n, steps, 2), dtype=np.float32)
    tracks[:, 0] = y0
    current = y0.astype(np.float32).copy()
    velocity = params[:, :2].astype(np.float32)
    noise = params[:, 2].astype(np.float32)
    sqrt_dt = np.sqrt(dt)
    for t in range(1, steps):
        current = current + velocity * dt + noise[:, None] * rng.normal(0.0, sqrt_dt, size=(n, 2)).astype(np.float32)
        if clip_to_pitch:
            current = _clip_pitch(current)
        tracks[:, t] = current
    return tracks


def simulate_piecewise_velocity(params, y0, change_points, steps, dt, rng, clip_to_pitch=True):
    n = len(params)
    tracks = np.zeros((n, steps, 2), dtype=np.float32)
    tracks[:, 0] = y0
    current = y0.astype(np.float32).copy()
    velocities = params[:, :6].reshape(n, MAX_SEGMENTS, 2).astype(np.float32)
    noise = params[:, 6].astype(np.float32)
    sqrt_dt = np.sqrt(dt)
    cps = np.asarray(change_points, dtype=np.int64)
    if cps.ndim == 1:
        cps = np.repeat(cps[None, :], n, axis=0)
    for t in range(1, steps):
        seg_idx = np.zeros(n, dtype=np.int64)
        if cps.shape[1] > 0:
            seg_idx += (t >= cps[:, 0]).astype(np.int64)
        if cps.shape[1] > 1:
            seg_idx += (t >= cps[:, 1]).astype(np.int64)
        step_velocity = velocities[np.arange(n), np.clip(seg_idx, 0, MAX_SEGMENTS - 1), :]
        current = current + step_velocity * dt + noise[:, None] * rng.normal(0.0, sqrt_dt, size=(n, 2)).astype(np.float32)
        if clip_to_pitch:
            current = _clip_pitch(current)
        tracks[:, t] = current
    return tracks
