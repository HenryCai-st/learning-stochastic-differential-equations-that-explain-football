"""
Candidate SDE simulators for the football model-voting SBI workflow.

All models describe the 2D ball position X_t = (x_t, y_t) on the pitch and are
simulated with an Euler-Maruyama update. The model-voting posterior estimates
both a discrete model family and a continuous parameter vector theta:

    brownian:
        dX_t = sigma dW_t
        theta = (sigma)

    constant_velocity:
        dX_t = v dt + sigma dW_t
        theta = (vx, vy, sigma)

    ou_target:
        dX_t = k(target - X_t) dt + sigma dW_t
        theta = (k, sigma)

    piecewise_velocity:
        dX_t = v_j dt + sigma dW_t, where j changes at two change points
        theta = (vx1, vy1, vx2, vy2, vx3, vy3, sigma)

Here dW_t is a 2D Brownian motion increment, approximated in code by
sqrt(dt) * Normal(0, I).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulators.ou import PITCH_LENGTH, PITCH_WIDTH, simulate_position_ou_batch


MODEL_NAMES = ("brownian", "constant_velocity", "ou_target", "piecewise_velocity")
MODEL_TO_ID = {name: i for i, name in enumerate(MODEL_NAMES)}
MAX_PARAM_DIM = 7
CONDITION_DIM = 8
MAX_SEGMENTS = 3
VELOCITY_PRIOR_ABS = 30.0


MODEL_PARAMETER_NAMES = {
    "brownian": ("sigma",),
    "constant_velocity": ("vx", "vy", "sigma"),
    "ou_target": ("k", "sigma"),
    "piecewise_velocity": ("vx1", "vy1", "vx2", "vy2", "vx3", "vy3", "sigma"),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    param_dim: int
    low: np.ndarray
    high: np.ndarray
    log_scale: np.ndarray


# Parameter bounds define the prior support used by the simulator and MCMC
# proposal normalization. Velocity units are pitch metres per second. The
# velocity bound is deliberately robust rather than maximal: Sample_Game ball
# speeds are mostly below 23 m/s at the 99th percentile, while the selected demo
# window reaches roughly 29 m/s before tracking-jump outliers.
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
        low=np.array([-VELOCITY_PRIOR_ABS, -VELOCITY_PRIOR_ABS, 0.05], dtype=np.float32),
        high=np.array([VELOCITY_PRIOR_ABS, VELOCITY_PRIOR_ABS, 4.0], dtype=np.float32),
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
        low=np.array(
            [
                -VELOCITY_PRIOR_ABS,
                -VELOCITY_PRIOR_ABS,
                -VELOCITY_PRIOR_ABS,
                -VELOCITY_PRIOR_ABS,
                -VELOCITY_PRIOR_ABS,
                -VELOCITY_PRIOR_ABS,
                0.05,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                VELOCITY_PRIOR_ABS,
                VELOCITY_PRIOR_ABS,
                VELOCITY_PRIOR_ABS,
                VELOCITY_PRIOR_ABS,
                VELOCITY_PRIOR_ABS,
                VELOCITY_PRIOR_ABS,
                4.0,
            ],
            dtype=np.float32,
        ),
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
    """Sample theta from the prior support of one candidate model family."""
    spec = MODEL_SPECS[model_name]
    params = np.zeros((n, spec.param_dim), dtype=np.float32)
    for i in range(spec.param_dim):
        if spec.log_scale[i]:
            params[:, i] = np.exp(rng.uniform(np.log(spec.low[i]), np.log(spec.high[i]), size=n))
        else:
            params[:, i] = rng.uniform(spec.low[i], spec.high[i], size=n)
    return params


def pad_parameters(model_name: str, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad a variable-length theta vector to the shared classifier input size."""
    spec = MODEL_SPECS[model_name]
    params = np.asarray(params, dtype=np.float32)
    padded = np.zeros((len(params), MAX_PARAM_DIM), dtype=np.float32)
    mask = np.zeros((len(params), MAX_PARAM_DIM), dtype=np.float32)
    padded[:, :spec.param_dim] = params[:, :spec.param_dim]
    mask[:, :spec.param_dim] = 1.0
    return padded, mask


def normalize_padded_parameters(model_name: str, padded_params: np.ndarray) -> np.ndarray:
    """Map active theta dimensions to roughly [-1, 1] for neural input features."""
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
    """Dispatch to the simulator associated with a sampled model family."""
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
    """Keep simulated positions inside the pitch rectangle."""
    current[:, 0] = np.clip(current[:, 0], 0.0, PITCH_LENGTH)
    current[:, 1] = np.clip(current[:, 1], 0.0, PITCH_WIDTH)
    return current


def simulate_brownian(params, y0, steps, dt, rng, clip_to_pitch=True):
    """
    Simulate the Brownian random-walk baseline.

    Math form:
        dX_t = sigma dW_t
        X_{t+dt} = X_t + sigma * sqrt(dt) * eps_t

    Parameter order:
        params[:, 0] = sigma
    """
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
    """
    Simulate a constant-velocity SDE.

    Math form:
        dX_t = v dt + sigma dW_t
        X_{t+dt} = X_t + v * dt + sigma * sqrt(dt) * eps_t

    Parameter order:
        params[:, 0] = vx, params[:, 1] = vy, params[:, 2] = sigma
    """
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
    """
    Simulate a three-segment piecewise constant-velocity SDE.

    Math form:
        dX_t = v_j dt + sigma dW_t
        X_{t+dt} = X_t + v_j * dt + sigma * sqrt(dt) * eps_t

    The segment index j is selected from the two supplied change points. This
    model is intended to represent straight ball movement with abrupt direction
    changes.

    Parameter order:
        params[:, 0:2] = (vx1, vy1)
        params[:, 2:4] = (vx2, vy2)
        params[:, 4:6] = (vx3, vy3)
        params[:, 6] = sigma
    """
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
