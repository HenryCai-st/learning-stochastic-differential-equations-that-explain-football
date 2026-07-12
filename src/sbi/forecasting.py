"""Shared controlled posterior-predictive simulation and forecast metrics."""

from __future__ import annotations

import numpy as np

from src.simulators.model_voting import MAX_SEGMENTS, simulate_model_batch
from src.simulators.ou import PITCH_LENGTH, PITCH_WIDTH


def prepare_future_parameters(
    model_name: str,
    theta: np.ndarray,
    observed_steps: int,
    observed_change_points: np.ndarray,
) -> np.ndarray:
    """Apply the explicit no-unobserved-turn policy to future parameters."""
    prepared = np.asarray(theta, dtype=np.float32).copy()
    if model_name != "piecewise_velocity":
        return prepared
    latest_segment = int(np.sum(observed_steps - 1 >= np.asarray(observed_change_points)))
    latest_segment = int(np.clip(latest_segment, 0, MAX_SEGMENTS - 1))
    latest_velocity = prepared[:, 2 * latest_segment:2 * latest_segment + 2].copy()
    for segment in range(MAX_SEGMENTS):
        prepared[:, 2 * segment:2 * segment + 2] = latest_velocity
    return prepared


def simulate_future_batch(
    model_name: str,
    theta: np.ndarray,
    start: np.ndarray,
    target: np.ndarray,
    observed_steps: int,
    observed_change_points: np.ndarray,
    future_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate future points and return the exact parameters used."""
    prepared = prepare_future_parameters(
        model_name,
        theta,
        observed_steps,
        observed_change_points,
    )
    n = len(prepared)
    change_points = np.full((n, MAX_SEGMENTS - 1), future_steps + 1, dtype=np.int64)
    full = simulate_model_batch(
        model_name=model_name,
        params=prepared,
        y0=np.repeat(np.asarray(start, dtype=np.float32)[None], n, axis=0),
        target=np.repeat(np.asarray(target, dtype=np.float32)[None], n, axis=0),
        change_points=change_points,
        steps=future_steps + 1,
        dt=dt,
        rng=rng,
    )
    return full[:, 1:], prepared


def deterministic_baselines(prefix: np.ndarray, future_steps: int, dt: float) -> dict[str, np.ndarray]:
    """Return stationary, recent-velocity, and damped-velocity forecasts."""
    start = np.asarray(prefix[-1], dtype=np.float32)
    lag = min(5, len(prefix) - 1)
    velocity = (prefix[-1] - prefix[-1 - lag]) / max(lag * dt, 1e-8)
    times = dt * np.arange(1, future_steps + 1, dtype=np.float32)
    stationary = np.repeat(start[None], future_steps, axis=0)
    last_velocity = start[None] + times[:, None] * velocity[None]
    decay = np.log(2.0) / 0.5
    damped_distance = (1.0 - np.exp(-decay * times)) / decay
    damped_velocity = start[None] + damped_distance[:, None] * velocity[None]
    for prediction in (stationary, last_velocity, damped_velocity):
        prediction[:, 0] = np.clip(prediction[:, 0], 0.0, PITCH_LENGTH)
        prediction[:, 1] = np.clip(prediction[:, 1], 0.0, PITCH_WIDTH)
    return {
        "stationary": stationary.astype(np.float32),
        "last_velocity": last_velocity.astype(np.float32),
        "damped_velocity": damped_velocity.astype(np.float32),
    }


def ade_fde(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    error = np.linalg.norm(np.asarray(prediction) - np.asarray(truth), axis=-1)
    return float(error.mean()), float(error[-1])


def radial_coverage(
    paths: np.ndarray,
    truth: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict[str, dict[str, float]]:
    center = paths.mean(axis=0)
    sample_radius = np.linalg.norm(paths - center[None], axis=2)
    truth_radius = np.linalg.norm(truth - center, axis=1)
    output = {}
    for level in levels:
        radius = np.quantile(sample_radius, level, axis=0)
        covered = truth_radius <= radius
        output[str(int(level * 100))] = {
            "time_fraction": float(covered.mean()),
            "endpoint_covered": float(covered[-1]),
            "mean_radius": float(radius.mean()),
        }
    return output


def energy_score(paths: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> float:
    """Approximate the multivariate energy score using paired path samples."""
    flat_paths = np.asarray(paths, dtype=np.float64).reshape(len(paths), -1)
    flat_truth = np.asarray(truth, dtype=np.float64).reshape(1, -1)
    first = np.linalg.norm(flat_paths - flat_truth, axis=1).mean()
    permutation = rng.permutation(len(flat_paths))
    second = np.linalg.norm(flat_paths - flat_paths[permutation], axis=1).mean()
    return float(first - 0.5 * second)
