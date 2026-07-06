from __future__ import annotations

import numpy as np

from src.data.trajectory_features import trajectory_feature_dict


def enforce_min_segment_length(change_points: list[int], steps: int, min_len: int) -> list[int]:
    """Remove change points that would create tiny neighboring segments."""
    accepted: list[int] = []
    last = 0
    for cp in sorted(change_points):
        if cp - last >= min_len and steps - cp >= min_len:
            accepted.append(int(cp))
            last = int(cp)
    return accepted


def detect_change_points(
    track: np.ndarray,
    dt: float,
    max_segments: int = 3,
    min_segment_len: int = 12,
    angle_threshold_rad: float = 0.65,
    acceleration_quantile: float = 0.92,
    smooth_radius: int = 1,
) -> np.ndarray:
    """
    Detect candidate piecewise-motion change points from position-only data.

    The score combines heading changes and acceleration spikes. For a first
    demo this is deliberately simple and interpretable.
    """
    steps = len(track)
    if steps < 2 * min_segment_len or max_segments <= 1:
        return np.array([], dtype=np.int64)

    features = trajectory_feature_dict(track, dt=dt, smooth_radius=smooth_radius)
    turn = features["turn_angle"]
    accel_mag = np.linalg.norm(features["acceleration"], axis=1)

    accel_cutoff = float(np.quantile(accel_mag, acceleration_quantile))
    score = turn / max(angle_threshold_rad, 1e-8)
    if accel_cutoff > 1e-8:
        score = score + accel_mag / accel_cutoff

    valid = np.zeros(steps, dtype=bool)
    valid[min_segment_len:steps - min_segment_len] = True
    candidate_idx = np.where(valid & ((turn >= angle_threshold_rad) | (accel_mag >= accel_cutoff)))[0]
    if len(candidate_idx) == 0:
        return fixed_even_change_points(steps, max_segments=max_segments, min_segment_len=min_segment_len)

    # Greedily pick high-scoring points while keeping segment lengths sane.
    ranked = sorted(candidate_idx.tolist(), key=lambda i: float(score[i]), reverse=True)
    selected: list[int] = []
    for idx in ranked:
        proposal = sorted(selected + [idx])
        boundaries = [0, *proposal, steps - 1]
        lengths = np.diff(boundaries)
        if np.all(lengths >= min_segment_len):
            selected.append(idx)
        if len(selected) >= max_segments - 1:
            break

    if len(selected) < max_segments - 1:
        fallback = fixed_even_change_points(steps, max_segments=max_segments, min_segment_len=min_segment_len)
        selected = sorted(set(selected).union(fallback.tolist()))[:max_segments - 1]

    return np.asarray(enforce_min_segment_length(selected, steps, min_segment_len), dtype=np.int64)


def fixed_even_change_points(steps: int, max_segments: int = 3, min_segment_len: int = 12) -> np.ndarray:
    """Fallback segmentation with evenly spaced change points."""
    if max_segments <= 1:
        return np.array([], dtype=np.int64)
    cps = [round(steps * i / max_segments) for i in range(1, max_segments)]
    cps = enforce_min_segment_length(cps, steps, min_segment_len)
    return np.asarray(cps, dtype=np.int64)


def segment_slices(steps: int, change_points: np.ndarray) -> list[slice]:
    boundaries = [0, *[int(cp) for cp in change_points], steps]
    return [slice(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def estimate_segment_velocities(track: np.ndarray, change_points: np.ndarray, dt: float, max_segments: int = 3) -> np.ndarray:
    """
    Estimate one constant velocity vector per segment.

    If fewer than max_segments are present, the last velocity is repeated so the
    parameter vector keeps a fixed size.
    """
    velocities: list[np.ndarray] = []
    for seg in segment_slices(len(track), change_points):
        part = track[seg]
        if len(part) < 2:
            velocity = np.zeros(2, dtype=np.float32)
        else:
            velocity = (part[-1] - part[0]) / (dt * max(len(part) - 1, 1))
        velocities.append(velocity.astype(np.float32))

    while len(velocities) < max_segments:
        velocities.append(velocities[-1].copy() if velocities else np.zeros(2, dtype=np.float32))
    return np.stack(velocities[:max_segments]).astype(np.float32)

