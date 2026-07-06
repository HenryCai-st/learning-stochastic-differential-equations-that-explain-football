from __future__ import annotations

import numpy as np


def smooth_positions(track: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Light moving-average smoothing for position-only tracking data.

    A radius of 1 uses a 3-frame window. This is intentionally conservative:
    enough to reduce frame jitter before differencing, but not enough to erase
    sharp football turns.
    """
    track = np.asarray(track, dtype=np.float32)
    if radius <= 0:
        return track.copy()

    padded = np.pad(track, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(track)
    width = 2 * radius + 1
    for i in range(len(track)):
        out[i] = padded[i:i + width].mean(axis=0)
    return out


def finite_difference_velocity(track: np.ndarray, dt: float) -> np.ndarray:
    """Return velocity estimates with the same length as `track`."""
    track = np.asarray(track, dtype=np.float32)
    vel = np.zeros_like(track)
    if len(track) < 2:
        return vel
    vel[1:] = (track[1:] - track[:-1]) / dt
    vel[0] = vel[1]
    return vel


def acceleration_from_velocity(velocity: np.ndarray, dt: float) -> np.ndarray:
    """Return finite-difference acceleration with the same length as velocity."""
    velocity = np.asarray(velocity, dtype=np.float32)
    acc = np.zeros_like(velocity)
    if len(velocity) < 2:
        return acc
    acc[1:] = (velocity[1:] - velocity[:-1]) / dt
    acc[0] = acc[1]
    return acc


def speed_from_velocity(velocity: np.ndarray) -> np.ndarray:
    return np.linalg.norm(velocity, axis=1)


def heading_from_velocity(velocity: np.ndarray, min_speed: float = 1e-6) -> np.ndarray:
    """Return heading angle in radians; slow frames get previous valid heading."""
    velocity = np.asarray(velocity, dtype=np.float32)
    speed = speed_from_velocity(velocity)
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    valid = speed > min_speed
    if not np.any(valid):
        return np.zeros(len(velocity), dtype=np.float32)

    first = int(np.argmax(valid))
    heading[:first] = heading[first]
    for i in range(first + 1, len(heading)):
        if not valid[i]:
            heading[i] = heading[i - 1]
    return heading.astype(np.float32)


def wrapped_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed angle difference a - b in radians."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def turn_angle_from_velocity(velocity: np.ndarray, min_speed: float = 0.2) -> np.ndarray:
    """
    Return absolute heading changes between consecutive velocity vectors.

    Low-speed frames are set to zero to avoid unstable angles when the ball is
    almost stationary.
    """
    speed = speed_from_velocity(velocity)
    heading = heading_from_velocity(velocity, min_speed=min_speed)
    turn = np.zeros(len(velocity), dtype=np.float32)
    if len(velocity) < 2:
        return turn
    turn[1:] = np.abs(wrapped_angle_diff(heading[1:], heading[:-1]))
    turn[(speed < min_speed)] = 0.0
    return turn


def trajectory_feature_dict(track: np.ndarray, dt: float, smooth_radius: int = 1) -> dict[str, np.ndarray]:
    """Compute position-derived features used for segmentation and diagnostics."""
    smoothed = smooth_positions(track, radius=smooth_radius)
    velocity = finite_difference_velocity(smoothed, dt=dt)
    acceleration = acceleration_from_velocity(velocity, dt=dt)
    speed = speed_from_velocity(velocity)
    heading = heading_from_velocity(velocity)
    turn_angle = turn_angle_from_velocity(velocity)
    return {
        "position": smoothed,
        "velocity": velocity,
        "acceleration": acceleration,
        "speed": speed,
        "heading": heading,
        "turn_angle": turn_angle,
    }

