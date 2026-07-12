"""Football-independent conditions for controlled 2D trajectory benchmarks."""

from __future__ import annotations

import numpy as np

from src.simulators.ou import PITCH_LENGTH, PITCH_WIDTH


def generate_condition_pool(
    n: int,
    steps: int,
    rng: np.random.Generator,
    *,
    min_target_distance: float = 8.0,
    max_target_distance: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate simple shared start, target, and fixed-segment conditions."""
    y0 = np.column_stack(
        [
            rng.uniform(20.0, PITCH_LENGTH - 20.0, size=n),
            rng.uniform(15.0, PITCH_WIDTH - 15.0, size=n),
        ]
    ).astype(np.float32)

    angle = rng.uniform(-np.pi, np.pi, size=n)
    distance = rng.uniform(min_target_distance, max_target_distance, size=n)
    displacement = np.column_stack([np.cos(angle), np.sin(angle)]) * distance[:, None]
    target = y0 + displacement.astype(np.float32)
    target[:, 0] = np.clip(target[:, 0], 0.0, PITCH_LENGTH)
    target[:, 1] = np.clip(target[:, 1], 0.0, PITCH_WIDTH)

    change_points = np.tile(
        np.asarray([round(steps / 3), round(2 * steps / 3)], dtype=np.int64),
        (n, 1),
    )
    return y0, target.astype(np.float32), change_points
