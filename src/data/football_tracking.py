from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def load_tracking(path: str | Path, team: str) -> pd.DataFrame:
    """
    Parse a Metrica-style tracking CSV with a 3-row header.

    Returns a tidy DataFrame with columns:
        Period, Frame, Time [s], Team,
        PlayerN_x, PlayerN_y, ..., Ball_x, Ball_y
    """
    raw = pd.read_csv(path, header=None, low_memory=False)
    header_row = raw.iloc[2].tolist()

    col_names: list[str] = []
    current_entity: str | None = None
    for value in header_row:
        if pd.isna(value):
            col_names.append(f"{current_entity}_y")
        else:
            current_entity = str(value).strip()
            if current_entity in ("Period", "Frame", "Time [s]"):
                col_names.append(current_entity)
            else:
                col_names.append(f"{current_entity}_x")

    df = raw.iloc[3:].reset_index(drop=True)
    df.columns = col_names
    df = df.apply(pd.to_numeric, errors="coerce")
    df["Frame"] = df["Frame"].astype(int)
    df["Period"] = df["Period"].astype(int)
    df["Team"] = team
    return df


def denormalize(
    x: np.ndarray | float,
    y: np.ndarray | float,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
):
    """Convert Metrica normalized [0, 1] coordinates to pitch metres."""
    return np.asarray(x) * length, np.asarray(y) * width


def available_entities(df: pd.DataFrame) -> list[str]:
    return sorted(c[:-2] for c in df.columns if c.endswith("_x"))


def entity_xy(df: pd.DataFrame, entity: str) -> np.ndarray:
    """Return one entity trajectory in metres as `(frames, 2)`."""
    xcol = f"{entity}_x"
    ycol = f"{entity}_y"
    if xcol not in df.columns or ycol not in df.columns:
        examples = available_entities(df)[:20]
        raise ValueError(f"Entity {entity!r} not found. Available examples: {examples}")
    x, y = denormalize(df[xcol].to_numpy(dtype=float), df[ycol].to_numpy(dtype=float))
    return np.stack([x, y], axis=1).astype(np.float32)


def extract_fixed_windows(
    xy: np.ndarray,
    frames: np.ndarray,
    period: np.ndarray,
    steps: int,
    stride: int,
    max_gap_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice a trajectory into fixed windows.

    Returns:
        tracks:  (N, steps, 2)
        y0:      (N, 2)
        target:  (N, 2), currently the final point of each window
        meta:    object array with frame/period metadata
    """
    tracks, y0s, targets, meta = [], [], [], []
    for start in range(0, len(xy) - steps + 1, stride):
        window = xy[start:start + steps].copy()
        invalid = np.isnan(window).any(axis=1)
        if invalid.mean() > max_gap_fraction:
            continue

        if invalid.any():
            valid_idx = np.where(~invalid)[0]
            if len(valid_idx) < 2:
                continue
            for dim in range(2):
                window[invalid, dim] = np.interp(
                    np.where(invalid)[0],
                    valid_idx,
                    window[valid_idx, dim],
                )

        tracks.append(window.astype(np.float32))
        y0s.append(window[0].astype(np.float32))
        targets.append(window[-1].astype(np.float32))
        meta.append({
            "start_row": int(start),
            "start_frame": int(frames[start]),
            "end_frame": int(frames[start + steps - 1]),
            "period": int(period[start]),
        })

    return (
        np.asarray(tracks, dtype=np.float32),
        np.asarray(y0s, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(meta, dtype=object),
    )


def extract_single_window(
    xy: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    period: np.ndarray,
    steps: int,
    start_index: int,
    max_gap_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Extract one fixed window from a chosen row index.

    This is used when the user wants a specific observed interval, for example
    "start at 37.2 seconds for a 5 second window" instead of scanning all
    possible windows.
    """
    if start_index < 0 or start_index + steps > len(xy):
        raise ValueError(
            f"Chosen start index {start_index} cannot provide {steps} steps "
            f"from trajectory length {len(xy)}."
        )

    window = xy[start_index:start_index + steps].copy()
    invalid = np.isnan(window).any(axis=1)
    if invalid.mean() > max_gap_fraction:
        raise ValueError(
            "Chosen window has too many missing positions. "
            "Try another start time/frame or increase --max-gap-fraction."
        )

    if invalid.any():
        valid_idx = np.where(~invalid)[0]
        if len(valid_idx) < 2:
            raise ValueError("Chosen window has fewer than two valid positions.")
        for dim in range(2):
            window[invalid, dim] = np.interp(
                np.where(invalid)[0],
                valid_idx,
                window[valid_idx, dim],
            )

    meta = {
        "start_row": int(start_index),
        "start_frame": int(frames[start_index]),
        "end_frame": int(frames[start_index + steps - 1]),
        "start_time_s": float(times[start_index]),
        "end_time_s": float(times[start_index + steps - 1]),
        "period": int(period[start_index]),
    }
    return (
        window[None].astype(np.float32),
        window[None, 0].astype(np.float32),
        window[None, -1].astype(np.float32),
        meta,
    )


def find_start_index(
    frames: np.ndarray,
    times: np.ndarray,
    period: np.ndarray,
    start_time: float | None = None,
    start_frame: int | None = None,
    chosen_period: int | None = None,
) -> int:
    """
    Locate the row index closest to a requested start time or exact frame.

    If period is provided, the search is restricted to that match period.
    """
    mask = np.ones(len(frames), dtype=bool)
    if chosen_period is not None:
        mask &= period == chosen_period
    candidate_idx = np.where(mask)[0]
    if len(candidate_idx) == 0:
        raise ValueError(f"No rows found for period={chosen_period}.")

    if start_frame is not None:
        frame_matches = candidate_idx[frames[candidate_idx] == start_frame]
        if len(frame_matches) == 0:
            raise ValueError(f"Frame {start_frame} not found for the selected period.")
        return int(frame_matches[0])

    if start_time is not None:
        local_times = times[candidate_idx]
        return int(candidate_idx[np.argmin(np.abs(local_times - start_time))])

    raise ValueError("Provide either start_time or start_frame.")
