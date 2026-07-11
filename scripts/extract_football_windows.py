"""
Extract real football tracking windows from Metrica-style sample-game CSV files.

Inputs:
    - home/away tracking CSV files from data/Sample_Game_x
    - entity name such as Ball or Player7
    - scan mode using stride, or one selected start time/frame

Outputs:
    - data/real_football_windows.npz containing tracks, y0, target, metadata,
      change points, diagnostics, and optional prefix/suffix splits.

Expected use:
    Run this before model-voting data generation. With --T 5.0 and
    --prefix-T 2.0, the first 2 seconds are observed and the next 3 seconds are
    held out for evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.football_tracking import (
    extract_fixed_windows,
    extract_single_window,
    entity_xy,
    find_start_index,
    load_tracking,
)
from src.data.segmentation import detect_change_points, fixed_even_change_points
from src.data.trajectory_features import trajectory_diagnostics


def main() -> None:
    """Parse CLI arguments, extract valid windows, and save them as `.npz`."""
    parser = argparse.ArgumentParser(description="Extract fixed football trajectory windows for football SBI inference.")
    parser.add_argument("--home", default="data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv")
    parser.add_argument("--away", default="data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv")
    parser.add_argument("--team", choices=["home", "away"], default="home")
    parser.add_argument("--entity", default="Ball", help="Ball or PlayerN, for example Player7")
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--prefix-T",
        type=float,
        default=None,
        help="Optional observed-prefix duration in seconds. The remaining window is saved as future suffix.",
    )
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Optional start time in seconds. If set, extract one window starting near this time.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Optional exact start frame. If set, extract one window starting at this frame.",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=None,
        help="Optional match period used with --start-time or --start-frame.",
    )
    parser.add_argument("--max-gap-fraction", type=float, default=0.05)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--min-segment-len", type=int, default=12)
    parser.add_argument("--jump-speed-threshold", type=float, default=45.0)
    parser.add_argument("--out", default="data/real_football_windows.npz")
    args = parser.parse_args()

    if args.start_time is not None and args.start_frame is not None:
        raise ValueError("Use either --start-time or --start-frame, not both.")

    df = load_tracking(args.home, "home") if args.team == "home" else load_tracking(args.away, "away")
    steps = int(round(args.T / args.dt))
    prefix_steps = None if args.prefix_T is None else int(round(args.prefix_T / args.dt))
    if prefix_steps is not None and not (1 < prefix_steps < steps):
        raise ValueError("--prefix-T must be larger than dt and smaller than --T.")
    xy = entity_xy(df, args.entity)
    frames = df["Frame"].to_numpy()
    times = df["Time [s]"].to_numpy()
    period = df["Period"].to_numpy()

    if args.start_time is not None or args.start_frame is not None:
        start_index = find_start_index(
            frames=frames,
            times=times,
            period=period,
            start_time=args.start_time,
            start_frame=args.start_frame,
            chosen_period=args.period,
        )
        tracks, y0s, targets, one_meta = extract_single_window(
            xy=xy,
            frames=frames,
            times=times,
            period=period,
            steps=steps,
            start_index=start_index,
            max_gap_fraction=args.max_gap_fraction,
        )
        meta = np.asarray([one_meta], dtype=object)
    else:
        tracks, y0s, targets, meta = extract_fixed_windows(
            xy=xy,
            frames=frames,
            period=period,
            steps=steps,
            stride=args.stride,
            max_gap_fraction=args.max_gap_fraction,
        )
    if len(tracks) == 0:
        raise ValueError("No usable windows extracted. Try a smaller T, smaller stride, or larger gap tolerance.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    change_points = []
    diagnostics = []
    for track in tracks:
        cps = detect_change_points(
            track,
            dt=args.dt,
            max_segments=args.max_segments,
            min_segment_len=args.min_segment_len,
        )
        if len(cps) < args.max_segments - 1:
            cps = fixed_even_change_points(
                len(track),
                max_segments=args.max_segments,
                min_segment_len=args.min_segment_len,
            )
        padded = np.zeros(args.max_segments - 1, dtype=np.int64)
        padded[:min(len(cps), args.max_segments - 1)] = cps[:args.max_segments - 1]
        change_points.append(padded)
        diagnostics.append(trajectory_diagnostics(track, dt=args.dt, jump_speed_threshold=args.jump_speed_threshold))

    extra_payload = {}
    if prefix_steps is not None:
        extra_payload = {
            "prefix_tracks": tracks[:, :prefix_steps].astype(np.float32),
            "suffix_tracks": tracks[:, prefix_steps:].astype(np.float32),
            "prefix_y0": tracks[:, 0].astype(np.float32),
            "prefix_target": tracks[:, prefix_steps - 1].astype(np.float32),
            "suffix_target": tracks[:, -1].astype(np.float32),
            "prefix_T": float(args.prefix_T),
            "suffix_T": float(args.T - args.prefix_T),
            "prefix_steps": int(prefix_steps),
            "suffix_steps": int(steps - prefix_steps),
        }
    np.savez_compressed(
        out,
        tracks=tracks,
        y0=y0s,
        target=targets,
        meta=meta,
        change_points=np.asarray(change_points, dtype=np.int64),
        diagnostics=np.asarray(diagnostics, dtype=object),
        entity=args.entity,
        team=args.team,
        T=args.T,
        dt=args.dt,
        steps=steps,
        max_segments=args.max_segments,
        min_segment_len=args.min_segment_len,
        stride=args.stride,
        start_time=args.start_time,
        start_frame=args.start_frame,
        period_filter=args.period,
        **extra_payload,
    )
    print(json.dumps({
        "out": str(out),
        "windows": int(len(tracks)),
        "steps": steps,
        "T": args.T,
        "prefix_steps": prefix_steps,
        "selected_start": meta[0] if len(meta) == 1 else None,
    }, indent=2))


if __name__ == "__main__":
    main()
