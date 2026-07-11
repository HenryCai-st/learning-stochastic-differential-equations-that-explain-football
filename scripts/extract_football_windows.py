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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fixed football trajectory windows for OU-SBI inference.")
    parser.add_argument("--home", default="data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv")
    parser.add_argument("--away", default="data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv")
    parser.add_argument("--team", choices=["home", "away"], default="home")
    parser.add_argument("--entity", default="Ball", help="Ball or PlayerN, for example Player7")
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.05)
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
    parser.add_argument("--out", default="data/real_football_windows.npz")
    args = parser.parse_args()

    if args.start_time is not None and args.start_frame is not None:
        raise ValueError("Use either --start-time or --start-frame, not both.")

    df = load_tracking(args.home, "home") if args.team == "home" else load_tracking(args.away, "away")
    steps = int(round(args.T / args.dt))
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
    np.savez_compressed(
        out,
        tracks=tracks,
        y0=y0s,
        target=targets,
        meta=meta,
        entity=args.entity,
        team=args.team,
        T=args.T,
        dt=args.dt,
        steps=steps,
        stride=args.stride,
        start_time=args.start_time,
        start_frame=args.start_frame,
        period_filter=args.period,
    )
    print(json.dumps({
        "out": str(out),
        "windows": int(len(tracks)),
        "steps": steps,
        "T": args.T,
        "selected_start": meta[0] if len(meta) == 1 else None,
    }, indent=2))


if __name__ == "__main__":
    main()
