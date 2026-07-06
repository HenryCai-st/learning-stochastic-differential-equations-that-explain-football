from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.football_tracking import extract_fixed_windows, entity_xy, load_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fixed football trajectory windows for OU-SBI inference.")
    parser.add_argument("--home", default="data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv")
    parser.add_argument("--away", default="data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv")
    parser.add_argument("--team", choices=["home", "away"], default="home")
    parser.add_argument("--entity", default="Ball", help="Ball or PlayerN, for example Player7")
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-gap-fraction", type=float, default=0.05)
    parser.add_argument("--out", default="data/real_football_windows.npz")
    args = parser.parse_args()

    df = load_tracking(args.home, "home") if args.team == "home" else load_tracking(args.away, "away")
    steps = int(round(args.T / args.dt))
    xy = entity_xy(df, args.entity)
    tracks, y0s, targets, meta = extract_fixed_windows(
        xy=xy,
        frames=df["Frame"].to_numpy(),
        period=df["Period"].to_numpy(),
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
    )
    print(json.dumps({"out": str(out), "windows": int(len(tracks)), "steps": steps}, indent=2))


if __name__ == "__main__":
    main()
