"""
Render one football tracking frame with a short future ball trail.

Inputs:
    - home and away tracking CSV files
    - frame number and trail length in frames

Outputs:
    - a static image at --out showing players, ball, and future ball trail.

Expected use:
    Use this for quick visual sanity checks of the CSV parser and selected
    match frame before extracting windows for SBI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.football.tracking import load_tracking
from src.football.visualization import visualize_tracking_frame


def main() -> None:
    """Load tracking CSVs, select one frame, and render a pitch image."""
    parser = argparse.ArgumentParser(description="Visualize football tracking data.")
    parser.add_argument("--home", default="data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv")
    parser.add_argument("--away", default="data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--t", type=int, default=25)
    parser.add_argument("--out", default="outputs/football_visualisation.png")
    args = parser.parse_args()

    df_home = load_tracking(args.home, team="home")
    df_away = load_tracking(args.away, team="away")

    print(f"Home: {len(df_home)} rows, frames {df_home['Frame'].min()}-{df_home['Frame'].max()}")
    print(f"Away: {len(df_away)} rows, frames {df_away['Frame'].min()}-{df_away['Frame'].max()}")

    visualize_tracking_frame(df_home, df_away, frame=args.frame, t=args.t, out_path=args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
