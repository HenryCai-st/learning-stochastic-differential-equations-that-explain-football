"""
Plot one real football window with detected piecewise segment boundaries.

Inputs:
    - data/real_football_windows.npz from extract_football_windows.py
    - a window index

Outputs:
    - an image showing the selected real trajectory split into colored segments.

Expected use:
    Use this to inspect whether the segmentation heuristic finds meaningful
    turn/change points before relying on piecewise-velocity model voting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.segmentation import detect_change_points, fixed_even_change_points
from src.sde.football_ou import PITCH_LENGTH, PITCH_WIDTH
from src.utils.football_viz import pitch_background


def main() -> None:
    """Load one real window, draw segment colors/change points, and save a plot."""
    parser = argparse.ArgumentParser(description="Plot one real football window with detected piecewise segments.")
    parser.add_argument("--real-windows", default="data/real_football_windows.npz")
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--out", default="outputs/real_window_segments.png")
    args = parser.parse_args()

    data = np.load(args.real_windows, allow_pickle=True)
    track = data["tracks"][args.window_index].astype(np.float32)
    dt = float(data["dt"]) if "dt" in data.files else 0.04
    if "change_points" in data.files:
        change_points = data["change_points"][args.window_index].astype(np.int64)
        change_points = change_points[(change_points > 0) & (change_points < len(track))]
    else:
        change_points = detect_change_points(track, dt=dt)
        if len(change_points) == 0:
            change_points = fixed_even_change_points(len(track))

    boundaries = [0, *[int(cp) for cp in change_points], len(track) - 1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]

    fig, ax = plt.subplots(figsize=(11, 7))
    pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)
    fig.patch.set_facecolor("#1a1a1a")

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        segment = track[start:end + 1]
        ax.plot(segment[:, 0], segment[:, 1], color=colors[i % len(colors)], linewidth=3, label=f"segment {i + 1}")

    for cp in change_points:
        ax.plot(track[cp, 0], track[cp, 1], "x", color="white", markersize=10, markeredgewidth=2.5)
        ax.text(track[cp, 0] + 1.0, track[cp, 1] + 1.0, f"cp={int(cp)}", color="white", fontsize=9)

    ax.plot(track[0, 0], track[0, 1], "o", color="#00bcd4", markersize=8, markeredgecolor="white", label="start")
    ax.plot(track[-1, 0], track[-1, 1], "s", color="#dc2626", markersize=8, markeredgecolor="white", label="end")
    ax.set_title("Real observed window with detected piecewise segments", color="white", fontsize=14, pad=10)
    ax.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="white", labelcolor="white")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved segment plot to {out}")


if __name__ == "__main__":
    main()
