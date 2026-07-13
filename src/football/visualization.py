from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.football.tracking import PITCH_LENGTH, PITCH_WIDTH, denormalize


TEAM_COLOURS = {
    "home": "#e63946",
    "away": "#457b9d",
}
BALL_COLOUR = "#f9c74f"
TRAJ_COLOUR = "#f8961e"


def pitch_background(ax, length: float = PITCH_LENGTH, width: float = PITCH_WIDTH) -> None:
    """Draw a standard football pitch."""
    ax.set_facecolor("#2d7a2d")

    def rect(x, y, w, h):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=1.5,
            edgecolor="white",
            facecolor="none",
        ))

    rect(0, 0, length, width)
    ax.plot([length / 2, length / 2], [0, width], "white", lw=1.5)
    centre = plt.Circle((length / 2, width / 2), 9.15, color="white", fill=False, lw=1.5)
    ax.add_patch(centre)
    ax.plot(length / 2, width / 2, "wo", ms=3)

    for x0 in [0, length - 16.5]:
        rect(x0, (width - 40.32) / 2, 16.5, 40.32)
    for x0 in [0, length - 5.5]:
        rect(x0, (width - 18.32) / 2, 5.5, 18.32)

    ax.set_xlim(-3, length + 3)
    ax.set_ylim(-3, width + 3)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_players(ax, row: pd.Series, df_team: pd.DataFrame, colour: str) -> None:
    player_cols = [
        c for c in df_team.columns
        if c.endswith("_x") and not c.startswith("Ball")
    ]
    for xcol in player_cols:
        ycol = xcol.replace("_x", "_y")
        px, py = row.get(xcol, np.nan), row.get(ycol, np.nan)
        if pd.isna(px) or pd.isna(py):
            continue
        mx, my = denormalize(float(px), float(py))
        ax.plot(
            mx,
            my,
            "o",
            color=colour,
            markersize=14,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=4,
        )


def visualize_tracking_frame(
    df_home: pd.DataFrame,
    df_away: pd.DataFrame,
    frame: int,
    t: int,
    out_path: str | Path | None = None,
) -> None:
    """Render one match frame with both teams and the ball's next `t` frames."""

    def get_row(df: pd.DataFrame, label: str):
        rows = df[df["Frame"] == frame]
        if rows.empty:
            raise ValueError(f"Frame {frame} not found in {label} data.")
        return rows.iloc[0]

    row_home = get_row(df_home, "home")
    row_away = get_row(df_away, "away")

    def ball_traj(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["Frame"] >= frame) & (df["Frame"] <= frame + t)].copy()

    traj = ball_traj(df_home) if "Ball_x" in df_home.columns else ball_traj(df_away)
    ball_row = row_home if "Ball_x" in df_home.columns else row_away

    fig, ax = plt.subplots(figsize=(14, 9))
    pitch_background(ax)
    draw_players(ax, row_home, df_home, TEAM_COLOURS["home"])
    draw_players(ax, row_away, df_away, TEAM_COLOURS["away"])

    bx = ball_row.get("Ball_x", np.nan)
    by = ball_row.get("Ball_y", np.nan)
    if not (pd.isna(bx) or pd.isna(by)):
        mbx, mby = denormalize(float(bx), float(by))
        ax.plot(
            mbx,
            mby,
            "o",
            color=BALL_COLOUR,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
            zorder=6,
        )

        traj_x = traj["Ball_x"].values
        traj_y = traj["Ball_y"].values
        valid = ~(np.isnan(traj_x) | np.isnan(traj_y))
        tx, ty = traj_x[valid], traj_y[valid]
        if len(tx) > 1:
            mtx, mty = denormalize(tx, ty)
            n = len(mtx) - 1
            for seg in range(n):
                alpha = 0.9 * (1 - seg / n) + 0.1
                lw = max(1, 3 * (1 - seg / n))
                ax.plot(
                    mtx[seg:seg + 2],
                    mty[seg:seg + 2],
                    color=TRAJ_COLOUR,
                    alpha=alpha,
                    linewidth=lw,
                    zorder=3,
                )
            ax.scatter(mtx[1:], mty[1:], c=TRAJ_COLOUR, s=20, zorder=4, alpha=0.6)
            ax.annotate(
                "",
                xy=(mtx[-1], mty[-1]),
                xytext=(mtx[-2], mty[-2]),
                arrowprops=dict(arrowstyle="->", color=TRAJ_COLOUR, lw=2),
            )

    handles = [
        mpatches.Patch(color=TEAM_COLOURS["home"], label="Home team"),
        mpatches.Patch(color=TEAM_COLOURS["away"], label="Away team"),
        mpatches.Patch(color=BALL_COLOUR, label="Ball"),
        mpatches.Patch(color=TRAJ_COLOUR, label=f"Ball trajectory (+{t} frames)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=9,
        facecolor="#1a1a1a",
        edgecolor="white",
        labelcolor="white",
    )

    period = int(row_home["Period"])
    time_s = row_home["Time [s]"]
    ax.set_title(
        f"Frame {frame} | Period {period} | t = {time_s:.2f}s\n"
        f"Ball trajectory: next {t} frame(s)",
        fontsize=13,
        color="white",
        pad=10,
    )
    fig.patch.set_facecolor("#1a1a1a")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
