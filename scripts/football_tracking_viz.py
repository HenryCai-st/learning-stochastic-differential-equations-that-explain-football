"""
Football Tracking Data Visualizer
==================================
Visualizes player positions for both home and away teams, and ball trajectory
at a given frame, with the ball's path over the next `t` frames.

Usage:
    python football_tracking_viz.py \
        --home  path/to/home_tracking.csv  \
        --away  path/to/away_tracking.csv  \
        --frame 1 \
        --t 10

Column structure of each raw CSV:
    Row 0: team labels (Home/Away repeated)
    Row 1: player numbers
    Row 2: readable header — Period, Frame, Time [s], PlayerN, NaN, ...
              each player occupies TWO columns: X (named) then Y (NaN)
    Row 3+: data rows
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Team colour scheme ────────────────────────────────────────────────────────
TEAM_COLOURS = {
    "home": "#e63946",   # red
    "away": "#457b9d",   # blue
}
BALL_COLOUR  = "#f9c74f"   # yellow
TRAJ_COLOUR  = "#f8961e"   # orange


# ── helpers ───────────────────────────────────────────────────────────────────

def load_tracking(path: str, team: str) -> pd.DataFrame:
    """
    Parse the Metrica-style tracking CSV with a 3-row header.

    Parameters
    ----------
    path : str
        Path to CSV file.
    team : str
        'home' or 'away' — stored in a 'Team' column for reference.

    Returns a tidy DataFrame with columns:
        Period, Frame, Time [s], Team,
        PlayerN_x, PlayerN_y, ..., Ball_x, Ball_y
    """
    raw = pd.read_csv(path, header=None)

    # Row 2 has real column names; Y columns for each entity are NaN.
    header_row = raw.iloc[2].tolist()

    col_names = []
    current_entity = None
    for val in header_row:
        if pd.isna(val):
            col_names.append(f"{current_entity}_y")
        else:
            current_entity = str(val).strip()
            if current_entity in ("Period", "Frame", "Time [s]"):
                col_names.append(current_entity)
            else:
                col_names.append(f"{current_entity}_x")

    df = raw.iloc[3:].reset_index(drop=True)
    df.columns = col_names
    df = df.apply(pd.to_numeric, errors="coerce")
    df["Frame"]  = df["Frame"].astype(int)
    df["Period"] = df["Period"].astype(int)
    df["Team"]   = team          # tag so visualise() knows which colour to use

    return df


def pitch_background(ax, length=105, width=68):
    """Draw a standard football pitch."""
    ax.set_facecolor("#2d7a2d")

    def rect(x, y, w, h):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=1.5, edgecolor="white", facecolor="none"
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


def denormalize(x, y, length=105, width=68):
    """Normalized [0,1] → metres (Metrica Sports convention)."""
    return x * length, y * width


def _draw_players(ax, row, df_team, colour):
    """Plot all players for one team at the given frame row."""
    player_cols = [c for c in df_team.columns if c.endswith("_x")
                   and not c.startswith("Ball")]
    for xcol in player_cols:
        ycol = xcol.replace("_x", "_y")
        px, py = row.get(xcol, np.nan), row.get(ycol, np.nan)
        if pd.isna(px) or pd.isna(py):
            continue
        mx, my = denormalize(float(px), float(py))
        pid = xcol.replace("Player", "").replace("_x", "")
        ax.plot(mx, my, "o", color=colour, markersize=14,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        # ax.text(mx, my, pid, ha="center", va="center",
        #         fontsize=7, fontweight="bold", color="white", zorder=5)


# ── main visualisation ────────────────────────────────────────────────────────

def visualize(df_home: pd.DataFrame, df_away: pd.DataFrame,
              frame: int, t: int, out_path: str = None):
    """
    Render one frame with both teams and the ball trajectory.

    Ball data is taken from whichever DataFrame has it for this frame
    (both files carry identical ball columns — home file is preferred).
    """

    # ── locate frame in both DataFrames ──
    def get_row(df, label):
        r = df[df["Frame"] == frame]
        if r.empty:
            avail = df["Frame"].tolist()
            raise ValueError(f"Frame {frame} not found in {label} data. "
                             f"Available frames: {avail}")
        return r.iloc[0]

    row_home = get_row(df_home, "home")
    row_away = get_row(df_away, "away")

    # Ball trajectory — prefer home df; fall back to away df
    def ball_traj(df):
        return df[(df["Frame"] >= frame) & (df["Frame"] <= frame + t)].copy()

    traj = ball_traj(df_home) if "Ball_x" in df_home.columns else ball_traj(df_away)
    ball_row = row_home if "Ball_x" in df_home.columns else row_away

    # ── plot ──
    fig, ax = plt.subplots(figsize=(14, 9))
    pitch_background(ax)

    # Draw home players (red)
    _draw_players(ax, row_home, df_home, TEAM_COLOURS["home"])
    # Draw away players (blue)
    _draw_players(ax, row_away, df_away, TEAM_COLOURS["away"])

    # ── ball (current position) ──
    bx = ball_row.get("Ball_x", np.nan)
    by = ball_row.get("Ball_y", np.nan)
    if not (pd.isna(bx) or pd.isna(by)):
        mbx, mby = denormalize(float(bx), float(by))
        ax.plot(mbx, mby, "o", color=BALL_COLOUR, markersize=8,
                markeredgecolor="black", markeredgewidth=1, zorder=6)

        # ── ball trajectory ──
        traj_x = traj["Ball_x"].values
        traj_y = traj["Ball_y"].values
        valid  = ~(np.isnan(traj_x) | np.isnan(traj_y))
        tx, ty = traj_x[valid], traj_y[valid]
        if len(tx) > 1:
            mtx, mty = denormalize(tx, ty)
            n = len(mtx) - 1
            for seg in range(n):
                alpha = 0.9 * (1 - seg / n) + 0.1
                lw    = max(1, 3 * (1 - seg / n))
                ax.plot(mtx[seg:seg+2], mty[seg:seg+2],
                        color=TRAJ_COLOUR, alpha=alpha, linewidth=lw, zorder=3)
            ax.scatter(mtx[1:], mty[1:], c=TRAJ_COLOUR, s=20, zorder=4, alpha=0.6)
            ax.annotate("", xy=(mtx[-1], mty[-1]), xytext=(mtx[-2], mty[-2]),
                        arrowprops=dict(arrowstyle="->", color=TRAJ_COLOUR, lw=2))

    # ── legend ──
    home_patch = mpatches.Patch(color=TEAM_COLOURS["home"], label="Home team")
    away_patch = mpatches.Patch(color=TEAM_COLOURS["away"], label="Away team")
    ball_patch = mpatches.Patch(color=BALL_COLOUR,           label="Ball (current frame)")
    traj_patch = mpatches.Patch(color=TRAJ_COLOUR,           label=f"Ball trajectory (+{t} frames)")
    ax.legend(handles=[home_patch, away_patch, ball_patch, traj_patch],
              loc="upper left", fontsize=9,
              facecolor="#1a1a1a", edgecolor="white", labelcolor="white")

    period = int(row_home["Period"])
    time_s = row_home["Time [s]"]
    ax.set_title(
        f"Frame {frame}  |  Period {period}  |  t = {time_s:.2f}s\n"
        f"Ball trajectory: next {t} frame(s)",
        fontsize=13, color="white", pad=10
    )
    fig.patch.set_facecolor("#1a1a1a")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize football tracking data (both teams).")
    parser.add_argument("--home",  default="data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv", help="Path to home team tracking CSV")
    parser.add_argument("--away",  default="data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv", help="Path to away team tracking CSV")
    parser.add_argument("--frame", type=int, default=1,  help="Frame number to visualise")
    parser.add_argument("--t",     type=int, default=5,  help="Future frames for ball trajectory")
    parser.add_argument("--out",   default="outputs/football_visualisation.png",         help="Save path (omit to show interactively)")
    args = parser.parse_args()

    df_home = load_tracking(args.home, team="home")
    df_away = load_tracking(args.away, team="away")

    print(f"Home: {len(df_home)} rows, frames {df_home['Frame'].min()}–{df_home['Frame'].max()}")
    print(f"Away: {len(df_away)} rows, frames {df_away['Frame'].min()}–{df_away['Frame'].max()}")

    visualize(df_home, df_away, frame=args.frame, t=args.t, out_path=args.out)
