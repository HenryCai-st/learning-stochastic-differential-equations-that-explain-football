from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.football_tracking import PITCH_LENGTH, PITCH_WIDTH, denormalize, find_start_index, load_tracking
from src.utils.football_viz import BALL_COLOUR, TEAM_COLOURS, TRAJ_COLOUR, pitch_background


def resolve_game_dir(game: str | Path) -> Path:
    path = Path(game)
    if path.is_dir():
        return path
    data_path = Path("data") / str(game)
    if data_path.is_dir():
        return data_path
    raise FileNotFoundError(f"Game folder not found: {game}")


def tracking_paths(game_dir: Path) -> tuple[Path, Path]:
    game_name = game_dir.name
    home = game_dir / f"{game_name}_RawTrackingData_Home_Team.csv"
    away = game_dir / f"{game_name}_RawTrackingData_Away_Team.csv"
    if not home.exists() or not away.exists():
        raise FileNotFoundError(
            f"Expected tracking files not found in {game_dir}. "
            "Use a Sample_Game_x folder with Home/Away tracking CSV files."
        )
    return home, away


def player_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    xcols = [c for c in df.columns if c.endswith("_x") and not c.startswith("Ball")]
    return [(xcol, xcol.replace("_x", "_y")) for xcol in xcols]


def player_offsets(row: pd.Series, columns: list[tuple[str, str]]) -> np.ndarray:
    points = []
    for xcol, ycol in columns:
        px, py = row.get(xcol, np.nan), row.get(ycol, np.nan)
        if pd.isna(px) or pd.isna(py):
            continue
        mx, my = denormalize(float(px), float(py))
        points.append([float(mx), float(my)])
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def ball_xy(row_home: pd.Series, row_away: pd.Series) -> tuple[float, float] | None:
    row = row_home if "Ball_x" in row_home.index else row_away
    bx, by = row.get("Ball_x", np.nan), row.get("Ball_y", np.nan)
    if pd.isna(bx) or pd.isna(by):
        return None
    mx, my = denormalize(float(bx), float(by))
    return float(mx), float(my)


def select_indices(
    df: pd.DataFrame,
    start_time: float | None,
    start_frame: int | None,
    period: int | None,
    duration: float,
    frame_step: int,
) -> np.ndarray:
    frames = df["Frame"].to_numpy()
    times = df["Time [s]"].to_numpy()
    periods = df["Period"].to_numpy()
    start_idx = find_start_index(
        frames=frames,
        times=times,
        period=periods,
        start_time=start_time,
        start_frame=start_frame,
        chosen_period=period,
    )
    end_time = times[start_idx] + duration
    mask = (periods == periods[start_idx]) & (times >= times[start_idx]) & (times <= end_time)
    indices = np.where(mask)[0][::max(1, frame_step)]
    if len(indices) < 2:
        raise ValueError("Selected time window has fewer than two rendered frames.")
    return indices


def save_animation(anim: FuncAnimation, out_path: Path, fps: int, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    elif suffix == ".mp4":
        anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=dpi)
    else:
        raise ValueError("Output must end with .gif or .mp4")


def make_clip(
    df_home: pd.DataFrame,
    df_away: pd.DataFrame,
    indices: np.ndarray,
    out_path: Path,
    fps: int,
    dpi: int,
    trail_seconds: float | None,
) -> None:
    home_cols = player_columns(df_home)
    away_cols = player_columns(df_away)
    first_home = df_home.iloc[int(indices[0])]
    first_away = df_away.iloc[int(indices[0])]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#1a1a1a")
    pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)

    home_scatter = ax.scatter(
        [],
        [],
        s=105,
        c=TEAM_COLOURS["home"],
        edgecolors="white",
        linewidths=1.1,
        zorder=5,
        label="Home",
    )
    away_scatter = ax.scatter(
        [],
        [],
        s=105,
        c=TEAM_COLOURS["away"],
        edgecolors="white",
        linewidths=1.1,
        zorder=5,
        label="Away",
    )
    ball_scatter = ax.scatter(
        [],
        [],
        s=75,
        c=BALL_COLOUR,
        edgecolors="black",
        linewidths=0.9,
        zorder=7,
        label="Ball",
    )
    trail_line, = ax.plot([], [], color=TRAJ_COLOUR, linewidth=2.2, alpha=0.9, zorder=4, label="Ball trail")
    time_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=11,
        va="top",
        bbox={"facecolor": "#1a1a1a", "edgecolor": "white", "alpha": 0.82, "pad": 5},
    )
    ax.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="white", labelcolor="white")

    ball_history: list[tuple[float, float, float]] = []
    start_time = float(first_home["Time [s]"])
    start_frame = int(first_home["Frame"])

    def update(frame_i: int):
        idx = int(indices[frame_i])
        row_home = df_home.iloc[idx]
        row_away = df_away.iloc[idx]
        current_time = float(row_home["Time [s]"])

        home_scatter.set_offsets(player_offsets(row_home, home_cols))
        away_scatter.set_offsets(player_offsets(row_away, away_cols))

        current_ball = ball_xy(row_home, row_away)
        if current_ball is not None:
            ball_scatter.set_offsets(np.asarray([current_ball], dtype=np.float32))
            ball_history.append((current_ball[0], current_ball[1], current_time))
        else:
            ball_scatter.set_offsets(np.empty((0, 2), dtype=np.float32))

        if trail_seconds is None:
            trail = ball_history
        else:
            cutoff = current_time - trail_seconds
            trail = [point for point in ball_history if point[2] >= cutoff]

        if len(trail) >= 2:
            trail_arr = np.asarray([[x, y] for x, y, _ in trail], dtype=np.float32)
            trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])
        else:
            trail_line.set_data([], [])

        time_text.set_text(
            f"Period {int(row_home['Period'])} | Frame {int(row_home['Frame'])}\n"
            f"t = {current_time:.2f}s | window +{current_time - start_time:.2f}s"
        )
        return home_scatter, away_scatter, ball_scatter, trail_line, time_text

    ax.set_title(
        f"Ball tracking clip | start frame {start_frame} | start t={start_time:.2f}s",
        color="white",
        fontsize=13,
        pad=10,
    )
    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps, blit=False)
    save_animation(anim, out_path=out_path, fps=fps, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a short football tracking clip for a selected time window.")
    parser.add_argument("--game", default="data/Sample_Game_1", help="Sample game folder or name, e.g. data/Sample_Game_1")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--start-time", type=float, default=None, help="Start time in seconds.")
    parser.add_argument("--start-frame", type=int, default=None, help="Exact start frame.")
    parser.add_argument("--duration", type=float, default=5.0, help="Clip duration in seconds.")
    parser.add_argument("--frame-step", type=int, default=2, help="Render every n-th tracking frame.")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=None,
        help="Keep only the last N seconds of ball trail. Default keeps the full selected-window trail.",
    )
    parser.add_argument("--out", default="outputs/football_window_clip.gif")
    args = parser.parse_args()

    if args.start_time is None and args.start_frame is None:
        raise ValueError("Provide --start-time or --start-frame.")
    if args.start_time is not None and args.start_frame is not None:
        raise ValueError("Use either --start-time or --start-frame, not both.")

    game_dir = resolve_game_dir(args.game)
    home_path, away_path = tracking_paths(game_dir)
    df_home = load_tracking(home_path, "home")
    df_away = load_tracking(away_path, "away")
    indices = select_indices(
        df=df_home,
        start_time=args.start_time,
        start_frame=args.start_frame,
        period=args.period,
        duration=args.duration,
        frame_step=args.frame_step,
    )
    make_clip(
        df_home=df_home,
        df_away=df_away,
        indices=indices,
        out_path=Path(args.out),
        fps=args.fps,
        dpi=args.dpi,
        trail_seconds=args.trail_seconds,
    )
    print(f"Saved football tracking clip to {args.out}")


if __name__ == "__main__":
    main()
