"""
Create a football tracking GIF with a live model-voting gauge.

Inputs:
    - Sample_Game_x tracking data
    - a trained checkpoints/model_voting_ratio_best.pt classifier
    - a selected time window and sliding score-window duration

Outputs:
    - a GIF/MP4 showing the pitch, ball trail, and live soft votes over the
      candidate SDE families.

Expected use:
    Run this after one model-voting training session. The gauge is an evaluation
    visualization: it scores recent ball trajectories with the trained ratio
    classifier and shows which model family currently best matches the motion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tools.football_window_clip import (
    ball_xy,
    player_columns,
    player_offsets,
    resolve_game_dir,
    save_animation,
    select_indices,
    tracking_paths,
)
from src.sbi.scoring import checkpoint_array, load_checkpoint, score_params
from src.football.tracking import PITCH_LENGTH, PITCH_WIDTH, load_tracking
from src.football.segmentation import detect_change_points, fixed_even_change_points
from src.simulators.model_voting import MODEL_NAMES, pitch_normalize_condition, sample_model_parameters
from src.football.visualization import BALL_COLOUR, TEAM_COLOURS, TRAJ_COLOUR, pitch_background


MODEL_COLORS = {
    "brownian": "#e45756",
    "constant_velocity": "#4c78a8",
    "ou_target": "#54a24b",
    "piecewise_velocity": "#f58518",
}


def softmax(values: np.ndarray) -> np.ndarray:
    """Convert unnormalized model scores into stable probability-like weights."""
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / max(float(exp_values.sum()), 1e-8)


def normalized_track_tensor(track: np.ndarray, ckpt: dict, device: torch.device) -> torch.Tensor:
    """Normalize a pitch-metre trajectory using checkpoint statistics."""
    mean = checkpoint_array(ckpt, "track_mean").astype(np.float32)
    std = checkpoint_array(ckpt, "track_std").astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    norm = ((track - mean) / std).astype(np.float32)
    return torch.from_numpy(norm.T[None]).float().to(device)


def ball_track_from_rows(df_home, df_away, row_indices: np.ndarray) -> np.ndarray:
    """Extract and interpolate a ball trajectory from selected tracking rows."""
    points = []
    for idx in row_indices:
        point = ball_xy(df_home.iloc[int(idx)], df_away.iloc[int(idx)])
        if point is None:
            points.append([np.nan, np.nan])
        else:
            points.append([point[0], point[1]])
    track = np.asarray(points, dtype=np.float32)
    invalid = np.isnan(track).any(axis=1)
    if invalid.any():
        valid_idx = np.where(~invalid)[0]
        if len(valid_idx) < 2:
            raise ValueError("Sliding score window has fewer than two valid ball positions.")
        for dim in range(2):
            track[invalid, dim] = np.interp(np.where(invalid)[0], valid_idx, track[valid_idx, dim])
    return track


def score_one_window(
    model,
    ckpt: dict,
    device: torch.device,
    track: np.ndarray,
    dt: float,
    candidate_bank: dict[str, np.ndarray],
    max_segments: int,
    min_segment_len: int,
    top_k: int,
) -> tuple[np.ndarray, str]:
    """Return soft model-vote weights for one recent observed ball track."""
    cps = detect_change_points(track, dt=dt, max_segments=max_segments, min_segment_len=min_segment_len)
    if len(cps) < max_segments - 1:
        cps = fixed_even_change_points(len(track), max_segments=max_segments, min_segment_len=min_segment_len)
    padded_cps = np.zeros(max_segments - 1, dtype=np.int64)
    padded_cps[:min(len(cps), max_segments - 1)] = cps[:max_segments - 1]

    condition = pitch_normalize_condition(track[0], track[-1], padded_cps, len(track))
    track_t = normalized_track_tensor(track, ckpt, device)
    condition_t = torch.from_numpy(condition[None]).float().to(device)

    scores = []
    for model_name in MODEL_NAMES:
        logits = score_params(
            model=model,
            track_t=track_t,
            condition_t=condition_t,
            model_name=model_name,
            params=candidate_bank[model_name],
            device=device,
        )
        k = min(top_k, len(logits))
        scores.append(float(np.sort(logits)[-k:].mean()))
    weights = softmax(np.asarray(scores, dtype=np.float32))
    return weights, MODEL_NAMES[int(np.argmax(weights))]


def precompute_model_votes(
    df_home,
    df_away,
    indices: np.ndarray,
    checkpoint: Path,
    score_window_seconds: float,
    dt: float,
    n_candidates: int,
    top_k: int,
    score_every: int,
    max_segments: int,
    min_segment_len: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Precompute live model-vote weights for every rendered animation frame."""
    model, ckpt = load_checkpoint(checkpoint, device)
    rng = np.random.default_rng(seed)
    candidate_bank = {
        model_name: sample_model_parameters(model_name, n_candidates, rng)
        for model_name in MODEL_NAMES
    }

    times = df_home["Time [s]"].to_numpy()
    periods = df_home["Period"].to_numpy()
    votes = np.zeros((len(indices), len(MODEL_NAMES)), dtype=np.float32)
    winners: list[str] = []
    last_vote = np.ones(len(MODEL_NAMES), dtype=np.float32) / len(MODEL_NAMES)
    last_winner = MODEL_NAMES[0]

    for frame_i, idx in enumerate(indices):
        if frame_i % max(1, score_every) != 0 and frame_i > 0:
            votes[frame_i] = last_vote
            winners.append(last_winner)
            continue

        current_time = float(times[int(idx)])
        current_period = int(periods[int(idx)])
        start_time = current_time - score_window_seconds
        row_mask = (periods == current_period) & (times >= start_time) & (times <= current_time)
        row_indices = np.where(row_mask)[0]
        if len(row_indices) < 4:
            votes[frame_i] = last_vote
            winners.append(last_winner)
            continue

        track = ball_track_from_rows(df_home, df_away, row_indices)
        last_vote, last_winner = score_one_window(
            model=model,
            ckpt=ckpt,
            device=device,
            track=track,
            dt=dt,
            candidate_bank=candidate_bank,
            max_segments=max_segments,
            min_segment_len=min_segment_len,
            top_k=top_k,
        )
        votes[frame_i] = last_vote
        winners.append(last_winner)
        print(f"Scored frame {frame_i + 1}/{len(indices)}: best={last_winner}")

    return votes, winners


def make_model_voting_clip(
    df_home,
    df_away,
    indices: np.ndarray,
    votes: np.ndarray,
    winners: list[str],
    out_path: Path,
    fps: int,
    dpi: int,
    trail_seconds: float | None,
    score_window_seconds: float,
) -> None:
    """Render the tracking animation with a side panel for model-vote bars."""
    home_cols = player_columns(df_home)
    away_cols = player_columns(df_away)
    first_home = df_home.iloc[int(indices[0])]

    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor("#1a1a1a")
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.35], wspace=0.08)
    ax_pitch = fig.add_subplot(gs[0, 0])
    ax_gauge = fig.add_subplot(gs[0, 1])
    pitch_background(ax_pitch, length=PITCH_LENGTH, width=PITCH_WIDTH)

    home_scatter = ax_pitch.scatter([], [], s=105, c=TEAM_COLOURS["home"], edgecolors="white", linewidths=1.1, zorder=5, label="Home")
    away_scatter = ax_pitch.scatter([], [], s=105, c=TEAM_COLOURS["away"], edgecolors="white", linewidths=1.1, zorder=5, label="Away")
    ball_scatter = ax_pitch.scatter([], [], s=75, c=BALL_COLOUR, edgecolors="black", linewidths=0.9, zorder=7, label="Ball")
    trail_line, = ax_pitch.plot([], [], color=TRAJ_COLOUR, linewidth=2.2, alpha=0.9, zorder=4, label="Ball trail")
    time_text = ax_pitch.text(
        0.02,
        0.97,
        "",
        transform=ax_pitch.transAxes,
        color="white",
        fontsize=11,
        va="top",
        bbox={"facecolor": "#1a1a1a", "edgecolor": "white", "alpha": 0.82, "pad": 5},
    )
    ax_pitch.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
    ax_pitch.set_title("Football tracking with live model-voting gauge", color="white", fontsize=13, pad=10)

    ball_history: list[tuple[float, float, float]] = []
    start_time = float(first_home["Time [s]"])

    def draw_gauge(frame_i: int) -> None:
        """Redraw the side gauge for one animation frame."""
        ax_gauge.clear()
        ax_gauge.set_facecolor("#111827")
        values = votes[frame_i]
        y = np.arange(len(MODEL_NAMES))
        colors = [MODEL_COLORS[name] for name in MODEL_NAMES]
        ax_gauge.barh(y, values, color=colors, alpha=0.95)
        ax_gauge.set_yticks(y)
        ax_gauge.set_yticklabels(MODEL_NAMES, color="white", fontsize=9)
        ax_gauge.set_xlim(0.0, 1.0)
        ax_gauge.invert_yaxis()
        ax_gauge.tick_params(axis="x", colors="white")
        ax_gauge.grid(axis="x", color="white", alpha=0.18)
        for i, value in enumerate(values):
            ax_gauge.text(min(float(value) + 0.02, 0.96), i, f"{value:.2f}", color="white", va="center", fontsize=9)
        ax_gauge.set_title("Live model vote", color="white", fontsize=13, pad=8)
        ax_gauge.text(
            0.0,
            1.08,
            f"best: {winners[frame_i]}\nscore window: last {score_window_seconds:.1f}s",
            transform=ax_gauge.transAxes,
            color="#f9fafb",
            fontsize=10,
            va="bottom",
        )
        for spine in ax_gauge.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.35)

    def update(frame_i: int):
        """Update pitch markers, ball trail, text labels, and the gauge."""
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

        cutoff = -np.inf if trail_seconds is None else current_time - trail_seconds
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
        draw_gauge(frame_i)
        return home_scatter, away_scatter, ball_scatter, trail_line, time_text

    draw_gauge(0)
    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps, blit=False)
    save_animation(anim, out_path=out_path, fps=fps, dpi=dpi)
    plt.close(fig)


def main() -> None:
    """Parse CLI options, score sliding windows, and save the gauge clip."""
    parser = argparse.ArgumentParser(description="Create a football GIF with a live model-voting gauge.")
    parser.add_argument("--game", default="data/Sample_Game_1")
    parser.add_argument("--checkpoint", default="checkpoints/model_voting_ratio_best.pt")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--trail-seconds", type=float, default=2.0)
    parser.add_argument("--score-window-seconds", type=float, default=2.0)
    parser.add_argument("--score-every", type=int, default=2, help="Recompute model votes every N rendered frames.")
    parser.add_argument("--n-candidates", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--min-segment-len", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="outputs/football_model_voting_clip.gif")
    args = parser.parse_args()

    if args.start_time is None and args.start_frame is None:
        raise ValueError("Provide --start-time or --start-frame.")
    if args.start_time is not None and args.start_frame is not None:
        raise ValueError("Use either --start-time or --start-frame, not both.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    votes, winners = precompute_model_votes(
        df_home=df_home,
        df_away=df_away,
        indices=indices,
        checkpoint=Path(args.checkpoint),
        score_window_seconds=args.score_window_seconds,
        dt=args.dt,
        n_candidates=args.n_candidates,
        top_k=args.top_k,
        score_every=args.score_every,
        max_segments=args.max_segments,
        min_segment_len=args.min_segment_len,
        seed=args.seed,
        device=device,
    )
    make_model_voting_clip(
        df_home=df_home,
        df_away=df_away,
        indices=indices,
        votes=votes,
        winners=winners,
        out_path=Path(args.out),
        fps=args.fps,
        dpi=args.dpi,
        trail_seconds=args.trail_seconds,
        score_window_seconds=args.score_window_seconds,
    )
    print(f"Saved model-voting football clip to {args.out}")


if __name__ == "__main__":
    main()
