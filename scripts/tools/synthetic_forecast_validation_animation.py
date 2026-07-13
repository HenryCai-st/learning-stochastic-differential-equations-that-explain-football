"""Animate four model forecasts against one held-out synthetic trajectory."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.football.visualization import pitch_background
from src.sbi.artifacts import validate_checkpoint_contract
from src.sbi.evidence import softmax
from src.sbi.forecasting import simulate_future_batch
from src.sbi.scoring import load_checkpoint, normalize_track, score_params
from src.simulators.model_voting import MODEL_NAMES, sample_model_parameters
from src.simulators.ou import PITCH_LENGTH, PITCH_WIDTH


MODEL_COLORS = {
    "brownian": "#e45756",
    "constant_velocity": "#4c78a8",
    "ou_target": "#54a24b",
    "piecewise_velocity": "#f58518",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_case(
    prefixes: np.ndarray,
    suffixes: np.ndarray,
    weights: np.ndarray,
    true_model_ids: np.ndarray,
    metric_rows: list[dict[str, str]],
) -> int:
    """Choose an interior, dynamic case with a confident and accurate forecast."""
    interior = (
        (prefixes[:, :, 0] > 2.0).all(axis=1)
        & (prefixes[:, :, 0] < PITCH_LENGTH - 2.0).all(axis=1)
        & (prefixes[:, :, 1] > 2.0).all(axis=1)
        & (prefixes[:, :, 1] < PITCH_WIDTH - 2.0).all(axis=1)
        & (suffixes[:, :, 0] > 2.0).all(axis=1)
        & (suffixes[:, :, 0] < PITCH_LENGTH - 2.0).all(axis=1)
        & (suffixes[:, :, 1] > 2.0).all(axis=1)
        & (suffixes[:, :, 1] < PITCH_WIDTH - 2.0).all(axis=1)
    )
    selected_ids = np.argmax(weights, axis=1)
    correct = selected_ids == true_model_ids
    candidates = np.flatnonzero(interior & correct)
    if len(candidates) == 0:
        candidates = np.flatnonzero(correct)
    errors = np.asarray([float(metric_rows[index]["sbi_ade"]) for index in candidates])
    future_displacement = np.linalg.norm(suffixes[candidates, -1] - prefixes[candidates, -1], axis=1)
    confidence = weights[candidates].max(axis=1)
    display_score = future_displacement * confidence / (1.0 + errors)
    return int(candidates[np.argmax(display_score)])


def conditional_model_paths(
    model,
    checkpoint: dict,
    prefix: np.ndarray,
    condition: np.ndarray,
    target: np.ndarray,
    change_points: np.ndarray,
    future_steps: int,
    dt: float,
    n_candidates: int,
    n_paths: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Draw an equal number of conditional posterior paths for every model."""
    track_t = torch.from_numpy(normalize_track(prefix, checkpoint).T[None]).float().to(device)
    condition_t = torch.from_numpy(condition[None].astype(np.float32)).to(device)
    output = {}
    for model_name in MODEL_NAMES:
        candidates = sample_model_parameters(model_name, n_candidates, rng)
        logits = score_params(model, track_t, condition_t, model_name, candidates, device)
        posterior_weights = softmax(logits)
        theta = candidates[rng.choice(len(candidates), size=n_paths, p=posterior_weights)]
        paths, _ = simulate_future_batch(
            model_name=model_name,
            theta=theta,
            start=prefix[-1],
            target=target,
            observed_steps=len(prefix),
            observed_change_points=change_points,
            future_steps=future_steps,
            dt=dt,
            rng=rng,
        )
        output[model_name] = paths
    return output


def probability_text(weights: np.ndarray, highest_model: str) -> str:
    lines = ["model posterior"]
    for model_name, weight in zip(MODEL_NAMES, weights):
        marker = "  highest" if model_name == highest_model else ""
        lines.append(f"{model_name:>20}  {weight:5.1%}{marker}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate four model forecasts for one controlled trajectory.")
    parser.add_argument(
        "--samples",
        default="outputs/method_validation/forecast_evaluation/posterior_predictive_samples.npz",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/method_validation/forecast_evaluation/case_metrics.csv",
    )
    parser.add_argument("--forecast-data", default="data/method_validation/forecast_test.npz")
    parser.add_argument("--checkpoint", default="checkpoints/method_validation/ratio_estimator_best.pt")
    parser.add_argument("--case-row", type=int, default=-1, help="-1 selects a dynamic, confident, low-error case.")
    parser.add_argument("--n-candidates", type=int, default=2048)
    parser.add_argument("--paths-per-model", type=int, default=64)
    parser.add_argument("--shown-paths-per-model", type=int, default=14)
    parser.add_argument("--prefix-stride", type=int, default=2)
    parser.add_argument("--prefix-hold-frames", type=int, default=8)
    parser.add_argument("--final-hold-frames", type=int, default=18)
    parser.add_argument("--fps", type=int, default=4, help="Lower values increase each frame's duration.")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--out",
        default="outputs/method_validation/forecast_evaluation/validation_walkthrough.gif",
    )
    parser.add_argument(
        "--final-frame",
        default="outputs/method_validation/forecast_evaluation/validation_walkthrough_final.png",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    samples = np.load(args.samples, allow_pickle=True)
    forecast_data = np.load(args.forecast_data, allow_pickle=True)
    metric_rows = load_rows(Path(args.metrics))
    prefixes = samples["prefix_tracks"].astype(np.float32)
    suffixes = samples["suffix_tracks"].astype(np.float32)
    model_weights = samples["model_weights"].astype(np.float32)
    true_model_ids = samples["true_model_ids"].astype(np.int64)
    source_indices = samples["source_indices"].astype(np.int64)
    validate_checkpoint_contract(
        checkpoint,
        steps=prefixes.shape[1],
        dt=float(forecast_data["dt"]),
    )

    case = args.case_row
    if case < 0:
        case = choose_case(prefixes, suffixes, model_weights, true_model_ids, metric_rows)
    if case >= len(prefixes):
        raise IndexError(f"--case-row {case} is outside 0..{len(prefixes) - 1}.")
    source_index = int(source_indices[case])
    prefix = prefixes[case]
    truth = suffixes[case]
    weights = model_weights[case]
    highest_model = MODEL_NAMES[int(np.argmax(weights))]
    true_model = MODEL_NAMES[int(true_model_ids[case])]
    model_paths = conditional_model_paths(
        model=model,
        checkpoint=checkpoint,
        prefix=prefix,
        condition=forecast_data["conditions"][source_index],
        target=forecast_data["target"][source_index],
        change_points=forecast_data["change_points"][source_index],
        future_steps=len(truth),
        dt=float(forecast_data["dt"]),
        n_candidates=args.n_candidates,
        n_paths=args.paths_per_model,
        rng=rng,
        device=device,
    )
    shown_indices = {
        model_name: rng.choice(
            args.paths_per_model,
            size=min(args.shown_paths_per_model, args.paths_per_model),
            replace=False,
        )
        for model_name in MODEL_NAMES
    }

    prefix_frames = int(np.ceil(len(prefix) / args.prefix_stride))
    total_frames = prefix_frames + args.prefix_hold_frames + len(truth) + args.final_hold_frames
    fig, ax = plt.subplots(figsize=(12.5, 7.8))
    fig.patch.set_facecolor("#101820")
    legend = [Line2D([0], [0], color="#2dd4bf", linewidth=4, label="observed prefix")]
    for model_name in MODEL_NAMES:
        label = f"{model_name} mean"
        if model_name == highest_model:
            label += " (highest probability)"
        legend.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[model_name],
                linewidth=4 if model_name == highest_model else 2,
                label=label,
            )
        )
    legend.append(Line2D([0], [0], color="#ffe066", linewidth=4, linestyle="--", label="true future"))
    fig.legend(
        handles=legend,
        loc="lower center",
        ncols=3,
        frameon=False,
        labelcolor="white",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )

    def draw(frame: int) -> None:
        ax.clear()
        pitch_background(ax)
        if frame < prefix_frames:
            prefix_end = min(len(prefix), (frame + 1) * args.prefix_stride)
            stage = f"Observed trajectory: {prefix_end}/{len(prefix)} frames"
            future_end = 0
        elif frame < prefix_frames + args.prefix_hold_frames:
            prefix_end = len(prefix)
            stage = "Observed prefix complete - infer model and parameters"
            future_end = 0
        else:
            prefix_end = len(prefix)
            future_end = min(len(truth), frame - prefix_frames - args.prefix_hold_frames + 1)
            stage = f"Synchronized forecast: {future_end}/{len(truth)} future frames"
        fig.suptitle(stage, color="white", fontsize=16, fontweight="bold", y=0.97)
        ax.set_title(
            f"one trajectory | true simulator: {true_model} | selected: {highest_model}",
            color="white",
            fontsize=11,
            pad=7,
        )
        observed = prefix[:prefix_end]
        ax.plot(observed[:, 0], observed[:, 1], color="#2dd4bf", linewidth=4, zorder=8)
        ax.plot(observed[-1, 0], observed[-1, 1], "o", color="white", markersize=6, zorder=9)

        if future_end > 0:
            for model_name in MODEL_NAMES:
                color = MODEL_COLORS[model_name]
                is_highest = model_name == highest_model
                for path_index in shown_indices[model_name]:
                    drawn = np.vstack([prefix[-1], model_paths[model_name][path_index, :future_end]])
                    ax.plot(
                        drawn[:, 0],
                        drawn[:, 1],
                        color=color,
                        alpha=0.13 if is_highest else 0.045,
                        linewidth=1.0,
                    )
                mean = model_paths[model_name][:, :future_end].mean(axis=0)
                mean_drawn = np.vstack([prefix[-1], mean])
                ax.plot(
                    mean_drawn[:, 0],
                    mean_drawn[:, 1],
                    color=color,
                    linewidth=4.5 if is_highest else 2.0,
                    alpha=1.0 if is_highest else 0.78,
                    zorder=6 if is_highest else 5,
                )
                ax.plot(
                    mean[-1, 0],
                    mean[-1, 1],
                    "o",
                    color=color,
                    markersize=8 if is_highest else 5,
                    zorder=7,
                )
            true_drawn = np.vstack([prefix[-1], truth[:future_end]])
            ax.plot(
                true_drawn[:, 0],
                true_drawn[:, 1],
                color="#ffe066",
                linewidth=4.5,
                linestyle="--",
                zorder=10,
            )
            ax.plot(truth[future_end - 1, 0], truth[future_end - 1, 1], "o", color="#ffe066", markersize=9, zorder=11)
            ax.text(
                0.98,
                0.96,
                probability_text(weights, highest_model),
                transform=ax.transAxes,
                va="top",
                ha="right",
                color="white",
                family="monospace",
                fontsize=9,
                bbox={"facecolor": "#101820", "alpha": 0.82, "edgecolor": "none", "pad": 5},
            )
            ax.text(
                0.98,
                0.04,
                f"case ADE: {float(metric_rows[case]['sbi_ade']):.2f} m",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="white",
                fontsize=10,
                bbox={"facecolor": "#101820", "alpha": 0.82, "edgecolor": "none", "pad": 4},
            )

    animation = FuncAnimation(fig, draw, frames=total_frames, interval=1000 / args.fps, repeat=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out_path, writer=PillowWriter(fps=args.fps), dpi=95)
    draw(total_frames - 1)
    final_path = Path(args.final_frame)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(final_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Case row: {case}; source index: {source_index}")
    print(f"True model: {true_model}; highest probability: {highest_model}")
    print(f"Saved {out_path}")
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()
