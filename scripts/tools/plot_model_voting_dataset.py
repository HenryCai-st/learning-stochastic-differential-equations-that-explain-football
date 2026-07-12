"""
Visualize diversity in the mixed model-voting synthetic dataset.

Inputs:
    - data/model_voting_dataset/dataset.npz

Outputs:
    - track overlays grouped by candidate model family
    - parameter-prior histograms
    - trajectory-statistic plots comparing model families

Expected use:
    Run this after generate_model_voting_data.py to verify that the candidate
    simulators produce visibly different motion patterns before training.
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

from src.simulators.ou import PITCH_LENGTH, PITCH_WIDTH
from src.simulators.model_voting import MODEL_NAMES, MODEL_PARAMETER_NAMES
from src.football.visualization import pitch_background


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load a model-voting dataset `.npz` into a normal dictionary."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def choose_indices(indices: np.ndarray, max_tracks: int, rng: np.random.Generator) -> np.ndarray:
    """Choose a subset of row indices without replacement for plotting."""
    if len(indices) <= max_tracks:
        return indices
    return rng.choice(indices, size=max_tracks, replace=False)


def plot_tracks_by_model(dataset: dict[str, np.ndarray], max_tracks: int, seed: int, out_path: Path) -> None:
    """Plot sampled synthetic tracks in a separate panel for each model family."""
    tracks = dataset["tracks"].astype(np.float32)
    model_id = dataset["model_id"].astype(np.int64)
    change_points = dataset.get("change_points")
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a1a")
    colors = {
        "brownian": "#e45756",
        "constant_velocity": "#4c78a8",
        "ou_target": "#54a24b",
        "piecewise_velocity": "#f58518",
    }

    for ax, model_name in zip(axes.ravel(), MODEL_NAMES):
        pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)
        mid = MODEL_NAMES.index(model_name)
        selected = choose_indices(np.where(model_id == mid)[0], max_tracks=max_tracks, rng=rng)
        color = colors[model_name]

        for idx in selected:
            track = tracks[idx]
            ax.plot(track[:, 0], track[:, 1], color=color, alpha=0.28, linewidth=0.8)
            ax.plot(track[0, 0], track[0, 1], "o", color="white", markersize=2.5, alpha=0.7)
            ax.plot(track[-1, 0], track[-1, 1], "s", color=color, markersize=3.0, alpha=0.85)

            if change_points is not None and model_name == "piecewise_velocity":
                cps = change_points[idx]
                for cp in cps:
                    cp = int(cp)
                    if 0 < cp < len(track):
                        ax.plot(track[cp, 0], track[cp, 1], "x", color="black", markersize=4, alpha=0.45)

        ax.set_title(f"{model_name} (n={len(selected)})", color="white", fontsize=12, pad=8)

    fig.suptitle("Model-voting synthetic football tracks", color="white", fontsize=16, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_model_parameter_histograms(dataset: dict[str, np.ndarray], out_path: Path) -> None:
    """Plot active theta dimensions with model-specific parameter names."""
    parameters = dataset["parameters"].astype(np.float32)
    parameter_mask = dataset["parameter_mask"].astype(np.float32)
    model_id = dataset["model_id"].astype(np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    for ax, model_name in zip(axes, MODEL_NAMES):
        mid = MODEL_NAMES.index(model_name)
        rows = model_id == mid
        params = parameters[rows]
        mask = parameter_mask[rows]
        active_dims = np.where(mask[0] > 0.0)[0] if len(mask) else []
        parameter_names = MODEL_PARAMETER_NAMES[model_name]

        for dim in active_dims:
            ax.hist(params[:, dim], bins=30, alpha=0.45, label=f"theta[{dim}] = {parameter_names[dim]}")

        ax.set_title(model_name)
        ax.set_xlabel("parameter value")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Model-voting parameter priors", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_displacement_summary(dataset: dict[str, np.ndarray], out_path: Path) -> None:
    """Compare displacement, speed, and path length across model families."""
    tracks = dataset["tracks"].astype(np.float32)
    model_id = dataset["model_id"].astype(np.int64)
    dt = float(dataset["dt"]) if "dt" in dataset else 0.04

    step = np.linalg.norm(np.diff(tracks, axis=1), axis=2)
    mean_speed = step.mean(axis=1) / dt
    displacement = np.linalg.norm(tracks[:, -1] - tracks[:, 0], axis=1)
    path_length = step.sum(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#e45756", "#4c78a8", "#54a24b", "#f58518"]

    for mid, model_name in enumerate(MODEL_NAMES):
        rows = model_id == mid
        axes[0].scatter(displacement[rows], mean_speed[rows], s=8, alpha=0.35, label=model_name, color=colors[mid])
        axes[1].scatter(displacement[rows], path_length[rows], s=8, alpha=0.35, color=colors[mid])
        axes[2].hist(displacement[rows], bins=30, alpha=0.35, label=model_name, color=colors[mid])

    axes[0].set_xlabel("start-end displacement (m)")
    axes[0].set_ylabel("mean speed proxy (m/s)")
    axes[0].set_title("Speed vs displacement")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("start-end displacement (m)")
    axes[1].set_ylabel("path length (m)")
    axes[1].set_title("Path length vs displacement")
    axes[1].grid(True, alpha=0.25)

    axes[2].set_xlabel("start-end displacement (m)")
    axes[2].set_ylabel("count")
    axes[2].set_title("Displacement distribution")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.suptitle("Model-voting trajectory statistics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load the mixed dataset and write all diversity diagnostic plots."""
    parser = argparse.ArgumentParser(description="Visualize the mixed-model football SBI dataset.")
    parser.add_argument("--dataset", default="data/model_voting_dataset/dataset.npz")
    parser.add_argument("--out-dir", default="outputs/model_voting_dataset_viz")
    parser.add_argument("--max-tracks-per-model", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_tracks_by_model(
        dataset=dataset,
        max_tracks=args.max_tracks_per_model,
        seed=args.seed,
        out_path=out_dir / "tracks_by_model.png",
    )
    plot_model_parameter_histograms(dataset, out_dir / "parameter_priors.png")
    plot_displacement_summary(dataset, out_dir / "trajectory_statistics.png")

    print(f"Saved model-voting dataset visualizations to {out_dir}")


if __name__ == "__main__":
    main()
