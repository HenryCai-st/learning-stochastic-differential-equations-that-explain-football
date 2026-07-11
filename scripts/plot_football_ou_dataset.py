"""
Visualize diversity in the generated football OU baseline dataset.

Inputs:
    - data/football_ou_dataset/dataset.npz

Outputs:
    - track overlays colored by OU parameters
    - parameter histograms
    - simple trajectory-statistic scatter plots

Expected use:
    Use this before OU baseline training to confirm that the synthetic data has
    enough parameter and path diversity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sde.football_ou import PITCH_LENGTH, PITCH_WIDTH
from src.utils.football_viz import pitch_background


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load an OU dataset `.npz` into a plain dictionary."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def choose_indices(n_items: int, max_tracks: int, seed: int) -> np.ndarray:
    """Choose a reproducible subset of tracks for readable plotting."""
    rng = np.random.default_rng(seed)
    if n_items <= max_tracks:
        return np.arange(n_items)
    return rng.choice(n_items, size=max_tracks, replace=False)


def plot_tracks_colored(
    tracks: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
    title: str,
    colorbar_label: str,
    out_path: Path,
) -> None:
    """Overlay sampled tracks on a pitch and color them by one scalar value."""
    fig, ax = plt.subplots(figsize=(11, 7))
    pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)
    fig.patch.set_facecolor("#1a1a1a")

    selected_values = values[indices]
    vmin, vmax = float(selected_values.min()), float(selected_values.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    for idx in indices:
        track = tracks[idx]
        color = cmap(norm(values[idx]))
        ax.plot(track[:, 0], track[:, 1], color=color, alpha=0.35, linewidth=0.8)
        ax.plot(track[0, 0], track[0, 1], "o", color="white", markersize=2, alpha=0.45)
        ax.plot(track[-1, 0], track[-1, 1], "s", color=color, markersize=3, alpha=0.75)

    ax.set_title(title, color="white", fontsize=13, pad=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label(colorbar_label, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_parameter_histograms(parameters: np.ndarray, out_path: Path) -> None:
    """Plot histograms of the OU baseline parameters."""
    k = parameters[:, 0]
    noise = parameters[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(k, bins=35, color="#4c78a8", edgecolor="white", linewidth=0.5)
    axes[0].set_title("OU attraction rate k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(noise, bins=35, color="#f58518", edgecolor="white", linewidth=0.5)
    axes[1].set_title("OU noise scale")
    axes[1].set_xlabel("noise_scale")
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Generated football OU parameter diversity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_speed_displacement_summary(
    tracks: np.ndarray,
    parameters: np.ndarray,
    dt: float,
    out_path: Path,
) -> None:
    """Plot simple speed/displacement summaries for generated OU tracks."""
    step = np.linalg.norm(np.diff(tracks, axis=1), axis=2)
    mean_speed = step.mean(axis=1) / dt
    displacement = np.linalg.norm(tracks[:, -1] - tracks[:, 0], axis=1)
    path_length = step.sum(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(parameters[:, 0], displacement, s=8, alpha=0.35)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("start-end displacement (m)")
    axes[0].set_title("Displacement vs k")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(parameters[:, 1], mean_speed, s=8, alpha=0.35, color="#f58518")
    axes[1].set_xlabel("noise_scale")
    axes[1].set_ylabel("mean speed proxy (m/s)")
    axes[1].set_title("Speed proxy vs noise")
    axes[1].grid(True, alpha=0.25)

    axes[2].scatter(displacement, path_length, s=8, alpha=0.35, color="#54a24b")
    axes[2].set_xlabel("start-end displacement (m)")
    axes[2].set_ylabel("path length (m)")
    axes[2].set_title("Path length vs displacement")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Generated football OU trajectory statistics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load the OU dataset and write all diversity diagnostic figures."""
    parser = argparse.ArgumentParser(description="Plot diversity of generated football OU tracks.")
    parser.add_argument("--dataset", default="data/football_ou_dataset/dataset.npz")
    parser.add_argument("--out-dir", default="outputs/football_ou_dataset_viz")
    parser.add_argument("--max-tracks", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    tracks = dataset["tracks"].astype(np.float32)
    parameters = dataset["parameters"].astype(np.float32)
    dt = float(dataset["dt"]) if "dt" in dataset else 0.04
    indices = choose_indices(len(tracks), args.max_tracks, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_tracks_colored(
        tracks=tracks,
        values=parameters[:, 0],
        indices=indices,
        title=f"Generated football OU tracks colored by k (n={len(indices)})",
        colorbar_label="k",
        out_path=out_dir / "tracks_colored_by_k.png",
    )
    plot_tracks_colored(
        tracks=tracks,
        values=parameters[:, 1],
        indices=indices,
        title=f"Generated football OU tracks colored by noise_scale (n={len(indices)})",
        colorbar_label="noise_scale",
        out_path=out_dir / "tracks_colored_by_noise.png",
    )
    plot_parameter_histograms(parameters, out_dir / "parameter_histograms.png")
    plot_speed_displacement_summary(tracks, parameters, dt, out_dir / "trajectory_statistics.png")

    print(f"Saved football OU dataset visualizations to {out_dir}")


if __name__ == "__main__":
    main()
