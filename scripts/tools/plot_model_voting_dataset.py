"""
Visualize diversity in the mixed model-voting synthetic dataset.

Inputs:
    - data/model_voting_dataset/dataset.npz

Outputs:
    - track overlays grouped by candidate model family
    - discrete model-family prior
    - model-specific parameter-prior histograms with theoretical densities
    - simulation-condition priors for starts, targets, and change points
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

from src.sde.football_ou import PITCH_LENGTH, PITCH_WIDTH
from src.sde.model_voting import MAX_PARAM_DIM, MODEL_NAMES, MODEL_PARAMETER_NAMES, MODEL_SPECS
from src.utils.football_viz import pitch_background


MODEL_COLORS = {
    "brownian": "#e45756",
    "constant_velocity": "#4c78a8",
    "ou_target": "#54a24b",
    "piecewise_velocity": "#f58518",
}


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
    for ax, model_name in zip(axes.ravel(), MODEL_NAMES):
        pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)
        mid = MODEL_NAMES.index(model_name)
        selected = choose_indices(np.where(model_id == mid)[0], max_tracks=max_tracks, rng=rng)
        color = MODEL_COLORS[model_name]

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


def plot_model_prior(dataset: dict[str, np.ndarray], out_path: Path) -> None:
    """Plot the empirical discrete prior probability of each model family."""
    model_id = dataset["model_id"].astype(np.int64)
    counts = np.bincount(model_id, minlength=len(MODEL_NAMES))
    probabilities = counts / max(counts.sum(), 1)
    expected = 1.0 / len(MODEL_NAMES)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(
        MODEL_NAMES,
        probabilities,
        color=[MODEL_COLORS[name] for name in MODEL_NAMES],
        edgecolor="white",
    )
    ax.axhline(expected, color="#222222", linestyle="--", linewidth=1.4, label=f"balanced prior = {expected:.2f}")
    for bar, probability, count in zip(bars, probabilities, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"p={probability:.3f}\nn={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0.0, max(0.35, float(probabilities.max()) * 1.25))
    ax.set_xlabel("candidate SDE model family")
    ax.set_ylabel("empirical prior probability")
    ax.set_title("Discrete model prior in synthetic SBI training data", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def theoretical_prior_density(values: np.ndarray, low: float, high: float, log_scale: bool) -> np.ndarray:
    """Evaluate the configured uniform or log-uniform prior density."""
    if log_scale:
        return 1.0 / (values * np.log(high / low))
    return np.full_like(values, 1.0 / (high - low), dtype=np.float64)


def plot_model_parameter_histograms(dataset: dict[str, np.ndarray], out_path: Path) -> None:
    """Plot every active theta prior on its own scale with theoretical density."""
    parameters = dataset["parameters"].astype(np.float32)
    parameter_mask = dataset["parameter_mask"].astype(np.float32)
    model_id = dataset["model_id"].astype(np.int64)

    fig, axes = plt.subplots(len(MODEL_NAMES), MAX_PARAM_DIM, figsize=(22, 11.5), squeeze=False)

    for row, model_name in enumerate(MODEL_NAMES):
        mid = MODEL_NAMES.index(model_name)
        rows = model_id == mid
        params = parameters[rows]
        mask = parameter_mask[rows]
        active_dims = np.where(mask[0] > 0.0)[0] if len(mask) else []
        parameter_names = MODEL_PARAMETER_NAMES[model_name]
        spec = MODEL_SPECS[model_name]

        for dim in active_dims:
            ax = axes[row, dim]
            low = float(spec.low[dim])
            high = float(spec.high[dim])
            log_scale = bool(spec.log_scale[dim])
            samples = params[:, dim]
            x = np.geomspace(low, high, 300) if log_scale else np.linspace(low, high, 300)
            ax.hist(
                samples,
                bins=32,
                density=True,
                alpha=0.62,
                color=MODEL_COLORS[model_name],
                edgecolor="white",
                linewidth=0.4,
                label="sampled prior",
            )
            ax.plot(x, theoretical_prior_density(x, low, high, log_scale), color="#111111", linewidth=1.5, label="configured density")
            ax.axvline(low, color="#666666", linestyle=":", linewidth=1.0)
            ax.axvline(high, color="#666666", linestyle=":", linewidth=1.0)
            if log_scale:
                ax.set_xscale("log")
            ax.set_title(f"theta[{dim}] = {parameter_names[dim]}", fontsize=9.5)
            ax.set_xlabel(f"[{low:g}, {high:g}]" + (" log-uniform" if log_scale else " uniform"), fontsize=8)
            ax.set_ylabel("density" if dim == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7.5)
            ax.grid(True, alpha=0.2)
            if dim == 0:
                ax.text(-0.36, 0.5, model_name, transform=ax.transAxes, rotation=90, ha="center", va="center", fontweight="bold")

        for dim in range(MAX_PARAM_DIM):
            if dim not in active_dims:
                axes[row, dim].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.suptitle("Continuous parameter priors sampled for synthetic training", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_condition_priors(dataset: dict[str, np.ndarray], out_path: Path) -> None:
    """Plot empirical simulation-condition priors used to generate the tracks."""
    y0 = dataset["y0"].astype(np.float32)
    target = dataset["target"].astype(np.float32)
    change_points = dataset["change_points"].astype(np.float32)
    dt = float(dataset["dt"]) if "dt" in dataset else 0.04

    panels = [
        (y0[:, 0], "start x", "metres"),
        (y0[:, 1], "start y", "metres"),
        (target[:, 0], "condition target x", "metres"),
        (target[:, 1], "condition target y", "metres"),
        (change_points[:, 0] * dt, "change point 1", "seconds from start"),
        (change_points[:, 1] * dt, "change point 2", "seconds from start"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (values, title, xlabel) in zip(axes.ravel(), panels):
        unique_count = len(np.unique(values))
        if unique_count == 1:
            value = float(values[0])
            span = 0.5 if xlabel == "metres" else 0.1
            ax.axvline(value, color="#2a9d8f", linewidth=7, alpha=0.75)
            ax.set_xlim(value - span, value + span)
            ax.set_yticks([])
            ax.set_ylabel("point mass")
        else:
            ax.hist(values, bins=32, density=True, color="#72b7b2", edgecolor="white", linewidth=0.5)
            ax.set_ylabel("density")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.text(
            0.98,
            0.92,
            f"unique values = {unique_count}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.grid(True, alpha=0.22)
    all_unique = [len(np.unique(values)) for values, _, _ in panels]
    limitation = " (single bootstrapped condition: not diverse)" if max(all_unique) == 1 else ""
    fig.suptitle(
        f"Simulation-condition priors in synthetic training data{limitation}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_displacement_summary(dataset: dict[str, np.ndarray], out_path: Path) -> None:
    """Compare prior-predictive trajectory statistics across model families."""
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

    fig.suptitle("Prior-predictive trajectory diversity by model", fontsize=13, fontweight="bold")
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
    plot_model_prior(dataset, out_dir / "model_prior.png")
    plot_model_parameter_histograms(dataset, out_dir / "parameter_priors.png")
    plot_condition_priors(dataset, out_dir / "condition_priors.png")
    plot_displacement_summary(dataset, out_dir / "prior_predictive_trajectory_statistics.png")

    y0_unique = len(np.unique(dataset["y0"], axis=0))
    target_unique = len(np.unique(dataset["target"], axis=0))
    change_point_unique = len(np.unique(dataset["change_points"], axis=0))
    if min(y0_unique, target_unique, change_point_unique) == 1:
        print(
            "WARNING: condition priors are not diverse: "
            f"unique starts={y0_unique}, targets={target_unique}, change-point sets={change_point_unique}. "
            "Extract multiple real windows or use the synthetic condition pool before final training."
        )
    print(f"Saved model-voting dataset visualizations to {out_dir}")


if __name__ == "__main__":
    main()
