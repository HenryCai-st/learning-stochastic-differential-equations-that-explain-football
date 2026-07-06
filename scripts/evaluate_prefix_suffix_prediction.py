from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.segmentation import fixed_even_change_points
from src.sde.football_ou import PITCH_LENGTH, PITCH_WIDTH
from src.sde.model_voting import MODEL_NAMES, simulate_model_batch
from src.utils.football_viz import pitch_background


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def sample_suffix_paths(posterior: dict[str, np.ndarray], n_paths: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prefix_end = posterior["prefix_end"].astype(np.float32)
    target = posterior["target_for_inference"].astype(np.float32)
    suffix_steps = int(posterior["suffix_steps"])
    dt = float(posterior["dt"])
    weights = posterior["model_vote_weights"].astype(np.float64)
    weights = weights / weights.sum()
    sampled_model_ids = rng.choice(len(MODEL_NAMES), size=n_paths, p=weights)
    all_paths = []
    suffix_change_points = fixed_even_change_points(suffix_steps + 1, max_segments=3, min_segment_len=12)

    for mid, model_name in enumerate(MODEL_NAMES):
        rows = np.where(sampled_model_ids == mid)[0]
        if len(rows) == 0:
            continue
        samples = posterior[f"{model_name}_samples"].astype(np.float32)
        sample_idx = rng.choice(len(samples), size=len(rows), replace=True)
        theta = samples[sample_idx]
        paths = simulate_model_batch(
            model_name=model_name,
            params=theta,
            y0=np.repeat(prefix_end[None], len(rows), axis=0),
            target=np.repeat(target[None], len(rows), axis=0),
            change_points=np.repeat(suffix_change_points[None], len(rows), axis=0),
            steps=suffix_steps + 1,
            dt=dt,
            rng=rng,
        )
        all_paths.append(paths[:, 1:])

    return np.concatenate(all_paths, axis=0).astype(np.float32), sampled_model_ids.astype(np.int64)


def coverage_rate_90(paths: np.ndarray, truth: np.ndarray) -> float:
    centers = np.median(paths, axis=0)
    distances = np.linalg.norm(paths - centers[None], axis=2)
    radii = np.quantile(distances, 0.90, axis=0)
    true_distances = np.linalg.norm(truth - centers, axis=1)
    return float(np.mean(true_distances <= radii))


def plot_predictive_paths(prefix: np.ndarray, suffix: np.ndarray, paths: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)
    fig.patch.set_facecolor("#1a1a1a")
    prefix_end = prefix[-1]
    for path in paths:
        drawn = np.vstack([prefix_end[None], path])
        ax.plot(drawn[:, 0], drawn[:, 1], color="#2ca02c", alpha=0.16, linewidth=0.8)
    ax.plot(prefix[:, 0], prefix[:, 1], color="#1f77b4", linewidth=3, label="observed prefix")
    true_future = np.vstack([prefix_end[None], suffix])
    ax.plot(true_future[:, 0], true_future[:, 1], color="#dc2626", linewidth=3, label="held-out suffix")
    ax.plot(prefix[0, 0], prefix[0, 1], "o", color="#1f77b4", markersize=8, markeredgecolor="white", label="start")
    ax.plot(prefix_end[0], prefix_end[1], "s", color="white", markersize=7, markeredgecolor="#1f77b4", label="prefix end")
    ax.plot(suffix[-1, 0], suffix[-1, 1], "s", color="#dc2626", markersize=8, markeredgecolor="white", label="true suffix end")
    ax.set_title("Prefix-only posterior predictive suffix paths", color="white", fontsize=14, pad=10)
    ax.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_endpoint_density(prefix: np.ndarray, suffix: np.ndarray, paths: np.ndarray, out_path: Path) -> None:
    endpoints = paths[:, -1]
    fig, ax = plt.subplots(figsize=(11, 7))
    pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)
    fig.patch.set_facecolor("#1a1a1a")
    hb = ax.hexbin(endpoints[:, 0], endpoints[:, 1], gridsize=38, cmap="magma", mincnt=1, alpha=0.85)
    ax.plot(prefix[:, 0], prefix[:, 1], color="#00bcd4", linewidth=2.5)
    ax.plot(suffix[:, 0], suffix[:, 1], color="#dc2626", linewidth=2.5)
    ax.plot(suffix[-1, 0], suffix[-1, 1], "s", color="white", markersize=8, markeredgecolor="#dc2626")
    ax.set_title("Prefix-only predictive endpoint density", color="white", fontsize=14, pad=10)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("sample count", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_model_votes(weights: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(MODEL_NAMES, weights, color=["#e45756", "#4c78a8", "#54a24b", "#f58518"])
    ax.set_ylabel("posterior vote weight")
    ax.set_ylim(0.0, max(1.0, float(weights.max()) * 1.15))
    ax.set_title("Prefix-only model posterior vote weights")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out suffix prediction from prefix-only posterior.")
    parser.add_argument("--posterior", default="outputs/prefix_suffix_posterior/posterior_chains.npz")
    parser.add_argument("--n-paths", type=int, default=300)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--out-dir", default="outputs/prefix_suffix_prediction")
    args = parser.parse_args()

    posterior = load_npz(args.posterior)
    prefix = posterior["prefix"].astype(np.float32)
    suffix = posterior["suffix"].astype(np.float32)
    paths, sampled_model_ids = sample_suffix_paths(posterior, args.n_paths, args.seed)
    weights = posterior["model_vote_weights"].astype(np.float64)
    weights = weights / weights.sum()
    winning_model = MODEL_NAMES[int(np.argmax(weights))]

    endpoint_error = np.linalg.norm(paths[:, -1] - suffix[-1][None], axis=1)
    path_rmse = np.sqrt(((paths - suffix[None]) ** 2).sum(axis=2).mean(axis=1))
    coverage = coverage_rate_90(paths, suffix)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_predictive_paths(prefix, suffix, paths, out_dir / "posterior_predictive_paths.png")
    plot_endpoint_density(prefix, suffix, paths, out_dir / "endpoint_density.png")
    plot_model_votes(weights, out_dir / "model_vote_weights.png")

    summary = {
        "posterior": args.posterior,
        "n_paths": args.n_paths,
        "winning_model": winning_model,
        "model_vote_weights": {name: float(weight) for name, weight in zip(MODEL_NAMES, weights)},
        "sampled_model_counts": {
            name: int(np.sum(sampled_model_ids == i))
            for i, name in enumerate(MODEL_NAMES)
        },
        "suffix_endpoint_error_m": {
            "median": float(np.median(endpoint_error)),
            "p10": float(np.quantile(endpoint_error, 0.10)),
            "p90": float(np.quantile(endpoint_error, 0.90)),
        },
        "path_rmse_m": {
            "median": float(np.median(path_rmse)),
            "p10": float(np.quantile(path_rmse, 0.10)),
            "p90": float(np.quantile(path_rmse, 0.90)),
        },
        "coverage_rate": coverage,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "posterior_predictive_samples.npz",
        prefix=prefix,
        suffix=suffix,
        paths=paths,
        sampled_model_ids=sampled_model_ids,
        endpoint_error=endpoint_error,
        path_rmse=path_rmse,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved prefix/suffix prediction outputs to {out_dir}")


if __name__ == "__main__":
    main()
