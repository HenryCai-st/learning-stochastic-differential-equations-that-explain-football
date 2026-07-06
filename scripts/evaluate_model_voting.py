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

from src.sde.football_ou import PITCH_LENGTH, PITCH_WIDTH
from src.sde.model_voting import MODEL_NAMES, MODEL_SPECS, simulate_model_batch
from src.utils.football_viz import pitch_background


PARAMETER_LABELS = {
    "brownian": ["noise_scale"],
    "constant_velocity": ["vx", "vy", "noise_scale"],
    "ou_target": ["k", "noise_scale"],
    "piecewise_velocity": ["vx1", "vy1", "vx2", "vy2", "vx3", "vy3", "noise_scale"],
}


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def sample_posterior_paths(
    posterior: dict[str, np.ndarray],
    n_paths: int,
    steps: int,
    dt: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    y0 = posterior["y0"].astype(np.float32)
    target = posterior["target"].astype(np.float32)
    change_points = posterior["change_points"].astype(np.int64)
    weights = posterior["model_vote_weights"].astype(np.float64)
    weights = weights / weights.sum()

    sampled_model_ids = rng.choice(len(MODEL_NAMES), size=n_paths, p=weights)
    all_paths = []
    all_model_names = []

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
            y0=np.repeat(y0[None], len(rows), axis=0),
            target=np.repeat(target[None], len(rows), axis=0),
            change_points=np.repeat(change_points[None], len(rows), axis=0),
            steps=steps,
            dt=dt,
            rng=rng,
        )
        all_paths.append(paths)
        all_model_names.extend([model_name] * len(rows))

    return (
        np.concatenate(all_paths, axis=0).astype(np.float32),
        sampled_model_ids.astype(np.int64),
        all_model_names,
    )


def plot_posterior_predictive_paths(observed: np.ndarray, paths: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)
    fig.patch.set_facecolor("#1a1a1a")

    for path in paths:
        ax.plot(path[:, 0], path[:, 1], color="#2ca02c", alpha=0.18, linewidth=0.8)
    ax.plot(observed[:, 0], observed[:, 1], color="#1f77b4", linewidth=3, label="observed")
    ax.plot(observed[0, 0], observed[0, 1], "o", color="#1f77b4", markersize=8, markeredgecolor="white", label="start")
    ax.plot(observed[-1, 0], observed[-1, 1], "s", color="#dc2626", markersize=8, markeredgecolor="white", label="end")
    ax.set_title("Model-voting posterior predictive paths", color="white", fontsize=14, pad=10)
    ax.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_endpoint_density(paths: np.ndarray, observed: np.ndarray, out_path: Path) -> None:
    endpoints = paths[:, -1]
    fig, ax = plt.subplots(figsize=(11, 7))
    pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)
    fig.patch.set_facecolor("#1a1a1a")
    hb = ax.hexbin(endpoints[:, 0], endpoints[:, 1], gridsize=38, cmap="magma", mincnt=1, alpha=0.85)
    ax.plot(observed[:, 0], observed[:, 1], color="#00bcd4", linewidth=2.5)
    ax.plot(observed[-1, 0], observed[-1, 1], "s", color="white", markersize=8, markeredgecolor="#00bcd4")
    ax.set_title("Posterior predictive endpoint density", color="white", fontsize=14, pad=10)
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
    ax.set_title("Model posterior vote weights")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_winning_parameter_histograms(posterior: dict[str, np.ndarray], winning_model: str, out_path: Path) -> None:
    samples = posterior[f"{winning_model}_samples"].astype(np.float32)
    labels = PARAMETER_LABELS[winning_model]
    n_params = MODEL_SPECS[winning_model].param_dim
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for dim in range(n_params):
        ax = axes[dim]
        ax.hist(samples[:, dim], bins=35, color="#4c78a8", edgecolor="white", linewidth=0.5)
        ax.set_title(labels[dim])
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    for ax in axes[n_params:]:
        ax.axis("off")

    fig.suptitle(f"Winning model parameter posterior: {winning_model}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model-voting posterior predictive distribution.")
    parser.add_argument("--posterior", default="outputs/model_voting_posterior/posterior_chains.npz")
    parser.add_argument("--n-paths", type=int, default=300)
    parser.add_argument("--steps", type=int, default=0, help="0 means use observed window length.")
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--out-dir", default="outputs/model_voting_evaluation")
    args = parser.parse_args()

    posterior = load_npz(args.posterior)
    observed = posterior["observed"].astype(np.float32)
    steps = len(observed) if args.steps <= 0 else args.steps
    paths, sampled_model_ids, _ = sample_posterior_paths(
        posterior=posterior,
        n_paths=args.n_paths,
        steps=steps,
        dt=args.dt,
        seed=args.seed,
    )

    weights = posterior["model_vote_weights"].astype(np.float64)
    weights = weights / weights.sum()
    winning_model = MODEL_NAMES[int(np.argmax(weights))]
    endpoints = paths[:, -1]
    observed_end = observed[-1]
    endpoint_error = np.linalg.norm(endpoints - observed_end[None], axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_posterior_predictive_paths(observed, paths, out_dir / "posterior_predictive_paths.png")
    plot_endpoint_density(paths, observed, out_dir / "endpoint_density.png")
    plot_model_votes(weights, out_dir / "model_vote_weights.png")
    plot_winning_parameter_histograms(posterior, winning_model, out_dir / "winning_model_parameter_histograms.png")

    summary = {
        "posterior": args.posterior,
        "n_paths": args.n_paths,
        "winning_model": winning_model,
        "model_vote_weights": {name: float(weight) for name, weight in zip(MODEL_NAMES, weights)},
        "sampled_model_counts": {
            name: int(np.sum(sampled_model_ids == i))
            for i, name in enumerate(MODEL_NAMES)
        },
        "endpoint_error_m": {
            "median": float(np.median(endpoint_error)),
            "p10": float(np.quantile(endpoint_error, 0.10)),
            "p90": float(np.quantile(endpoint_error, 0.90)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "posterior_predictive_samples.npz",
        observed=observed,
        paths=paths,
        sampled_model_ids=sampled_model_ids,
        endpoint_error=endpoint_error,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved model-voting evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
