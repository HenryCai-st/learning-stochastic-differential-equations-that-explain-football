"""
Evaluate the model-voting SBI posterior on a real football window.

Inputs:
    - outputs/model_voting_posterior/posterior_chains.npz from
      recover_model_voting_posterior.py.

Outputs:
    - posterior predictive path plot, endpoint density, model vote bar chart,
      winning-model parameter histograms, summary.json, and sampled paths.

Expected use:
    Run this after model-voting MCMC recovery to check whether sampled future
    trajectories cover the held-out suffix of a real ball window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sde.football_ou import PITCH_LENGTH, PITCH_WIDTH
from src.sde.model_voting import MAX_SEGMENTS, MODEL_NAMES, MODEL_PARAMETER_NAMES, MODEL_SPECS, simulate_model_batch
from src.utils.football_viz import pitch_background


MODEL_DISPLAY_NAMES = {
    "brownian": "Brownian\nsigma",
    "constant_velocity": "Constant velocity\nvx, vy, sigma",
    "ou_target": "OU target\nk, sigma",
    "piecewise_velocity": "Piecewise velocity\nv1, v2, v3, sigma",
}


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a compressed NumPy archive into a normal dictionary."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def scalar_string(value, default: str = "unknown") -> str:
    """Convert scalar values from `.npz` files into readable strings."""
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def sample_posterior_paths(
    posterior: dict[str, np.ndarray],
    n_paths: int,
    steps: int,
    dt: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample model/theta pairs from the posterior and simulate future paths."""
    rng = np.random.default_rng(seed)
    y0 = posterior.get("prediction_y0", posterior["y0"]).astype(np.float32)
    target = posterior.get("prediction_target", posterior["target"]).astype(np.float32)
    change_points = posterior["change_points"].astype(np.int64)
    observed = posterior["observed"].astype(np.float32)
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
        future_change_points = np.full(MAX_SEGMENTS - 1, steps + 1, dtype=np.int64)
        if model_name == "piecewise_velocity":
            # The detected change points describe the observed prefix. They are
            # not future events. Continue the most recent inferred segment and
            # assume no additional direction change over this short horizon.
            last_observed_step = len(observed) - 1
            latest_segment = int(np.sum(last_observed_step >= change_points))
            latest_segment = int(np.clip(latest_segment, 0, MAX_SEGMENTS - 1))
            latest_velocity = theta[:, 2 * latest_segment:2 * latest_segment + 2].copy()
            theta = theta.copy()
            for segment in range(MAX_SEGMENTS):
                theta[:, 2 * segment:2 * segment + 2] = latest_velocity
        paths = simulate_model_batch(
            model_name=model_name,
            params=theta,
            y0=np.repeat(y0[None], len(rows), axis=0),
            target=np.repeat(target[None], len(rows), axis=0),
            change_points=np.repeat(future_change_points[None], len(rows), axis=0),
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


def plot_posterior_predictive_paths(
    observed: np.ndarray,
    paths: np.ndarray,
    out_path: Path,
    future_suffix: np.ndarray | None = None,
) -> None:
    """Plot observed prefix, optional true future suffix, and sampled paths."""
    fig, ax = plt.subplots(figsize=(11, 7))
    pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)
    fig.patch.set_facecolor("#1a1a1a")

    for path in paths:
        ax.plot(path[:, 0], path[:, 1], color="#2ca02c", alpha=0.18, linewidth=0.8)
    ax.plot(observed[:, 0], observed[:, 1], color="#1f77b4", linewidth=3, label="observed prefix")
    if future_suffix is not None and len(future_suffix) > 0:
        true_future = np.vstack([observed[-1], future_suffix])
        ax.plot(true_future[:, 0], true_future[:, 1], color="#ffffff", linewidth=2.8, linestyle="--", label="true future")
    ax.plot(observed[0, 0], observed[0, 1], "o", color="#1f77b4", markersize=8, markeredgecolor="white", label="start")
    ax.plot(observed[-1, 0], observed[-1, 1], "s", color="#dc2626", markersize=8, markeredgecolor="white", label="prediction start")
    ax.set_title("Model-voting posterior predictive paths", color="white", fontsize=14, pad=10)
    ax.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_endpoint_density(paths: np.ndarray, observed: np.ndarray, out_path: Path, future_suffix: np.ndarray | None = None) -> None:
    """Plot a density map of posterior predictive final positions."""
    endpoints = paths[:, -1]
    fig, ax = plt.subplots(figsize=(11, 7))
    pitch_background(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)
    fig.patch.set_facecolor("#1a1a1a")
    hb = ax.hexbin(endpoints[:, 0], endpoints[:, 1], gridsize=38, cmap="magma", mincnt=1, alpha=0.85)
    ax.plot(observed[:, 0], observed[:, 1], color="#00bcd4", linewidth=2.5)
    if future_suffix is not None and len(future_suffix) > 0:
        ax.plot(future_suffix[:, 0], future_suffix[:, 1], color="white", linewidth=2.4, linestyle="--")
        ax.plot(future_suffix[-1, 0], future_suffix[-1, 1], "s", color="white", markersize=8, markeredgecolor="#00bcd4")
    else:
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
    """Render p(model | observed prefix) as a model-family bar chart."""
    labels = [MODEL_DISPLAY_NAMES[name] for name in MODEL_NAMES]
    colors = ["#e45756", "#4c78a8", "#54a24b", "#f58518"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(labels, weights, color=colors)
    for bar, weight in zip(bars, weights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{weight:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("candidate SDE model family")
    ax.set_ylabel("approximate model weight")
    ax.set_ylim(0.0, max(1.0, float(weights.max()) * 1.15))
    ax.set_title("Approximate model weights from prior-integrated ratio evidence")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)
    ax.text(
        0.01,
        -0.20,
        "Equal model priors; calibration must be checked on fresh synthetic and real windows.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def predictive_region_coverage(
    future_paths: np.ndarray,
    truth: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict[str, dict[str, float | bool]]:
    """Measure pointwise radial predictive-region coverage around the sample mean."""
    center = future_paths.mean(axis=0)
    sample_radius = np.linalg.norm(future_paths - center[None], axis=2)
    truth_radius = np.linalg.norm(truth - center, axis=1)
    result: dict[str, dict[str, float | bool]] = {}
    for level in levels:
        radius = np.quantile(sample_radius, level, axis=0)
        covered = truth_radius <= radius
        result[f"{int(level * 100)}pct"] = {
            "time_fraction_covered": float(covered.mean()),
            "endpoint_covered": bool(covered[-1]),
            "mean_radius_m": float(radius.mean()),
            "endpoint_radius_m": float(radius[-1]),
        }
    return result


def plot_winning_parameter_histograms(
    posterior: dict[str, np.ndarray],
    winning_model: str,
    out_path: Path,
) -> None:
    """Plot posterior parameter histograms for the highest-vote model family."""
    samples = posterior[f"{winning_model}_samples"].astype(np.float32)
    labels = MODEL_PARAMETER_NAMES[winning_model]
    n_params = MODEL_SPECS[winning_model].param_dim
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for dim in range(n_params):
        ax = axes[dim]
        ax.hist(samples[:, dim], bins=35, color="#4c78a8", edgecolor="white", linewidth=0.5)
        ax.set_title(f"theta[{dim}] = {labels[dim]}")
        ax.set_xlabel(f"{labels[dim]} value")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    for ax in axes[n_params:]:
        ax.axis("off")

    fig.suptitle(
        f"Parameter posterior p(theta | {winning_model}, observed prefix)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parse arguments, simulate posterior predictive paths, and save outputs."""
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
    future_suffix = posterior.get("future_suffix", np.empty((0, 2), dtype=np.float32)).astype(np.float32)
    if args.steps > 0:
        steps = args.steps
    elif len(future_suffix) > 0:
        # Include the prefix endpoint as step 0, then predict one point for each held-out suffix step.
        steps = len(future_suffix) + 1
    else:
        steps = len(observed)
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
    target_end = future_suffix[-1] if len(future_suffix) > 0 else observed[-1]
    endpoint_error = np.linalg.norm(endpoints - target_end[None], axis=1)
    if len(future_suffix) > 0 and paths.shape[1] == len(future_suffix) + 1:
        future_paths = paths[:, 1:]
        point_error = np.linalg.norm(future_paths - future_suffix[None], axis=2)
        path_error = point_error.mean(axis=1)
        final_error = point_error[:, -1]
        predictive_mean = future_paths.mean(axis=0)
        mean_path_point_error = np.linalg.norm(predictive_mean - future_suffix, axis=1)
        forecast_metrics = {
            "ade_predictive_mean_m": float(mean_path_point_error.mean()),
            "fde_predictive_mean_m": float(mean_path_point_error[-1]),
            "min_ade_over_samples_m": float(path_error.min()),
            "min_fde_over_samples_m": float(final_error.min()),
        }
        coverage = predictive_region_coverage(future_paths, future_suffix)
    else:
        path_error = np.full(len(paths), np.nan, dtype=np.float32)
        forecast_metrics = {
            "ade_predictive_mean_m": None,
            "fde_predictive_mean_m": None,
            "min_ade_over_samples_m": None,
            "min_fde_over_samples_m": None,
        }
        coverage = {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_posterior_predictive_paths(observed, paths, out_dir / "posterior_predictive_paths.png", future_suffix)
    plot_endpoint_density(paths, observed, out_dir / "endpoint_density.png", future_suffix)
    plot_model_votes(weights, out_dir / "model_vote_weights.png")
    plot_winning_parameter_histograms(posterior, winning_model, out_dir / "winning_model_parameter_histograms.png")

    summary = {
        "posterior": args.posterior,
        "n_paths": args.n_paths,
        "winning_model": winning_model,
        "protocol": scalar_string(posterior.get("protocol")),
        "model_weight_method": scalar_string(posterior.get("model_weight_method"), "prior_mc_ratio_evidence"),
        "model_weight_status": scalar_string(posterior.get("model_weight_status"), "approximate"),
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
        "future_path_error_m": {
            "median": None if np.isnan(path_error).all() else float(np.nanmedian(path_error)),
            "p10": None if np.isnan(path_error).all() else float(np.nanquantile(path_error, 0.10)),
            "p90": None if np.isnan(path_error).all() else float(np.nanquantile(path_error, 0.90)),
        },
        "forecast_metrics": forecast_metrics,
        "predictive_region_coverage": coverage,
        "coverage_note": (
            "Coverage is pointwise radial coverage for this one held-out suffix; "
            "calibration requires aggregation over many independent windows."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "posterior_predictive_samples.npz",
        observed=observed,
        paths=paths,
        future_suffix=future_suffix,
        sampled_model_ids=sampled_model_ids,
        endpoint_error=endpoint_error,
        path_error=path_error,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved model-voting evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
