"""
evaluate_mcmc.py
================
Posterior predictive evaluation using Random-Walk Metropolis-Hastings.

Figures produced:
  1. Posterior predictive 3D trajectories.
  2. Prior vs posterior parameter histograms with bold ground-truth lines.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_data import sample_parameters, simulate_batch
from recover_posterior import load_ratio_classifier, random_walk_metropolis_hastings
from src.data.dataset import SDEDataset

DISPLAY_NAMES = ["sigma", "rho", "beta", "noise"]
GT_COLOR = "#E8593C"
SIM_COLOR = "#3B8BD4"


def sample_ground_truth(rng: np.random.Generator, T: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = sample_parameters(1, rng)[0].astype(np.float64)
    y0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    t_grid = np.arange(0, T + dt, dt)
    track = simulate_batch(*params, y0[None], t_grid, rng)[0]
    return params, track, y0


def simulate_ensemble(param_samples: np.ndarray, y0: np.ndarray, T: float, dt: float, seed: int, max_tracks: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    t_grid = np.arange(0, T + dt, dt)
    if len(param_samples) > max_tracks:
        idx = rng.choice(len(param_samples), size=max_tracks, replace=False)
        param_samples = param_samples[idx]
    return [simulate_batch(*params, y0[None], t_grid, rng)[0] for params in param_samples]


def build_predictive_figure(gt_track: np.ndarray, sim_tracks: list[np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for i, sim in enumerate(sim_tracks):
        fig.add_trace(go.Scatter3d(
            x=sim[:, 0], y=sim[:, 1], z=sim[:, 2],
            mode="lines",
            line=dict(color=SIM_COLOR, width=1),
            opacity=0.25,
            name="Posterior predictive" if i == 0 else None,
            showlegend=(i == 0),
        ))
    fig.add_trace(go.Scatter3d(
        x=gt_track[:, 0], y=gt_track[:, 1], z=gt_track[:, 2],
        mode="lines",
        line=dict(color=GT_COLOR, width=5),
        name="Ground truth",
    ))
    fig.update_layout(
        title="Posterior predictive trajectories from MH posterior samples",
        height=760,
        paper_bgcolor="white",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_prior_posterior_histograms(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    gt_params: np.ndarray,
    out_path: Path,
    bins: int = 40,
) -> None:
    """Simple prior/posterior histograms with bold ground-truth vertical lines."""
    fig, axes = plt.subplots(4, 2, figsize=(10, 11))
    for i, name in enumerate(DISPLAY_NAMES):
        ax_prior = axes[i, 0]
        ax_post = axes[i, 1]

        ax_prior.hist(prior_samples[:, i], bins=bins, density=True, alpha=0.75)
        ax_prior.axvline(gt_params[i], color="black", linewidth=3.5, label="ground truth")
        ax_prior.set_title(f"Prior: {name}")
        ax_prior.set_ylabel("density")
        ax_prior.grid(True, alpha=0.25)

        ax_post.hist(posterior_samples[:, i], bins=bins, density=True, alpha=0.75)
        ax_post.axvline(gt_params[i], color="black", linewidth=3.5, label="ground truth")
        ax_post.set_title(f"Posterior: {name}")
        ax_post.grid(True, alpha=0.25)

        if i == 0:
            ax_prior.legend(frameon=True)
            ax_post.legend(frameon=True)

    fig.suptitle("Prior and posterior parameter distributions with ground truth", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio_ckpt", type=str, default="./checkpoints/ratio_classifier_best.pt")
    parser.add_argument("--data_dir", type=str, default="./data/lorenz_dataset")
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcmc_steps", type=int, default=6000)
    parser.add_argument("--burn_in", type=int, default=1500)
    parser.add_argument("--proposal_scale", nargs=4, type=float, default=[0.7, 1.5, 0.15, 0.02])
    parser.add_argument("--n_prior", type=int, default=6000)
    parser.add_argument("--n_predictive", type=int, default=60)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | MCMC posterior predictive evaluation")

    dataset = SDEDataset(args.data_dir)
    ckpt_path = Path(args.ratio_ckpt)
    if not ckpt_path.exists():
        print(f"Ratio classifier checkpoint not found: {ckpt_path}")
        print("Run train_ratio_classifier.py first.")
        sys.exit(1)
    model = load_ratio_classifier(ckpt_path, device)

    gt_params, gt_track, y0 = sample_ground_truth(rng, args.T, args.dt)
    print("\nGround truth parameters:")
    for name, value in zip(DISPLAY_NAMES, gt_params):
        print(f"  {name:8s} = {value:.4f}")

    result = random_walk_metropolis_hastings(
        model=model,
        track=gt_track,
        dataset=dataset,
        n_steps=args.mcmc_steps,
        burn_in=args.burn_in,
        proposal_scale=np.asarray(args.proposal_scale, dtype=np.float64),
        seed=args.seed + 123,
        device=device,
    )
    posterior_samples = result["samples"]
    print(f"\nMH acceptance rate: {result['acceptance_rate']:.2%}")
    print("Posterior summary:")
    for i, name in enumerate(DISPLAY_NAMES):
        print(
            f"  {name:8s} mean={result['posterior_mean_phys'][i]:.4f} "
            f"MAP={result['map_phys'][i]:.4f} GT={gt_params[i]:.4f}"
        )

    prior_samples = sample_parameters(args.n_prior, rng)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_path = out_dir / "prior_posterior_histograms.png"
    plot_prior_posterior_histograms(prior_samples, posterior_samples, gt_params, hist_path)
    print(f"Saved prior/posterior histogram figure to {hist_path}")

    sim_tracks = simulate_ensemble(posterior_samples, y0, args.T, args.dt, args.seed + 456, args.n_predictive)
    predictive_fig = build_predictive_figure(gt_track, sim_tracks)
    predictive_path = out_dir / "mcmc_posterior_predictive.html"
    predictive_fig.write_html(str(predictive_path), include_plotlyjs=True)
    print(f"Saved posterior predictive figure to {predictive_path}")

    sample_path = out_dir / "mcmc_single_track_posterior.npz"
    np.savez_compressed(
        sample_path,
        posterior_samples=posterior_samples,
        prior_samples=prior_samples,
        gt_params=gt_params,
        chain=result["chain"],
        logp=result["logp"],
        acceptance_rate=result["acceptance_rate"],
    )
    print(f"Saved posterior samples to {sample_path}")


if __name__ == "__main__":
    main()
