"""
evaluate.py
===========
Posterior predictive evaluation for route-A classification-based SBI.

Pipeline:
  1. Generate one ground-truth SDE trajectory with known parameters.
  2. Load the trained ratio classifier C_phi(D, theta).
  3. Score a candidate bank of theta values against the observed trajectory.
  4. Normalize scores into posterior weights over theta.
  5. Sample theta from the approximate posterior and simulate an ensemble.
  6. Visualize ground truth, posterior predictive trajectories, and parameter posterior.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_data import sample_parameters, simulate_batch
from recover_posterior import build_candidate_bank, load_ratio_classifier, score_candidates
from src.data.dataset import SDEDataset


PARAMETER_NAMES = ["sigma", "rho", "beta", "noise_scale"]
DISPLAY_NAMES = ["sigma", "rho", "beta", "noise"]
GT_COLOR = "#E8593C"
SIM_COLOR = "#3B8BD4"
POST_COLOR = "#48A868"


def sample_ground_truth(rng: np.random.Generator, T: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw one parameter set from the same Stage 1 prior and simulate a ground-truth track.

    Returns
    -------
    params : (4,) physical [sigma, rho, beta, noise_scale]
    track  : (steps, 3)
    y0     : (3,)
    """
    params = sample_parameters(1, rng)[0].astype(np.float64)
    y0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    t_grid = np.arange(0, T + dt, dt)
    track = simulate_batch(*params, y0[None], t_grid, rng)[0]
    return params, track, y0


@torch.no_grad()
def infer_posterior_for_track(
    model,
    track: np.ndarray,
    dataset: SDEDataset,
    candidate_bank_norm: np.ndarray,
    temperature: float,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    """Approximate p(theta | track) with normalized ratio-classifier scores."""
    norm_track = (track - dataset.track_mean) / (dataset.track_std + 1e-8)
    track_t = torch.from_numpy(norm_track.T[None]).float().to(device)
    candidates_t = torch.from_numpy(candidate_bank_norm).float().to(device)

    logits, weights = score_candidates(model, track_t, candidates_t, temperature)
    logits_np = logits.cpu().numpy()[0]
    weights_np = weights.cpu().numpy()[0]

    posterior_mean_norm = weights_np @ candidate_bank_norm
    map_idx = int(weights_np.argmax())
    map_norm = candidate_bank_norm[map_idx]

    candidate_bank_phys = dataset.normalizer.denormalize(candidate_bank_norm)
    posterior_mean_phys = dataset.normalizer.denormalize(posterior_mean_norm[None])[0]
    map_phys = dataset.normalizer.denormalize(map_norm[None])[0]

    entropy = float(-(weights_np * np.log(np.clip(weights_np, 1e-12, None))).sum())
    ess = float(1.0 / np.square(weights_np).sum())

    return {
        "logits": logits_np,
        "weights": weights_np,
        "candidate_bank_norm": candidate_bank_norm,
        "candidate_bank_phys": candidate_bank_phys,
        "posterior_mean_phys": posterior_mean_phys,
        "map_phys": map_phys,
        "map_idx": map_idx,
        "entropy": entropy,
        "ess": ess,
    }


def sample_posterior_parameters(
    posterior: dict[str, np.ndarray | float],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = posterior["candidate_bank_phys"]
    weights = posterior["weights"]
    indices = rng.choice(len(candidates), size=n_samples, replace=True, p=weights)
    return candidates[indices]


def simulate_ensemble(
    param_samples: np.ndarray,
    y0: np.ndarray,
    T: float,
    dt: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    t_grid = np.arange(0, T + dt, dt)
    tracks = []
    for params in param_samples:
        track = simulate_batch(*params, y0[None], t_grid, rng)[0]
        tracks.append(track)
    return tracks


def build_figure(
    gt_track: np.ndarray,
    sim_tracks: list[np.ndarray],
    gt_params: np.ndarray,
    posterior: dict[str, np.ndarray | float],
    posterior_samples: np.ndarray,
) -> go.Figure:
    stats_y_positions = [0.897, 0.632, 0.367, 0.102]

    fig = make_subplots(
        rows=4,
        cols=2,
        column_widths=[0.55, 0.45],
        specs=[
            [{"type": "scatter3d", "rowspan": 4}, {"type": "violin"}],
            [None, {"type": "violin"}],
            [None, {"type": "violin"}],
            [None, {"type": "violin"}],
        ],
        subplot_titles=("Posterior predictive trajectories", *DISPLAY_NAMES),
        horizontal_spacing=0.08,
        vertical_spacing=0.06,
    )

    for i, sim in enumerate(sim_tracks):
        fig.add_trace(
            go.Scatter3d(
                x=sim[:, 0], y=sim[:, 1], z=sim[:, 2],
                mode="lines",
                line=dict(color=SIM_COLOR, width=1),
                opacity=0.3,
                name="Posterior sample" if i == 0 else None,
                legendgroup="sim",
                showlegend=(i == 0),
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter3d(
            x=gt_track[:, 0], y=gt_track[:, 1], z=gt_track[:, 2],
            mode="lines",
            line=dict(color=GT_COLOR, width=4),
            name="Ground truth",
            legendgroup="gt",
        ),
        row=1, col=1,
    )

    posterior_mean = posterior["posterior_mean_phys"]
    map_theta = posterior["map_phys"]

    for i, name in enumerate(DISPLAY_NAMES):
        row = i + 1
        samples = posterior_samples[:, i]
        gt_val = gt_params[i]

        fig.add_trace(
            go.Violin(
                x=[name] * len(samples),
                y=samples,
                name=name,
                box_visible=True,
                meanline_visible=True,
                fillcolor=SIM_COLOR,
                line_color=SIM_COLOR,
                opacity=0.55,
                points=False,
                width=0.42,
                showlegend=False,
            ),
            row=row, col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[name],
                y=[gt_val],
                mode="markers",
                marker=dict(color=GT_COLOR, size=10, symbol="x"),
                name="GT parameter" if i == 0 else None,
                showlegend=(i == 0),
            ),
            row=row, col=2,
        )


        fig.add_annotation(
            text=(
                f"GT {gt_val:.3f}<br>"
                f"Mean {posterior_mean[i]:.3f}<br>"
                f"MAP {map_theta[i]:.3f}"
            ),
            xref="paper",
            yref="paper",
            x=1.01,
            y=stats_y_positions[i],
            showarrow=False,
            font=dict(size=10, color=GT_COLOR),
            xanchor="left",
            yanchor="middle",
            align="left",
            bgcolor="rgba(255,255,255,0.75)",
        )

    fig.update_layout(
        title=dict(
            text=(
                "Route-A SBI posterior predictive evaluation "
                f"(ESS={posterior['ess']:.1f}, entropy={posterior['entropy']:.2f})"
            ),
            font=dict(size=15),
        ),
        height=820,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
        margin=dict(l=20, r=130, t=70, b=20),
    )
    fig.update_scenes(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        bgcolor="rgb(245,245,245)",
    )
    for axis_id in ["xaxis", "xaxis2", "xaxis3", "xaxis4"]:
        fig.layout[axis_id].update(showgrid=False, zeroline=False, showticklabels=False)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio_ckpt", type=str, default="./checkpoints/ratio_classifier_best.pt")
    parser.add_argument("--data_dir", type=str, default="./data/lorenz_dataset")
    parser.add_argument("--candidate_source", type=str, default="prior", choices=["dataset", "prior"])
    parser.add_argument("--n_candidates", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=50, help="Posterior parameter samples used for ensemble simulation")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_html", type=str, default="./outputs/evaluation_posterior_predictive.html")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Posterior predictive evaluation")

    print("Loading dataset statistics and candidate prior...")
    dataset = SDEDataset(args.data_dir)
    candidate_bank_norm = build_candidate_bank(
        dataset,
        args.candidate_source,
        args.n_candidates,
        args.seed + 101,
    )
    print(f"Candidate source: {args.candidate_source} | candidates: {len(candidate_bank_norm)}")

    ckpt_path = Path(args.ratio_ckpt)
    if not ckpt_path.exists():
        print(f"Ratio classifier checkpoint not found: {ckpt_path}")
        print("Run scripts/train_ratio_classifier.py first.")
        sys.exit(1)

    model = load_ratio_classifier(ckpt_path, device)
    print(f"Loaded ratio classifier from {ckpt_path}")

    gt_params, gt_track, y0 = sample_ground_truth(rng, args.T, args.dt)
    print("\nGround truth parameters:")
    for name, val in zip(DISPLAY_NAMES, gt_params):
        print(f"  {name:11s} = {val:.4f}")

    posterior = infer_posterior_for_track(
        model,
        gt_track,
        dataset,
        candidate_bank_norm,
        args.temperature,
        device,
    )

    print("\nPosterior summary:")
    print(f"  ESS     = {posterior['ess']:.2f} / {len(candidate_bank_norm)}")
    print(f"  entropy = {posterior['entropy']:.3f}")
    for i, name in enumerate(DISPLAY_NAMES):
        print(
            f"  {name:11s} mean={posterior['posterior_mean_phys'][i]:.4f} "
            f"MAP={posterior['map_phys'][i]:.4f}"
        )

    posterior_samples = sample_posterior_parameters(posterior, args.n_samples, rng)
    print(f"\nSimulating {args.n_samples} posterior predictive trajectories...")
    sim_tracks = simulate_ensemble(posterior_samples, y0, args.T, args.dt, seed=args.seed + 1)

    fig = build_figure(gt_track, sim_tracks, gt_params, posterior, posterior_samples)

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"\nSaved interactive plot to {out_path}")


if __name__ == "__main__":
    main()
