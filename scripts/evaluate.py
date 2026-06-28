"""
evaluate.py
===========
End-to-end pipeline verification:

  1. Generate one ground-truth SDE track with known parameters
  2. Run the trained regressor to get predicted (mean, logvar) over θ
  3. Sample K parameter sets from the predicted distribution
  4. Simulate K SDE tracks using sampled parameters
  5. Visualize in an interactive Plotly figure:
       - Left:  3D trajectories (ground truth + simulated ensemble)
       - Right: predicted parameter distributions vs ground truth values

Usage
-----
python evaluate.py \
    --regressor_ckpt ./checkpoints/regressor_best.pt \
    --data_dir       ./data/lorenz_dataset \
    --n_samples      10 \
    --T              5.0 \
    --dt             0.005 \
    --seed           0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_data import simulate_batch
from src.data.dataset import SDEDataset, ParameterNormalizer
from src.models.encoder import TrajectoryEncoder, ProbabilisticRegressor


# ── Helpers ───────────────────────────────────────────────────────────────────

def sample_ground_truth(rng: np.random.Generator, T: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw one random parameter set and simulate a single ground-truth track.
    Parameters are sampled from the same hyperprior as training data so the
    regressor has seen this distribution.

    Returns
    -------
    params : (4,)        [sigma, rho, beta, noise_scale]
    track  : (steps, 3) xyz trajectory
    """
    sigma      = rng.uniform(1.0, 20.0)
    rho        = rng.uniform(0.5, 50.0)
    beta       = rng.uniform(0.5, 5.0)
    log_noise  = rng.uniform(-2.0, 0.176)
    noise      = 10 ** log_noise
    params     = np.array([sigma, rho, beta, noise], dtype=np.float64)

    y0         = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    t_grid     = np.arange(0, T + dt * 0.5, dt)
    track      = simulate_batch(sigma, rho, beta, noise, y0, t_grid)[0]  # (steps, 3)
    return params, track


def predict_params(
    regressor: ProbabilisticRegressor,
    track: np.ndarray,
    track_mean: np.ndarray,
    track_std: np.ndarray,
    normalizer: ParameterNormalizer,
    n_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode the track, get (mean, logvar), then draw n_samples parameter sets.

    Returns
    -------
    pred_mean    : (4,)          mean of predicted distribution (physical space)
    pred_std     : (4,)          std  of predicted distribution (physical space)
    param_samples: (n_samples, 4) sampled parameter sets        (physical space)
    """
    regressor.eval()
    with torch.no_grad():
        # Normalize track globally (same as dataset)
        norm_track = (track - track_mean) / (track_std + 1e-8)   # (steps, 3)
        x = torch.from_numpy(norm_track.T[None]).float().to(device)  # (1, 3, steps)

        mean_norm, logvar_norm = regressor(x)
        mean_norm   = mean_norm.cpu().numpy()[0]    # (4,) in [-1, 1]
        logvar_norm = logvar_norm.cpu().numpy()[0]  # (4,)
        std_norm    = np.exp(0.5 * logvar_norm)     # (4,)

    # Sample from predicted Gaussian in normalized space, then denormalize
    # Shape: (n_samples, 4)
    z             = np.random.randn(n_samples, 4)
    samples_norm  = mean_norm[None] + std_norm[None] * z        # (n_samples, 4)

    pred_mean     = normalizer.denormalize(mean_norm[None])[0]          # (4,)
    pred_std      = normalizer.denormalize(mean_norm + std_norm) \
                  - normalizer.denormalize(mean_norm - std_norm)      # approx (4,)
    param_samples = normalizer.denormalize(samples_norm)                 # (n_samples, 4)

    # Clip to physical bounds (samples can escape if std is large)
    param_samples[:, 0] = np.clip(param_samples[:, 0], 1.0,  20.0)   # sigma
    param_samples[:, 1] = np.clip(param_samples[:, 1], 0.5,  50.0)   # rho
    param_samples[:, 2] = np.clip(param_samples[:, 2], 0.5,   5.0)   # beta
    param_samples[:, 3] = np.clip(param_samples[:, 3], 0.01,  1.5)   # noise

    return pred_mean, pred_std, param_samples


def simulate_ensemble(
    param_samples: np.ndarray,
    T: float,
    dt: float,
    seed: int,
) -> list[np.ndarray]:
    """
    Simulate one track per sampled parameter set.
    Returns list of (steps, 3) arrays.
    """
    np.random.seed(seed)
    t_grid = np.arange(0, T + dt * 0.5, dt)
    tracks = []
    for p in param_samples:
        sigma, rho, beta, noise = p
        y0    = np.array([[1.0, 1.0, 1.0]])
        track = simulate_batch(sigma, rho, beta, noise, y0, t_grid)[0]
        tracks.append(track)
    return tracks


# ── Plotting ──────────────────────────────────────────────────────────────────

PARAM_NAMES  = ["σ (sigma)", "ρ (rho)", "β (beta)", "ε (noise)"]
PARAM_KEYS   = ["sigma", "rho", "beta", "noise"]

GT_COLOR  = "#E8593C"   # coral — ground truth
SIM_COLOR = "#3B8BD4"   # blue  — simulated ensemble


def build_figure(
    gt_track: np.ndarray,
    sim_tracks: list[np.ndarray],
    gt_params: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    param_samples: np.ndarray,
) -> go.Figure:
    """
    Two-panel Plotly figure:
      Left  (col 1): 3D trajectory — ground truth + simulated ensemble
      Right (col 2): 4 subplots — one per parameter, showing predicted
                     distribution (violin) vs ground truth (vertical line)
    """
    fig = make_subplots(
        rows=4, cols=2,
        column_widths=[0.55, 0.45],
        specs=[
            [{"type": "scatter3d", "rowspan": 4}, {"type": "violin"}],
            [None,                                 {"type": "violin"}],
            [None,                                 {"type": "violin"}],
            [None,                                 {"type": "violin"}],
        ],
        subplot_titles=(
            "3D trajectories",
            *PARAM_NAMES,
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.06,
    )

    # ── 3D trajectories ───────────────────────────────────────────────────────

    # Simulated ensemble (draw first so ground truth renders on top)
    for i, sim in enumerate(sim_tracks):
        fig.add_trace(
            go.Scatter3d(
                x=sim[:, 0], y=sim[:, 1], z=sim[:, 2],
                mode="lines",
                line=dict(color=SIM_COLOR, width=1),
                opacity=0.35,
                name="Simulated" if i == 0 else None,
                legendgroup="sim",
                showlegend=(i == 0),
            ),
            row=1, col=1,
        )

    # Ground truth
    fig.add_trace(
        go.Scatter3d(
            x=gt_track[:, 0], y=gt_track[:, 1], z=gt_track[:, 2],
            mode="lines",
            line=dict(color=GT_COLOR, width=3),
            name="Ground truth",
            legendgroup="gt",
        ),
        row=1, col=1,
    )

    # ── Parameter distributions ───────────────────────────────────────────────

    for i, (name, gt_val) in enumerate(zip(PARAM_NAMES, gt_params)):
        row = i + 1
        samples = param_samples[:, i]

        # Violin of sampled distribution
        fig.add_trace(
            go.Violin(
                y=samples,
                name=name,
                box_visible=True,
                meanline_visible=True,
                fillcolor=SIM_COLOR,
                line_color=SIM_COLOR,
                opacity=0.6,
                showlegend=False,
            ),
            row=row, col=2,
        )
        # Ground truth as a horizontal line
        # fig.add_hline(
        #     y=gt_val,
        #     line_dash="dash",
        #     line_color=GT_COLOR,
        #     line_width=2,
        #     row=row, col=2,
        # )

        # Annotation: ground truth value and predicted mean ± std
        fig.add_annotation(
            text=(
                f"GT: {gt_val:.3f}<br>"
                f"Pred: {pred_mean[i]:.3f} ± {pred_std[i]:.3f}"
            ),
            xref=f"x{i+2} domain" if i > 0 else "x2 domain",
            yref=f"y{i+2}" if i > 0 else "y2",
            x=1.0, y=gt_val,
            showarrow=False,
            font=dict(size=11, color=GT_COLOR),
            xanchor="right",
            bgcolor="rgba(255,255,255,0.7)",
            row=row, col=2,
        )

    # ── Layout ────────────────────────────────────────────────────────────────

    fig.update_layout(
        title=dict(
            text="SDE parameter inference — ground truth vs predicted ensemble",
            font=dict(size=15),
        ),
        height=800,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_scenes(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        bgcolor="rgb(245,245,245)",
    )

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regressor_ckpt", type=str, default="./checkpoints/regressor_best.pt")
    parser.add_argument("--data_dir",       type=str, default="./data/lorenz_dataset",
                        help="Needed to load global track mean/std from the training dataset")
    parser.add_argument("--n_samples",      type=int, default=10,
                        help="Number of parameter sets sampled from predicted distribution")
    parser.add_argument("--T",              type=float, default=5.0)
    parser.add_argument("--dt",             type=float, default=0.005)
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--out_html",       type=str, default="./evaluation.html",
                        help="Save interactive plot to this HTML file")
    args = parser.parse_args()

    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset stats (needed to normalize the track before encoding) ────
    # We only need track_mean / track_std / normalizer — not the full dataset.
    print("Loading dataset statistics...")
    dataset    = SDEDataset(args.data_dir)
    track_mean = dataset.track_mean   # (3,)
    track_std  = dataset.track_std    # (3,)
    normalizer = dataset.normalizer

    # ── Load regressor ────────────────────────────────────────────────────────
    encoder    = TrajectoryEncoder(feature_dim=256)
    regressor  = ProbabilisticRegressor(encoder, feature_dim=256, out_dim=4)
    ckpt_path  = Path(args.regressor_ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    regressor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    regressor.to(device)
    print(f"Loaded regressor from {ckpt_path}")

    # ── Step 1: ground-truth track ────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    gt_params, gt_track = sample_ground_truth(rng, args.T, args.dt)
    print(f"\nGround truth parameters:")
    for name, val in zip(PARAM_NAMES, gt_params):
        print(f"  {name:16s} = {val:.4f}")

    # ── Step 2: predict parameter distribution ────────────────────────────────
    pred_mean, pred_std, param_samples = predict_params(
        regressor, gt_track, track_mean, track_std,
        normalizer, args.n_samples, device,
    )
    print(f"\nPredicted parameters (mean ± std):")
    for name, m, s in zip(PARAM_NAMES, pred_mean, pred_std):
        print(f"  {name:16s} = {m:.4f} ± {s:.4f}")

    # ── Steps 3–4: simulate ensemble and visualize ────────────────────────────
    print(f"\nSimulating {args.n_samples} tracks from predicted parameters...")
    sim_tracks = simulate_ensemble(param_samples, args.T, args.dt, seed=args.seed + 1)

    fig = build_figure(gt_track, sim_tracks, gt_params, pred_mean, pred_std, param_samples)

    out_path = Path(args.out_html)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\nSaved interactive plot to {out_path}")
    print("Open it in a browser to explore the 3D trajectories and parameter distributions.")


if __name__ == "__main__":
    main()