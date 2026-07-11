"""
Score one real football window with the single-model OU baseline.

Inputs:
    - data/real_football_windows.npz from extract_football_windows.py
    - checkpoints/football_ou_ratio_best.pt from train_football_ou_ratio.py

Outputs:
    - top candidate CSV
    - posterior predictive SVG
    - parameter distribution SVG
    - summary.json and posterior_predictive.npz

Expected use:
    Use this as the OU baseline/failure comparison for the final model-voting
    presentation. The final project workflow uses model-voting scripts instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_football_ou_ratio import ConditionedRatioClassifier
from src.data.football_dataset import OUParameterNormalizer, RealFootballWindows
from src.models.encoder import TrajectoryEncoder
from src.sde.football_ou import PARAMETER_HIGH, PARAMETER_LOW, sample_ou_parameters, simulate_position_ou_batch


PARAMETER_NAMES = ("k", "noise_scale")


def load_model(path: Path, device: torch.device):
    """Load the trained OU baseline ratio classifier and checkpoint metadata."""
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run scripts/train_football_ou_ratio.py first."
        )

    # This is a local project checkpoint that also stores dataset statistics.
    # PyTorch 2.6 changed torch.load's default to weights_only=True, which
    # rejects older checkpoints containing NumPy arrays such as track_mean/std.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder = TrajectoryEncoder(in_channels=2, feature_dim=int(ckpt.get("feature_dim", 256)))
    model = ConditionedRatioClassifier(encoder, feature_dim=int(ckpt.get("feature_dim", 256)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def checkpoint_array(ckpt: dict, key: str) -> np.ndarray:
    """Read checkpoint statistics saved either as tensors or NumPy arrays."""
    value = ckpt[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def write_future_svg(out_path: Path, observed: np.ndarray, futures: list[np.ndarray]) -> None:
    """Small dependency-free SVG fan plot for posterior predictive tracks."""
    width, height, pad = 760, 500, 45
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="28" text-anchor="middle" font-size="20">Football OU posterior predictive paths</text>',
        f'<rect x="{pad}" y="{pad}" width="{width-2*pad}" height="{height-2*pad}" fill="#f6fff6" stroke="#333"/>',
    ]

    def pts(path):
        """Convert pitch-metre coordinates into SVG polyline point text."""
        x = pad + path[:, 0] / 105.0 * (width - 2 * pad)
        y = height - pad - path[:, 1] / 68.0 * (height - 2 * pad)
        return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))

    for f in futures:
        lines.append(f'<polyline points="{pts(f)}" fill="none" stroke="#2ca02c" stroke-width="1" opacity="0.28"/>')
    lines.append(f'<polyline points="{pts(observed)}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    start_x = pad + observed[0, 0] / 105.0 * (width - 2 * pad)
    start_y = height - pad - observed[0, 1] / 68.0 * (height - 2 * pad)
    end_x = pad + observed[-1, 0] / 105.0 * (width - 2 * pad)
    end_y = height - pad - observed[-1, 1] / 68.0 * (height - 2 * pad)
    lines.append(f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="6" fill="#1f77b4" stroke="white" stroke-width="2"/>')
    lines.append(f'<rect x="{end_x-6:.1f}" y="{end_y-6:.1f}" width="12" height="12" fill="#dc2626" stroke="white" stroke-width="2"/>')
    lines.append(f'<text x="{start_x+9:.1f}" y="{start_y-9:.1f}" font-size="12" fill="#1f77b4" font-weight="bold">start</text>')
    lines.append(f'<text x="{end_x+9:.1f}" y="{end_y-9:.1f}" font-size="12" fill="#dc2626" font-weight="bold">end</text>')
    lines.append('<text x="560" y="70" font-size="13" fill="#1f77b4">blue = real observed window</text>')
    lines.append('<text x="560" y="90" font-size="13" fill="#2ca02c">green = posterior predictive</text>')
    lines.append('<text x="560" y="110" font-size="13" fill="#1f77b4">circle = start</text>')
    lines.append('<text x="560" y="130" font-size="13" fill="#dc2626">square = end</text>')
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def ou_transition_loglik_per_step(
    observed: np.ndarray,
    candidates: np.ndarray,
    target: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Average Gaussian Euler-transition log-likelihood for the football OU baseline.

    This scores whether a parameter can explain the observed *increments*:
        x[t+1] = x[t] + k * (target - x[t]) dt + noise * sqrt(dt) eps

    It is used as a calibration term on top of the neural ratio score. Without
    it, the classifier can rank high-noise candidates that are plausible in a
    broad sense but render poor posterior predictive paths.
    """
    x_t = observed[:-1].astype(np.float32)
    x_next = observed[1:].astype(np.float32)
    target = target.astype(np.float32)
    k = candidates[:, 0].astype(np.float32)
    noise = np.clip(candidates[:, 1].astype(np.float32), 1e-6, None)

    expected = x_t[None, :, :] + k[:, None, None] * (target[None, None, :] - x_t[None, :, :]) * dt
    residual = x_next[None, :, :] - expected
    var = (noise ** 2)[:, None, None] * dt
    loglik = -0.5 * (residual ** 2 / var) - np.log(noise[:, None, None] * np.sqrt(dt))
    return loglik.mean(axis=(1, 2))


def log_prior_ou(theta: np.ndarray) -> float:
    """
    Log prior for football OU baseline parameters.

    k is uniform. noise_scale follows the log-uniform prior used by
    sample_ou_parameters(), so p(noise) is proportional to 1 / noise inside
    support. Constants are omitted because MCMC only needs ratios.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if np.any(theta < PARAMETER_LOW) or np.any(theta > PARAMETER_HIGH):
        return -np.inf
    if theta[1] <= 0:
        return -np.inf
    return -float(np.log(theta[1]))


@torch.no_grad()
def classifier_logits_for_params(
    model: ConditionedRatioClassifier,
    normalizer: OUParameterNormalizer,
    track_t: torch.Tensor,
    condition_t: torch.Tensor,
    params: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Evaluate C_phi(track, theta, y0, target) for many candidate theta values.

    This keeps inference memory bounded. The observed track and condition are
    repeated to match the candidate batch.
    """
    params = np.asarray(params, dtype=np.float32)
    out: list[np.ndarray] = []
    for start in range(0, len(params), batch_size):
        chunk = params[start:start + batch_size]
        params_norm = normalizer.normalize(chunk).astype(np.float32)
        params_t = torch.from_numpy(params_norm).to(device)
        tracks_t = track_t.repeat(len(chunk), 1, 1)
        cond_t = condition_t.repeat(len(chunk), 1)
        out.append(model(tracks_t, params_t, cond_t).cpu().numpy())
    return np.concatenate(out)


def deterministic_ou_path(
    params: np.ndarray,
    y0: np.ndarray,
    target: np.ndarray,
    steps: int,
    dt: float,
) -> np.ndarray:
    """Simulate deterministic OU mean paths without diffusion noise."""
    paths = np.zeros((len(params), steps, 2), dtype=np.float32)
    paths[:, 0] = y0[None]
    current = np.repeat(y0[None].astype(np.float32), len(params), axis=0)
    k = params[:, 0].astype(np.float32)
    target_batch = np.repeat(target[None].astype(np.float32), len(params), axis=0)
    for t in range(1, steps):
        current = current + k[:, None] * (target_batch - current) * dt
        paths[:, t] = current
    return paths


def rmse_to_observed(paths: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Compute per-path RMSE against one observed trajectory."""
    return np.sqrt(((paths - observed[None]) ** 2).sum(axis=2).mean(axis=1))


def bridge_to_endpoint(paths: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Linearly condition simulated paths to end at the observed endpoint.

    This is only for posterior-predictive visualization of an observed window
    where the endpoint is already part of the conditioning input.
    """
    alpha = np.linspace(0.0, 1.0, paths.shape[1], dtype=np.float32)[None, :, None]
    correction = target[None, None, :].astype(np.float32) - paths[:, -1:, :]
    return paths + alpha * correction


def mcmc_log_target(
    theta: np.ndarray,
    model: ConditionedRatioClassifier,
    normalizer: OUParameterNormalizer,
    track_t: torch.Tensor,
    condition_t: torch.Tensor,
    observed: np.ndarray,
    target: np.ndarray,
    dt: float,
    likelihood_weight: float,
    device: torch.device,
) -> float:
    """
    Unnormalized posterior target for RWMH:

        log p(theta | track) ~= log prior(theta)
                              + classifier_logit(track, theta, condition)
                              + w * OU_transition_loglik(track | theta)
    """
    lp = log_prior_ou(theta)
    if not np.isfinite(lp):
        return -np.inf
    logit = classifier_logits_for_params(
        model,
        normalizer,
        track_t,
        condition_t,
        theta[None].astype(np.float32),
        device,
        batch_size=1,
    )[0]
    ll = ou_transition_loglik_per_step(observed, theta[None].astype(np.float32), target, dt)[0]
    return float(lp + logit + likelihood_weight * ll)


def run_rwmh(
    initial_theta: np.ndarray,
    proposal_scale: np.ndarray,
    n_steps: int,
    burn_in: int,
    rng: np.random.Generator,
    log_target_fn,
) -> dict[str, np.ndarray | float]:
    """Random-walk Metropolis-Hastings in physical `(k, noise_scale)` space."""
    current = np.asarray(initial_theta, dtype=np.float64).copy()
    current_logp = log_target_fn(current)
    chain = np.zeros((n_steps, 2), dtype=np.float32)
    logp = np.zeros(n_steps, dtype=np.float32)
    accepted = 0

    for step in range(n_steps):
        proposal = current + rng.normal(0.0, proposal_scale, size=2)
        proposal_logp = log_target_fn(proposal)
        if np.log(rng.uniform()) < proposal_logp - current_logp:
            current = proposal
            current_logp = proposal_logp
            accepted += 1
        chain[step] = current
        logp[step] = current_logp

    samples = chain[burn_in:]
    logp_samples = logp[burn_in:]
    map_idx = int(np.argmax(logp_samples))
    return {
        "chain": chain,
        "logp": logp,
        "samples": samples,
        "posterior_mean": samples.mean(axis=0),
        "map": samples[map_idx],
        "acceptance_rate": accepted / max(1, n_steps),
    }


def write_histogram_svg(
    out_path: Path,
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    bins: int = 40,
) -> None:
    """Dependency-free prior/posterior histogram figure for k and noise_scale."""
    width, height = 900, 420
    panel_w = width // 2
    pad_l, pad_r, pad_t, pad_b = 55, 25, 55, 55
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" font-size="20">Football OU prior vs posterior</text>',
    ]

    colors = [("#9ca3af", "#2563eb"), ("#9ca3af", "#dc2626")]
    for dim, name in enumerate(PARAMETER_NAMES):
        x0 = dim * panel_w
        prior = prior_samples[:, dim]
        post = posterior_samples[:, dim]
        lo = min(prior.min(), post.min())
        hi = max(prior.max(), post.max())
        edges = np.linspace(lo, hi, bins + 1)
        prior_counts, _ = np.histogram(prior, bins=edges, density=True)
        post_counts, _ = np.histogram(post, bins=edges, density=True)
        ymax = max(prior_counts.max(), post_counts.max(), 1e-8)
        plot_x = x0 + pad_l
        plot_y = pad_t
        plot_w = panel_w - pad_l - pad_r
        plot_h = height - pad_t - pad_b

        lines.append(f'<line x1="{plot_x}" y1="{plot_y+plot_h}" x2="{plot_x+plot_w}" y2="{plot_y+plot_h}" stroke="#333"/>')
        lines.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y+plot_h}" stroke="#333"/>')
        lines.append(f'<text x="{x0+panel_w/2}" y="{height-16}" text-anchor="middle" font-size="13">{name}</text>')
        lines.append(f'<text x="{plot_x-38}" y="{plot_y+plot_h/2}" text-anchor="middle" font-size="12" transform="rotate(-90 {plot_x-38} {plot_y+plot_h/2})">density</text>')
        lines.append(f'<text x="{x0+panel_w/2}" y="50" text-anchor="middle" font-size="15">Prior vs posterior: {name}</text>')

        bar_w = plot_w / bins
        prior_color, post_color = colors[dim]
        for i in range(bins):
            px = plot_x + i * bar_w
            ph = prior_counts[i] / ymax * plot_h
            qh = post_counts[i] / ymax * plot_h
            lines.append(f'<rect x="{px:.2f}" y="{plot_y+plot_h-ph:.2f}" width="{bar_w:.2f}" height="{ph:.2f}" fill="{prior_color}" opacity="0.38"/>')
            lines.append(f'<rect x="{px:.2f}" y="{plot_y+plot_h-qh:.2f}" width="{bar_w:.2f}" height="{qh:.2f}" fill="{post_color}" opacity="0.62"/>')

        # Numeric x-axis ticks in physical parameter units.
        for tick in np.linspace(lo, hi, 5):
            tx = plot_x + (tick - lo) / max(hi - lo, 1e-8) * plot_w
            lines.append(f'<line x1="{tx:.1f}" y1="{plot_y+plot_h:.1f}" x2="{tx:.1f}" y2="{plot_y+plot_h+5:.1f}" stroke="#333"/>')
            lines.append(f'<text x="{tx:.1f}" y="{plot_y+plot_h+20:.1f}" text-anchor="middle" font-size="10">{tick:.2g}</text>')

        # Numeric y-axis ticks in density units.
        for tick in np.linspace(0.0, ymax, 4):
            ty = plot_y + plot_h - tick / max(ymax, 1e-8) * plot_h
            lines.append(f'<line x1="{plot_x-5:.1f}" y1="{ty:.1f}" x2="{plot_x:.1f}" y2="{ty:.1f}" stroke="#333"/>')
            lines.append(f'<text x="{plot_x-8:.1f}" y="{ty+3:.1f}" text-anchor="end" font-size="10">{tick:.2g}</text>')

        lines.append(f'<text x="{plot_x+plot_w-130}" y="{plot_y+20}" font-size="12" fill="{prior_color}">prior</text>')
        lines.append(f'<text x="{plot_x+plot_w-130}" y="{plot_y+40}" font-size="12" fill="{post_color}">posterior</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def main() -> None:
    """Score one real window, run candidate/MCMC inference, and save OU plots."""
    parser = argparse.ArgumentParser(description="Score a real football window with a trained OU ratio classifier.")
    parser.add_argument("--real-windows", default="data/real_football_windows.npz")
    parser.add_argument("--checkpoint", default="checkpoints/football_ou_ratio_best.pt")
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--n-candidates", type=int, default=5000)
    parser.add_argument("--n-predictive", type=int, default=80)
    parser.add_argument("--sampler", choices=["mcmc", "candidates"], default="mcmc")
    parser.add_argument("--mcmc-steps", type=int, default=6000)
    parser.add_argument("--burn-in", type=int, default=1500)
    parser.add_argument("--proposal-scale", nargs=2, type=float, default=[0.08, 0.12])
    parser.add_argument(
        "--likelihood-weight",
        type=float,
        default=1.0,
        help="Weight for the OU transition likelihood calibration term.",
    )
    parser.add_argument(
        "--no-bridge-endpoint",
        action="store_true",
        help="Disable endpoint bridge used for observed-window predictive rendering.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="outputs/football_ou_real")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(Path(args.checkpoint), device)
    if not Path(args.real_windows).exists():
        raise FileNotFoundError(
            f"Real windows not found: {args.real_windows}. Run scripts/extract_football_windows.py first."
        )
    real = RealFootballWindows(
        args.real_windows,
        track_mean=checkpoint_array(ckpt, "track_mean"),
        track_std=checkpoint_array(ckpt, "track_std"),
    )
    item = real[args.window_index]
    rng = np.random.default_rng(args.seed)
    normalizer = OUParameterNormalizer()

    track = item["track"][None].to(device)
    condition = item["condition"][None].to(device).repeat(args.n_candidates, 1)
    condition_single = item["condition"][None].to(device)
    observed = real.tracks[args.window_index]
    y0 = real.y0[args.window_index]
    target = real.target[args.window_index]
    real_npz = np.load(args.real_windows, allow_pickle=True)
    dt = float(real_npz["dt"])

    # Candidate grid is useful both as a fast fallback and as a robust MCMC
    # initializer. It approximates the prior from which the posterior is scored.
    candidates = sample_ou_parameters(args.n_candidates, rng)
    logits = classifier_logits_for_params(model, normalizer, track, condition_single, candidates, device)
    transition_loglik = ou_transition_loglik_per_step(observed, candidates, target, dt)
    # Put the classifier and likelihood on comparable scales before combining.
    logits_z = (logits - logits.mean()) / (logits.std() + 1e-8)
    ll_z = (transition_loglik - transition_loglik.mean()) / (transition_loglik.std() + 1e-8)
    combined_score = logits_z + args.likelihood_weight * ll_z

    mean_paths = deterministic_ou_path(candidates, y0, target, len(observed), dt)
    mean_rmse = rmse_to_observed(mean_paths, observed)
    order = np.argsort(-combined_score)

    if args.sampler == "mcmc":
        log_target_fn = lambda theta: mcmc_log_target(
            theta=theta,
            model=model,
            normalizer=normalizer,
            track_t=track,
            condition_t=condition_single,
            observed=observed,
            target=target,
            dt=dt,
            likelihood_weight=args.likelihood_weight,
            device=device,
        )
        mcmc_result = run_rwmh(
            initial_theta=candidates[order[0]],
            proposal_scale=np.asarray(args.proposal_scale, dtype=np.float64),
            n_steps=args.mcmc_steps,
            burn_in=args.burn_in,
            rng=rng,
            log_target_fn=log_target_fn,
        )
        posterior_samples = np.asarray(mcmc_result["samples"], dtype=np.float32)
        if len(posterior_samples) > args.n_predictive:
            chosen = rng.choice(len(posterior_samples), size=args.n_predictive, replace=False)
            top = posterior_samples[chosen]
        else:
            top = posterior_samples
    else:
        mcmc_result = None
        posterior_samples = candidates[order[: max(args.n_predictive, 200)]]
        top = candidates[order[: args.n_predictive]]

    futures = simulate_position_ou_batch(
        params=top,
        y0=np.repeat(y0[None], len(top), axis=0),
        target=np.repeat(target[None], len(top), axis=0),
        steps=len(observed),
        dt=dt,
        rng=rng,
    )
    if not args.no_bridge_endpoint:
        futures = bridge_to_endpoint(futures, target)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "top_candidates.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank",
            "k",
            "noise_scale",
            "classifier_logit",
            "transition_loglik",
            "mean_path_rmse_m",
            "combined_score",
        ])
        for rank, idx in enumerate(order[:100], 1):
            writer.writerow([
                rank,
                float(candidates[idx, 0]),
                float(candidates[idx, 1]),
                float(logits[idx]),
                float(transition_loglik[idx]),
                float(mean_rmse[idx]),
                float(combined_score[idx]),
            ])
    write_future_svg(out_dir / "posterior_predictive.svg", observed, list(futures))
    prior_for_plot = sample_ou_parameters(max(args.n_candidates, 5000), rng)
    write_histogram_svg(out_dir / "parameter_distributions.svg", prior_for_plot, posterior_samples)
    future_rmse = rmse_to_observed(futures, observed)
    summary = {
        "window_index": args.window_index,
        "dt": dt,
        "observed_start": y0.tolist(),
        "observed_target": target.tolist(),
        "observed_displacement_m": float(np.linalg.norm(target - y0)),
        "observed_mean_step_m": float(np.linalg.norm(np.diff(observed, axis=0), axis=1).mean()),
        "likelihood_weight": args.likelihood_weight,
        "sampler": args.sampler,
        "mcmc_acceptance_rate": None if mcmc_result is None else float(mcmc_result["acceptance_rate"]),
        "endpoint_bridge": not args.no_bridge_endpoint,
        "best_candidate": {
            "k": float(candidates[order[0], 0]),
            "noise_scale": float(candidates[order[0], 1]),
            "classifier_logit": float(logits[order[0]]),
            "transition_loglik": float(transition_loglik[order[0]]),
            "mean_path_rmse_m": float(mean_rmse[order[0]]),
            "combined_score": float(combined_score[order[0]]),
        },
        "posterior_parameter_summary": {
            "mean": {name: float(value) for name, value in zip(PARAMETER_NAMES, posterior_samples.mean(axis=0))},
            "std": {name: float(value) for name, value in zip(PARAMETER_NAMES, posterior_samples.std(axis=0))},
            "map_or_best": (
                {name: float(value) for name, value in zip(PARAMETER_NAMES, mcmc_result["map"])}
                if mcmc_result is not None
                else {name: float(value) for name, value in zip(PARAMETER_NAMES, candidates[order[0]])}
            ),
        },
        "posterior_predictive_rmse_m": {
            "best": float(future_rmse.min()),
            "median": float(np.median(future_rmse)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "posterior_predictive.npz",
        observed=observed,
        y0=y0,
        target=target,
        candidates=candidates,
        logits=logits,
        transition_loglik=transition_loglik,
        combined_score=combined_score,
        mean_rmse=mean_rmse,
        posterior_samples=posterior_samples,
        mcmc_chain=None if mcmc_result is None else mcmc_result["chain"],
        mcmc_logp=None if mcmc_result is None else mcmc_result["logp"],
        futures=futures,
        parameter_low=PARAMETER_LOW,
        parameter_high=PARAMETER_HIGH,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
