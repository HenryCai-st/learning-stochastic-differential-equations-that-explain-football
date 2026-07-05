"""
scripts/04_evaluate.py
======================
Part 4 of 4 — Evaluation and Plotting

Loads the saved MCMC posterior chain (from step 3) and produces:

  1. posterior_histograms.png
     — For each parameter (σ, ρ, β, ε): prior histogram (grey) vs posterior histogram
       (colour), with a bold black vertical line at the ground-truth value θ*.

  2. future_paths.png
     — Observed trajectory (blue) overlaid with trajectories simulated from
       posterior-sampled parameters (green fan), showing the posterior-predictive
       distribution of future tracks.

  3. evaluation_summary.json
     — Numerical posterior mean, std, and coverage statistics.

Usage:
    python scripts/04_evaluate.py
    python scripts/04_evaluate.py --chain outputs/inference/posterior_chain.npz
                                  --dataset lorenz_dataset.npz
                                  --future-samples 30

Outputs (all under --out-dir, default outputs/evaluation/):
    posterior_histograms.png
    future_paths.png
    evaluation_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sde.lorenz_sde import PARAM_NAMES, simulate_lorenz_np
from src.utils.plotting import plot_posterior_vs_prior, plot_future_paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation and posterior visualisation.")
    p.add_argument("--chain",           default=str(ROOT / "outputs/inference/posterior_chain.npz"),
                   help="Path to posterior_chain.npz from step 3")
    p.add_argument("--dataset",         default=str(ROOT / "lorenz_dataset.npz"),
                   help="Dataset file (needed to load the observed trajectory)")
    p.add_argument("--future-samples",  type=int, default=30,
                   help="Number of future trajectories to simulate from posterior")
    p.add_argument("--future-steps",    type=int, default=500,
                   help="Steps per future trajectory")
    p.add_argument("--future-dt",       type=float, default=0.01)
    p.add_argument("--thin",            type=int, default=10,
                   help="Thinning factor for drawing posterior samples (reduces autocorrelation)")
    p.add_argument("--seed",            type=int, default=99)
    p.add_argument("--out-dir",         default=str(ROOT / "outputs/evaluation"))
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_trajectory(dataset_path: str, index: int) -> np.ndarray:
    raw   = np.load(dataset_path, allow_pickle=True)
    trajs = raw["trajectories"]
    traj  = trajs[index]
    if isinstance(traj, np.ndarray) and traj.dtype == object:
        traj = np.asarray(traj, dtype=np.float32)
    return np.asarray(traj, dtype=np.float32)


def coverage_rate(
    posterior: np.ndarray,   # (N, 4)
    true_theta: np.ndarray,  # (4,)
    percentile: float = 90.0,
) -> dict[str, float]:
    """
    Check whether the true parameter falls within the posterior's
    central credible interval at the given percentile.
    """
    lo_pct = (100 - percentile) / 2
    hi_pct = 100 - lo_pct
    lo = np.percentile(posterior, lo_pct, axis=0)
    hi = np.percentile(posterior, hi_pct, axis=0)
    covered = {
        n: bool(lo[k] <= true_theta[k] <= hi[k])
        for k, n in enumerate(PARAM_NAMES)
    }
    return covered


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng     = np.random.default_rng(args.seed)

    # ── 1. Load chain ─────────────────────────────────────────────────────────
    print(f"[04_evaluate] Loading chain from {args.chain} …")
    chain_data      = np.load(args.chain, allow_pickle=True)
    posterior       = chain_data["posterior"].astype(np.float64)    # (N, 4)
    observed_params = chain_data["observed_params"].astype(np.float64)  # (4,)
    observed_index  = int(chain_data["observed_index"])
    prior_bounds    = chain_data["prior_bounds"].astype(np.float64) # (4, 2)

    print(f"  posterior samples: {len(posterior)}")
    print(f"  observed θ*: { {n: round(float(v), 3) for n, v in zip(PARAM_NAMES, observed_params)} }")

    # ── 2. Thin the chain ─────────────────────────────────────────────────────
    thinned = posterior[::args.thin]
    print(f"  thinned samples (thin={args.thin}): {len(thinned)}")

    # ── 3. Plot posterior vs prior ─────────────────────────────────────────────
    plot_posterior_vs_prior(
        posterior_chain=thinned,
        observed_params=observed_params,
        prior_bounds=prior_bounds,
        out_path=out_dir / "posterior_histograms.png",
    )

    # ── 4. Load observed trajectory ───────────────────────────────────────────
    observed_traj = load_dataset_trajectory(args.dataset, observed_index)

    # ── 5. Simulate future paths from posterior samples ───────────────────────
    print(f"[04_evaluate] Simulating {args.future_samples} posterior-predictive trajectories …")
    step = max(1, len(thinned) // args.future_samples)
    selected = thinned[::step][:args.future_samples]

    future_paths: list[np.ndarray] = []
    for i, theta in enumerate(selected):
        sigma, rho, beta, epsilon = theta
        traj = simulate_lorenz_np(
            sigma=sigma, rho=rho, beta=beta, epsilon=epsilon,
            steps=args.future_steps, dt=args.future_dt,
            seed=args.seed + i,
        )
        future_paths.append(traj)

    plot_future_paths(observed_traj, future_paths, out_dir / "future_paths.png")

    # ── 6. Numerical evaluation ───────────────────────────────────────────────
    post_mean = posterior.mean(axis=0)
    post_std  = posterior.std(axis=0)
    abs_error = np.abs(post_mean - observed_params)
    coverage  = coverage_rate(posterior, observed_params, percentile=90.0)

    summary = {
        "n_posterior_samples": int(len(posterior)),
        "n_thinned":           int(len(thinned)),
        "observed_params":     {n: float(v) for n, v in zip(PARAM_NAMES, observed_params)},
        "posterior_mean":      {n: float(v) for n, v in zip(PARAM_NAMES, post_mean)},
        "posterior_std":       {n: float(v) for n, v in zip(PARAM_NAMES, post_std)},
        "abs_error_mean":      {n: float(v) for n, v in zip(PARAM_NAMES, abs_error)},
        "coverage_90pct":      coverage,
        "future_trajectories": args.future_samples,
    }

    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[04_evaluate] Done.  Outputs in {out_dir}")


if __name__ == "__main__":
    main()
