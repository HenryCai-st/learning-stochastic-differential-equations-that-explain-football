"""
scripts/01_generate_data.py
===========================
Part 1 of 4 — Data Generation

Generates the Lorenz SDE dataset and saves:
  • data.npz                            — dataset file
  • outputs/generated_diversity.png     — diversity visualisation

Usage:
    python scripts/01_generate_data.py [--n-samples 1000] [--seed 42] [--out data.npz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.plotting import plot_generated_diversity


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Lorenz SDE dataset.")
    parser.add_argument("--n-samples", type=int,   default=500,
                        help="Number of samples to attempt (some may be discarded)")
    parser.add_argument("--T",         type=float, default=50.0,
                        help="Simulation time per trajectory")
    parser.add_argument("--dt",        type=float, default=0.01,
                        help="Euler-Maruyama step size")
    parser.add_argument("--cut",       type=int,   default=1000,
                        help="Transient steps to discard after simulation")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--out",       default=str(ROOT / "data" / "data.npz"),
                        help="Output path for the .npz dataset")
    parser.add_argument("--out-dir",   default=str(ROOT / "outputs"),
                        help="Directory for visualisation outputs")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_params(rng: np.random.Generator) -> tuple:
    """
    Sample one parameter set from the prior.

    σ ~ Uniform(1, 20)
    β ~ Uniform(0.5, 5)
    ρ ~ stratified over three regimes:
        Regime 1: fixed points / decay          [0.5, 15.0]
        Regime 2: curve / limit cycles         [15.0, 24.0]
        Regime 3: chaos / repulsor             [24.0, 50.0]
    ε ~ Uniform(0.0, 1.5)
    """
    sigma   = rng.uniform(1.0, 20.0)
    beta    = rng.uniform(0.5,  5.0)
    regime  = rng.integers(0, 3)
    if regime == 0:
        rho = rng.uniform(0.5, 15.0)
    elif regime == 1:
        rho = rng.uniform(15.0, 24.0)
    else:
        rho = rng.uniform(24.0, 50.0)
    epsilon = rng.uniform(0.0, 1.5)
    return sigma, rho, beta, epsilon


def simulate_lorenz(
    sigma: float, rho: float, beta: float, epsilon: float,
    T: float, dt: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euler-Maruyama integration of the stochastic Lorenz system."""
    N       = int(T / dt)
    x, y, z = np.zeros(N), np.zeros(N), np.zeros(N)
    x[0], y[0], z[0] = rng.standard_normal(3)
    sqrt_dt = np.sqrt(dt)

    for i in range(N - 1):
        noise    = epsilon * sqrt_dt * rng.standard_normal(3)
        x[i+1]   = x[i] + sigma * (y[i] - x[i]) * dt + noise[0]
        y[i+1]   = y[i] + (x[i] * (rho - z[i]) - y[i]) * dt + noise[1]
        z[i+1]   = z[i] + (x[i] * y[i] - beta * z[i]) * dt + noise[2]

    return x, y, z


def label_from_rho(rho: float) -> int:
    """Return 0 (fixed-point), 1 (curve), or 2 (chaotic/repulsor)."""
    if rho <= 15.0:
        return 0
    if rho <= 24.0:
        return 1
    return 2


def normalize_traj(traj: np.ndarray) -> np.ndarray:
    lo = traj.min(axis=0)
    hi = traj.max(axis=0)
    return (traj - lo) / np.maximum(hi - lo, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(args.seed)
    dataset: list[dict] = []

    print(f"[01_generate_data] Generating up to {args.n_samples} samples")

    for i in range(args.n_samples):
        sigma, rho, beta, epsilon = sample_params(rng_master)
        label = label_from_rho(rho)

        # Each trajectory gets its own child RNG for reproducibility
        child_seed = int(rng_master.integers(0, 2**31))
        x, y, z    = simulate_lorenz(
            sigma, rho, beta, epsilon, args.T, args.dt,
            rng=np.random.default_rng(child_seed),
        )

        # Discard transient
        x, y, z = x[args.cut:], y[args.cut:], z[args.cut:]

        traj = normalize_traj(np.stack([x, y], axis=1))

        dataset.append({
            "trajectory": traj.astype(np.float32),
            "params":     np.array([sigma, rho, beta, epsilon], dtype=np.float32),
            "label":      label,
        })

    print(f"[01_generate_data] Kept {len(dataset)} labelled samples "
          f"(fixed-point: {sum(1 for d in dataset if d['label']==0)}, "
          f"curve: {sum(1 for d in dataset if d['label']==1)}, "
          f"chaotic/repulsor: {sum(1 for d in dataset if d['label']==2)})")

    # ── Save dataset ──────────────────────────────────────────────────────────
    npz_path = Path(args.out)
    np.savez(
        npz_path,
        trajectories=np.array([d["trajectory"] for d in dataset], dtype=object),
        params=np.array([d["params"] for d in dataset]),
        labels=np.array([d["label"]  for d in dataset]),
    )
    print(f"[01_generate_data] Dataset saved {npz_path}")

    # ── Diversity visualisation ───────────────────────────────────────────────
    plot_generated_diversity(dataset, out_dir / "generated_diversity.png")
    print("[01_generate_data] Done.")


if __name__ == "__main__":
    main()
