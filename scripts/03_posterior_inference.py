"""
scripts/03_posterior_inference.py
==================================
Part 3 of 4 — Posterior Inference via RWMH MCMC

Uses the trained contrastive ratio estimator as a likelihood-ratio surrogate
and runs Random-Walk Metropolis-Hastings to draw posterior samples:

    p(θ | x_obs)  ∝  exp( logit(θ, x_obs) )  ·  p(θ)

where logit(θ, x_obs) is the classifier's output and p(θ) is the uniform prior.

The posterior chain is saved to outputs/posterior_chain.npz so the evaluation
script can load it without re-running MCMC.

Usage:
    # With the full neural ratio estimator (requires best.pt from step 2):
    python scripts/03_posterior_inference.py --mode neural --checkpoint outputs/training/ratio/best.pt

    # Lightweight NumPy logistic-regression baseline (no PyTorch checkpoint needed):
    python scripts/03_posterior_inference.py --mode logreg

Outputs:
    outputs/inference/posterior_chain.npz  — MCMC samples + metadata
    outputs/inference/mcmc_trace.png       — chain trace (visual convergence check)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sde.lorenz_sde import PRIOR_BOUNDS_ARRAY, PARAM_NAMES
from src.inference.mcmc import rwmh_mcmc, make_log_target
from src.utils.features import (
    summarize_trajectory,
    transform_params,
    standardize_fit,
    standardize,
    pair_design_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RWMH MCMC posterior inference.")
    p.add_argument("--dataset",     default=str(ROOT / "data" / "data.npz"))
    p.add_argument("--mode",
                   choices=["logreg", "neural"],
                   default="logreg",
                   help="logreg: NumPy logistic-regression baseline (no GPU needed)\n"
                        "neural: load PyTorch best.pt from step 2")
    p.add_argument("--checkpoint",     default=str(ROOT / "outputs/training/ratio/best.pt"),
                   help="Path to best.pt (required for --mode neural)")
    p.add_argument("--observed-index", type=int, default=None,
                   help="Index of the observed trajectory (default: first val sample)")
    p.add_argument("--mcmc-steps",     type=int,   default=6000)
    p.add_argument("--mcmc-burnin",    type=int,   default=1000)
    p.add_argument("--mcmc-step-size", type=float, default=0.5,
                   help="Gaussian proposal std per dimension")
    p.add_argument("--max-points",     type=int,   default=512)
    p.add_argument("--epochs",         type=int,   default=400,
                   help="Epochs for logistic-regression training (logreg mode)")
    p.add_argument("--lr",             type=float, default=0.15)
    p.add_argument("--l2",             type=float, default=1e-3)
    p.add_argument("--seed",           type=int,   default=7)
    p.add_argument("--out-dir",        default=str(ROOT / "outputs/inference"))
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(path: str | Path):
    raw = np.load(path, allow_pickle=True)
    trajs = raw["trajectories"]
    if trajs.dtype == object:
        trajs = np.stack([np.asarray(t, dtype=np.float32) for t in trajs])
    else:
        trajs = trajs.astype(np.float32)
    return trajs, raw["params"].astype(np.float32), raw["labels"].astype(np.int64)


def split_indices(n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return idx[:int(0.8*n)], idx[int(0.8*n):int(0.9*n)], idx[int(0.9*n):]


# ─────────────────────────────────────────────────────────────────────────────
# Logistic-regression contrastive classifier (logreg mode)
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def train_logreg(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_va: np.ndarray, y_va: np.ndarray,
    epochs: int, lr: float, l2: float,
) -> tuple[np.ndarray, list[dict]]:
    w = np.zeros(x_tr.shape[1] + 1, dtype=np.float64)
    history = []
    for epoch in range(1, epochs + 1):
        prob = sigmoid(x_tr @ w[:-1] + w[-1])
        err  = prob - y_tr
        w[:-1] -= lr * ((x_tr.T @ err) / len(y_tr) + l2 * w[:-1])
        w[-1]  -= lr * err.mean()

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            vp  = sigmoid(x_va @ w[:-1] + w[-1])
            eps = 1e-8
            history.append({
                "epoch":          float(epoch),
                "train_loss":     float(-np.mean(y_tr*np.log(prob+eps)+(1-y_tr)*np.log(1-prob+eps))),
                "train_accuracy": float(np.mean((prob>=0.5)==y_tr)),
                "val_loss":       float(-np.mean(y_va*np.log(vp+eps)+(1-y_va)*np.log(1-vp+eps))),
                "val_accuracy":   float(np.mean((vp>=0.5)==y_va)),
            })
    return w, history


def make_logreg_log_target(
    trajs: np.ndarray, params: np.ndarray,
    train_idx: np.ndarray, val_idx: np.ndarray,
    observed_index: int, max_points: int,
    epochs: int, lr: float, l2: float,
    rng: np.random.Generator,
) -> tuple:
    """Train logistic regression and return (log_target_fn, observed_params, history)."""
    traj_feats  = np.stack([summarize_trajectory(t, max_points) for t in trajs])
    param_feats = transform_params(params)

    # Build contrastive pairs
    def make_pairs(idx):
        pf = param_feats[idx]; tf = traj_feats[idx]
        shuffled = idx.copy()
        while True:
            rng.shuffle(shuffled)
            if np.all(shuffled != idx):
                break
        pos = pair_design_matrix(pf, tf)
        neg = pair_design_matrix(pf, traj_feats[shuffled])
        x   = np.vstack([pos, neg])
        y   = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
        order = rng.permutation(len(y))
        return x[order], y[order]

    x_tr_raw, y_tr = make_pairs(train_idx)
    x_va_raw, y_va = make_pairs(val_idx)
    x_mean, x_std  = standardize_fit(x_tr_raw)
    x_tr = standardize(x_tr_raw, x_mean, x_std)
    x_va = standardize(x_va_raw, x_mean, x_std)

    weights, history = train_logreg(x_tr, y_tr, x_va, y_va, epochs, lr, l2)
    print(f"  [logreg] val_acc = {history[-1]['val_accuracy']:.3f}")

    obs_feat  = traj_feats[observed_index].astype(np.float64)
    log_fn    = make_log_target(weights, obs_feat, x_mean, x_std, PRIOR_BOUNDS_ARRAY)
    obs_params = params[observed_index].astype(np.float64)
    return log_fn, obs_params, history


# ─────────────────────────────────────────────────────────────────────────────
# Neural ratio estimator mode
# ─────────────────────────────────────────────────────────────────────────────

def make_neural_log_target(
    checkpoint_path: str,
    obs_traj: np.ndarray,          # (T, 2) raw observed trajectory
    max_points: int,
    prior_bounds: np.ndarray,
) -> tuple:
    """
    Load best.pt and return a log_target closure that calls the neural model.
    """
    import torch
    from src.models.lorenz_models import LorenzRatioEstimator
    from src.utils.features import summarize_trajectory

    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LorenzRatioEstimator()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Observed trajectory feature tensor (fixed)
    traj_np = obs_traj.copy().astype(np.float32)
    if len(traj_np) > max_points:
        pick   = np.linspace(0, len(traj_np)-1, max_points).astype(np.int64)
        traj_np = traj_np[pick]
    # Normalise to [-1, 1]
    lo = traj_np.min(axis=0, keepdims=True)
    hi = traj_np.max(axis=0, keepdims=True)
    traj_np = (traj_np - lo) / np.maximum(hi - lo, 1e-8) * 2.0 - 1.0
    traj_t  = torch.tensor(traj_np.T[None, :, :], dtype=torch.float32)  # (1, 2, T)

    # Param scaler: we need min/max from the dataset — approximate using prior bounds
    p_min = prior_bounds[:, 0].astype(np.float32)
    p_max = prior_bounds[:, 1].astype(np.float32)

    def _scale_params(theta_raw: np.ndarray) -> np.ndarray:
        return (theta_raw.astype(np.float32) - p_min) / np.maximum(p_max - p_min, 1e-8)

    def log_target(theta_raw: np.ndarray) -> float:
        lo_b, hi_b = prior_bounds[:, 0], prior_bounds[:, 1]
        if np.any(theta_raw < lo_b) or np.any(theta_raw > hi_b):
            return -np.inf
        theta_sc = _scale_params(theta_raw)
        theta_t  = torch.tensor(theta_sc[None, :], dtype=torch.float32)
        with torch.no_grad():
            logit = model(traj_t, theta_t).item()
        return float(logit)

    return log_target


# ─────────────────────────────────────────────────────────────────────────────
# MCMC trace plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_trace(chain: np.ndarray, burnin: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim    = chain.shape[1]
    labels = list(PARAM_NAMES)
    fig, axes = plt.subplots(dim, 1, figsize=(12, 3 * dim), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(chain[:burnin, k], color="grey",  lw=0.5, alpha=0.7, label="burn-in")
        ax.plot(range(burnin, len(chain)),
                chain[burnin:, k], color="#1f77b4", lw=0.5, label="posterior")
        ax.set_ylabel(labels[k], fontsize=9)
        ax.axvline(burnin, color="red", lw=1.0, linestyle="--")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("MCMC step", fontsize=9)
    fig.suptitle("RWMH MCMC trace (red dashed = end of burn-in)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] mcmc_trace  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    rng     = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[03_posterior_inference] mode={args.mode}  "
          f"steps={args.mcmc_steps}  burnin={args.mcmc_burnin}")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    trajs, params, labels = load_npz(args.dataset)
    train_idx, val_idx, _ = split_indices(len(params), args.seed)
    observed_index = args.observed_index if args.observed_index is not None else int(val_idx[0])

    # ── 2. Build log-target ───────────────────────────────────────────────────
    classifier_history = None

    if args.mode == "logreg":
        print("[03_posterior_inference] Training logistic-regression classifier ")
        log_fn, observed_params, classifier_history = make_logreg_log_target(
            trajs, params, train_idx, val_idx,
            observed_index, args.max_points,
            args.epochs, args.lr, args.l2, rng,
        )
    else:  # neural
        print(f"[03_posterior_inference] Loading neural checkpoint: {args.checkpoint}")
        log_fn = make_neural_log_target(
            args.checkpoint, trajs[observed_index],
            args.max_points, PRIOR_BOUNDS_ARRAY,
        )
        observed_params = params[observed_index].astype(np.float64)

    # ── 3. Initialise MCMC chain ──────────────────────────────────────────────
    # Start near a dataset sample from the training set to avoid a cold start
    theta_init = params[train_idx[0]].astype(np.float64).copy()
    theta_init = np.clip(theta_init, PRIOR_BOUNDS_ARRAY[:, 0], PRIOR_BOUNDS_ARRAY[:, 1])

    print(f"[03_posterior_inference] Running RWMH (step_size={args.mcmc_step_size}) ")
    chain, accept_rate = rwmh_mcmc(
        log_target_fn=log_fn,
        theta_init=theta_init,
        n_steps=args.mcmc_steps,
        step_size=args.mcmc_step_size,
        rng=np.random.default_rng(args.seed + 1),
    )
    posterior = chain[args.mcmc_burnin:]
    print(f"  acceptance rate: {accept_rate:.3f}   posterior samples: {len(posterior)}")

    # ── 4. Save chain ─────────────────────────────────────────────────────────
    npz_path = out_dir / "posterior_chain.npz"
    np.savez(
        npz_path,
        chain=chain,
        posterior=posterior,
        observed_params=observed_params,
        observed_index=np.array(observed_index),
        prior_bounds=PRIOR_BOUNDS_ARRAY,
    )
    print(f"[03_posterior_inference] Chain saved  {npz_path}")

    # ── 5. Trace plot ─────────────────────────────────────────────────────────
    plot_trace(chain, args.mcmc_burnin, out_dir / "mcmc_trace.png")

    # ── 6. Summary JSON ───────────────────────────────────────────────────────
    summary = {
        "mode":             args.mode,
        "observed_index":   observed_index,
        "observed_params":  {n: float(v) for n, v in zip(PARAM_NAMES, observed_params)},
        "mcmc_steps":       args.mcmc_steps,
        "mcmc_burnin":      args.mcmc_burnin,
        "mcmc_step_size":   args.mcmc_step_size,
        "mcmc_accept_rate": float(accept_rate),
        "posterior_mean":   {n: float(v) for n, v in zip(PARAM_NAMES, posterior.mean(axis=0))},
        "posterior_std":    {n: float(v) for n, v in zip(PARAM_NAMES, posterior.std(axis=0))},
    }
    if classifier_history:
        summary["classifier_final_val_accuracy"] = classifier_history[-1]["val_accuracy"]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[03_posterior_inference] Done.")


if __name__ == "__main__":
    main()
