"""
src/utils/plotting.py
=====================
Shared visualisation helpers used by the evaluation script.

Public API
----------
plot_posterior_vs_prior(posterior_chain, observed_params, prior_bounds, out_path)
plot_future_paths(observed, futures, out_path)
plot_training_curves(history, out_path)
plot_generated_diversity(dataset, out_path)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Posterior evaluation plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_posterior_vs_prior(
    posterior_chain: np.ndarray,    # (n_samples, 4)
    observed_params: np.ndarray,    # (4,)  ground-truth θ*
    prior_bounds:    np.ndarray,    # (4, 2)  [[lo, hi], ...]
    out_path:        Path | str,
) -> None:
    """
    Four-panel figure: one panel per parameter (σ, ρ, β, ε).

    Each panel:
      grey  histogram  — prior  (uniform samples for display)
      colour histogram — posterior MCMC samples
      bold black line  — ground-truth parameter value θ*
    """
    out_path = Path(out_path)
    n_prior  = 5000
    rng_vis  = np.random.default_rng(0)
    prior_samples = np.column_stack([
        rng_vis.uniform(lo, hi, n_prior)
        for lo, hi in prior_bounds
    ])

    param_labels = [
        r"$\sigma$  (Prandtl)",
        r"$\rho$  (Rayleigh)",
        r"$\beta$  (geometric)",
        r"$\varepsilon$  (noise scale)",
    ]
    colours = ["#3b82d4", "#e05c2e", "#2ca02c", "#9467bd"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(
        "Posterior vs. Prior  —  RWMH MCMC inference\n"
        "Bold black line = ground-truth  θ*",
        fontsize=13, fontweight="bold",
    )

    for k, (ax, label, colour) in enumerate(zip(axes, param_labels, colours)):
        prior_v     = prior_samples[:, k]
        post_v      = posterior_chain[:, k]
        gt          = float(observed_params[k])

        lo_plot = min(prior_v.min(), post_v.min()) - 0.05
        hi_plot = max(prior_v.max(), post_v.max()) + 0.05
        bins    = np.linspace(lo_plot, hi_plot, 35)

        ax.hist(prior_v, bins=bins, density=True, alpha=0.40,
                color="grey",  label="Prior",     edgecolor="white", lw=0.4)
        ax.hist(post_v,  bins=bins, density=True, alpha=0.75,
                color=colour,  label="Posterior", edgecolor="white", lw=0.4)
        ax.axvline(gt, color="black", linewidth=3.0, linestyle="-",
                   label=f"True: {gt:.2f}", zorder=5)

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, lw=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] posterior_vs_prior  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Future trajectory fan plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_future_paths(
    observed: np.ndarray,           # (T, 2)
    futures:  list[np.ndarray],     # list of (T', 2)
    out_path: Path | str,
) -> None:
    """
    Phase-plane overlay: observed track (blue) + posterior-sampled futures (green).
    """
    out_path = Path(out_path)

    def _norm(path: np.ndarray) -> np.ndarray:
        lo = path.min(axis=0, keepdims=True)
        hi = path.max(axis=0, keepdims=True)
        return (path - lo) / np.maximum(hi - lo, 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    for f in futures:
        n = _norm(f)
        ax.plot(n[:, 0], n[:, 1], lw=0.8, alpha=0.35, color="#2ca02c")
    obs = _norm(observed)
    ax.plot(obs[:, 0], obs[:, 1], lw=2.0, alpha=0.9, color="#1f77b4",
            label="Observed track")
    ax.plot(obs[0,  0], obs[0,  1], "o", ms=7, color="#1f77b4")
    ax.plot(obs[-1, 0], obs[-1, 1], "s", ms=7, color="#1f77b4")

    # Legend proxies
    ax.plot([], [], lw=1.5, color="#2ca02c", label=f"Posterior paths (n={len(futures)})")
    ax.set_xlabel("x (normalised)", fontsize=10)
    ax.set_ylabel("y (normalised)", fontsize=10)
    ax.set_title("Posterior-predictive trajectories\n"
                 "(parameters drawn from RWMH chain)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] future_paths  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Training-curve plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history:  list[dict],
    out_path: Path | str,
) -> None:
    """
    Two-panel figure: loss curve and accuracy curve across epochs.
    Works for both the logistic-regression and PyTorch training loops.
    """
    out_path = Path(out_path)
    epochs   = [r["epoch"] for r in history]
    tr_loss  = [r["train_loss"] for r in history]
    va_loss  = [r["val_loss"]   for r in history]
    has_acc  = "train_accuracy" in history[0] and history[0]["train_accuracy"] != ""

    ncols = 2 if has_acc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    axes[0].plot(epochs, tr_loss, label="train", color="#1f77b4")
    axes[0].plot(epochs, va_loss, label="val",   color="#d62728")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Training loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if has_acc:
        tr_acc = [r["train_accuracy"] for r in history]
        va_acc = [r["val_accuracy"]   for r in history]
        axes[1].plot(epochs, tr_acc, label="train", color="#1f77b4")
        axes[1].plot(epochs, va_acc, label="val",   color="#d62728")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
        axes[1].set_title("Training accuracy"); axes[1].legend()
        axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] training_curves  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset diversity visualisation
# ─────────────────────────────────────────────────────────────────────────────

def classify_rho_regime(rho: float) -> str:
    """Classify a Rayleigh number into one of the three requested regimes."""
    rho = float(rho)
    if rho <= 15.0:
        return "fixed-point"
    if rho <= 24.0:
        return "curve"
    return "chaotic/repulsor"


def plot_generated_diversity(
    dataset:  list[dict],   # list of {"trajectory", "params", "label"}
    out_path: Path | str,
) -> None:
    """
    Summary figure for the full generated dataset:
      - Upper row: one subplot for each rho regime
      - Bottom row: histograms of σ, ρ, β, ε

    Regime split:
      • ρ ≤ 15  -> fixed-point
      • 15 < ρ ≤ 24  -> curve
      • ρ > 24  -> chaotic / repulsor
    """
    out_path = Path(out_path)

    regime_groups: dict[str, list[dict]] = {
        "fixed-point": [],
        "curve": [],
        "chaotic/repulsor": [],
    }
    for d in dataset:
        params = np.asarray(d["params"], dtype=float)
        rho = float(params[1]) if params.size > 1 else 0.0
        regime_groups[classify_rho_regime(rho)].append(d)

    n_total = len(dataset)
    n_fp = len(regime_groups["fixed-point"])
    n_curve = len(regime_groups["curve"])
    n_ch = len(regime_groups["chaotic/repulsor"])

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"Generated Lorenz SDE Dataset — {n_total} trajectories  "
        f"({n_fp} fixed-point  |  {n_curve} curve  |  {n_ch} chaotic/repulsor)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.35, height_ratios=[1.6, 1])

    ax_fp    = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_ch    = fig.add_subplot(gs[0, 2])
    # ax_blank = fig.add_subplot(gs[0, 3])
    # ax_blank.axis("off")
    ax_sigma = fig.add_subplot(gs[1, 0])
    ax_rho   = fig.add_subplot(gs[1, 1])
    ax_beta  = fig.add_subplot(gs[1, 2])
    ax_eps   = fig.add_subplot(gs[1, 3])

    for ax, subset, cmap_name, title in [
        (ax_fp, regime_groups["fixed-point"], "Blues",
         "Fixed-point trajectories  (ρ ≤ 15)"),
        (ax_curve, regime_groups["curve"], "Greens",
         "Curve trajectories  (15 < ρ ≤ 24)"),
        (ax_ch, regime_groups["chaotic/repulsor"], "Reds",
         "Chaotic / repulsor trajectories  (ρ > 24)"),
    ]:
        n_sub = len(subset)
        cmap = plt.get_cmap(cmap_name)
        for k, d in enumerate(subset):
            col = cmap(0.2 + 0.7 * k / max(n_sub - 1, 1))
            traj = np.asarray(d["trajectory"], dtype=float)
            ax.plot(traj[:, 0], traj[:, 1],
                    lw=0.25, alpha=0.35, color=col, rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (normalised)", fontsize=9)
        ax.set_ylabel("y (normalised)", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.text(0.02, 0.97, f"n = {n_sub}", transform=ax.transAxes,
                va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    all_params = np.array([d["params"] for d in dataset])
    for ax, vals, xlabel, colour in [
        (ax_sigma, all_params[:, 0], "σ  (Prandtl)", "C0"),
        (ax_rho,   all_params[:, 1], "ρ  (Rayleigh)", "C1"),
        (ax_beta,  all_params[:, 2], "β  (geometric)", "C2"),
        (ax_eps,   all_params[:, 3], "ε  (noise scale)", "C3"),
    ]:
        ax.hist(vals, bins=30, color=colour, alpha=0.78, edgecolor="white", lw=0.4)
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("count", fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.5)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] generated_diversity {out_path}")
