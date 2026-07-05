"""
src/sde/lorenz_sde.py
=====================
Stochastic Lorenz system — two backends:

  1. ``LorenzSDE``          — torchsde-based (fast, for dataset generation)
  2. ``simulate_lorenz_np`` — NumPy Euler-Maruyama (dependency-free, used inside MCMC)

Public API
----------
LorenzSDE(T, dt, seed)
    .simulate(sigma, rho, beta, noise_scale, y0)  -> np.ndarray (steps, 3)
    .run_grid(...)
    .run_sensitivity(...)

simulate_lorenz_np(sigma, rho, beta, epsilon, steps, dt, y0, seed)
    -> np.ndarray (steps, 2)   [x, y only — matches dataset format]

PARAM_BOUNDS : dict  — prior ranges for each parameter
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsde
from torchsde import BrownianInterval

sys.setrecursionlimit(10000)

# ─────────────────────────────────────────────────────────────────────────────
# Prior parameter bounds (used by both dataset_gen and MCMC)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_NAMES: tuple[str, ...] = ("sigma", "rho", "beta", "epsilon")

# Bounds for the *chaotic* regime only (label=1)
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "sigma":   (1.0,  20.0),
    "rho":     (25.0, 50.0),
    "beta":    (0.5,   5.0),
    "epsilon": (0.0,   1.5),
}

PRIOR_BOUNDS_ARRAY = np.array([
    PARAM_BOUNDS["sigma"],
    PARAM_BOUNDS["rho"],
    PARAM_BOUNDS["beta"],
    PARAM_BOUNDS["epsilon"],
], dtype=np.float64)   # shape (4, 2) — columns are [lo, hi]


# ─────────────────────────────────────────────────────────────────────────────
# NumPy Euler-Maruyama simulator  (no torch dependency, used in MCMC loop)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_lorenz_np(
    sigma:   float,
    rho:     float,
    beta:    float,
    epsilon: float,
    steps:   int   = 400,
    dt:      float = 0.01,
    y0:      tuple = (1.0, 1.0, 1.0),
    seed:    int   = 0,
) -> np.ndarray:
    """
    Euler-Maruyama integration of the stochastic Lorenz system.

    Returns
    -------
    np.ndarray shape (steps, 2)  — columns are x, y  (z dropped for 2-D matching)
    """
    rng     = np.random.default_rng(seed)
    xyz     = np.zeros((steps, 3), dtype=np.float64)
    xyz[0]  = np.asarray(y0, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)

    for i in range(steps - 1):
        x, y, z = xyz[i]
        drift = np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ])
        noise     = epsilon * sqrt_dt * rng.standard_normal(3)
        xyz[i+1]  = np.clip(xyz[i] + drift * dt + noise, -1e4, 1e4)

    return xyz[:, :2]


# ─────────────────────────────────────────────────────────────────────────────
# Internal torchsde SDE class
# ─────────────────────────────────────────────────────────────────────────────

class _StochasticLorenz(torch.nn.Module):
    """Diagonal-noise Itô SDE for the Lorenz system (torchsde backend)."""

    noise_type = "diagonal"
    sde_type   = "ito"

    def __init__(self, sigma: float, rho: float, beta: float, noise_scale: float):
        super().__init__()
        self.sigma       = float(sigma)
        self.rho         = float(rho)
        self.beta        = float(beta)
        self.noise_scale = float(noise_scale)

    def f(self, t, state):
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.stack([dx, dy, dz], dim=1)

    def g(self, t, state):
        ns = self.noise_scale
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        return torch.stack([ns * x, ns * y, ns * z], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slug(**kwargs) -> str:
    return "__".join(f"{k}={v}" for k, v in kwargs.items())


def _save_params(folder: Path, params: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "params.json", "w") as fh:
        json.dump(params, fh, indent=2)


def _save_csv(folder: Path, t: np.ndarray, traj: np.ndarray) -> None:
    with open(folder / "trajectory.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "x", "y", "z"])
        for i, ti in enumerate(t):
            writer.writerow([ti, traj[i, 0], traj[i, 1], traj[i, 2]])


def _save_plot(folder: Path, traj: np.ndarray, axes: tuple[int, int] = (0, 1)) -> None:
    a, b   = axes
    labels = ["x", "y", "z"]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(traj[:, a], traj[:, b], lw=0.6, color="black")
    ax.axis("off")
    ax.set_aspect("equal", "box")
    plt.tight_layout(pad=0)
    fig.savefig(folder / f"phase_{labels[a]}{labels[b]}.png",
                dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _get_cmap(name: str, n: int):
    try:
        return matplotlib.colormaps[name].resampled(n)
    except AttributeError:
        return plt.cm.get_cmap(name, n)


# ─────────────────────────────────────────────────────────────────────────────
# Public torchsde-based class
# ─────────────────────────────────────────────────────────────────────────────

class LorenzSDE:
    """
    torchsde wrapper for the stochastic Lorenz system.

    Parameters
    ----------
    T    : float — total simulation time
    dt   : float — integration step size
    seed : int   — base seed (each simulate() call uses seed + offset)
    """

    def __init__(self, T: float = 5.0, dt: float = 0.001, seed: int = 0):
        self.T    = float(T)
        self.dt   = float(dt)
        self.seed = seed

        t         = np.arange(0, T + dt, dt)
        self._t   = t
        self._ts  = torch.tensor(t, dtype=torch.float32)

    # ── core simulation ───────────────────────────────────────────────────────

    def simulate(
        self,
        sigma:       float,
        rho:         float,
        beta:        float,
        noise_scale: float,
        y0:          Sequence[float] = (1.0, 1.0, 1.0),
        seed_offset: int = 0,
    ) -> np.ndarray:
        """Returns np.ndarray shape (n_steps, 3) — columns x, y, z."""
        torch.manual_seed(self.seed + seed_offset)
        sde  = _StochasticLorenz(sigma, rho, beta, noise_scale)
        y0t  = torch.tensor([list(y0)], dtype=torch.float32)
        bm   = BrownianInterval(
            t0=self._ts[0], t1=self._ts[-1],
            size=(1, 3), device=y0t.device,
            levy_area_approximation="none",
            entropy=self.seed + seed_offset,
        )
        ys = torchsde.sdeint(sde, y0t, self._ts, bm=bm, method="euler")
        return ys.squeeze(1).detach().numpy()

    # ── grid run ──────────────────────────────────────────────────────────────

    def run_grid(
        self,
        sigmas:       Sequence[float],
        rhos:         Sequence[float],
        betas:        Sequence[float],
        noise_scales: Sequence[float],
        y0s:          Sequence[Sequence[float]],
        out_dir:      str | Path = "lorenz_output",
        plot_axes:    tuple[int, int] = (0, 1),
    ) -> list[dict]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        combos = list(itertools.product(sigmas, rhos, betas, noise_scales, y0s))
        print(f"[LorenzSDE] Running {len(combos)} grid combinations …")
        results: list[dict] = []

        for idx, (sigma, rho, beta, ns, y0) in enumerate(combos, 1):
            y0_str = f"({','.join(str(v) for v in y0)})"
            folder = out_dir / _slug(sigma=sigma, rho=rho, beta=beta, noise=ns, y0=y0_str)
            folder.mkdir(parents=True, exist_ok=True)

            params = dict(sigma=sigma, rho=rho, beta=beta, noise_scale=ns, y0=list(y0))
            _save_params(folder, params)
            traj = self.simulate(sigma, rho, beta, ns, y0, seed_offset=idx)
            _save_csv(folder, self._t, traj)
            _save_plot(folder, traj, plot_axes)
            results.append({**params, "folder": str(folder)})
            print(f"  [{idx}/{len(combos)}] {folder.name}")

        print(f"[LorenzSDE] Done. Output in '{out_dir}'")
        return results

    # ── sensitivity sweep ─────────────────────────────────────────────────────

    def run_sensitivity(
        self,
        base_params: dict,
        ranges:      dict[str, Sequence[float]],
        out_dir:     str | Path = "lorenz_sensitivity",
        plot_axes:   tuple[int, int] = (0, 1),
    ) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bp     = base_params
        a, b   = plot_axes
        labels = ["x", "y", "z"]

        for param_name, values in ranges.items():
            n_vals    = len(values)
            param_dir = out_dir / param_name
            param_dir.mkdir(parents=True, exist_ok=True)
            cmap      = _get_cmap("plasma", n_vals)

            ov_fig, ov_ax = plt.subplots(figsize=(6, 6))
            ov_ax.set_title(f"Sensitivity: {param_name}", fontsize=13)
            ov_ax.set_xlabel(labels[a]); ov_ax.set_ylabel(labels[b])
            ov_ax.grid(True, alpha=0.3)

            ts_fig, ts_axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            ts_fig.suptitle(f"Time series — sweeping '{param_name}'", fontsize=13)

            print(f"[sensitivity] Sweeping '{param_name}' over {values}")

            for ci, val in enumerate(values):
                p    = {**bp, param_name: val}
                traj = self.simulate(p["sigma"], p["rho"], p["beta"],
                                      p["noise_scale"], p["y0"], seed_offset=ci)
                sub  = param_dir / f"{param_name}={val}"
                sub.mkdir(parents=True, exist_ok=True)
                _save_params(sub, p); _save_csv(sub, self._t, traj)
                _save_plot(sub, traj, plot_axes)

                ov_ax.plot(traj[:, a], traj[:, b], lw=0.5,
                           color=cmap(ci), label=f"{param_name}={val}")

                for dim, ax in enumerate(ts_axes):
                    ax.plot(self._t, traj[:, dim], lw=0.5,
                            color=cmap(ci), label=str(val) if dim == 0 else "")
                    ax.set_ylabel(labels[dim]); ax.grid(True, alpha=0.3)

                print(f"  [{ci+1}/{n_vals}] {param_name}={val}")

            ov_ax.legend(fontsize=7, loc="best")
            ov_fig.tight_layout()
            ov_fig.savefig(param_dir / "overview.png", dpi=150, bbox_inches="tight")
            plt.close(ov_fig)

            ts_axes[-1].set_xlabel("time")
            ts_axes[0].legend(title=param_name, fontsize=7,
                               ncol=min(n_vals, 6), loc="upper right")
            ts_fig.tight_layout()
            ts_fig.savefig(param_dir / "timeseries_overview.png",
                           dpi=150, bbox_inches="tight")
            plt.close(ts_fig)
            print(f"  → saved to '{param_dir}'")

        print(f"[LorenzSDE] Sensitivity done. Output in '{out_dir}'")
