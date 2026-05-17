"""
lorenz_sde.py
=============
Production module for simulating the stochastic Lorenz system.

Public API
----------
LorenzSDE(T, dt)
    .simulate(sigma, rho, beta, noise_scale, y0)  -> np.ndarray  shape (steps, 3)
    .run_grid(sigmas, rhos, betas, noise_scales, y0s, out_dir)
    .run_sensitivity(base_params, ranges, out_dir)
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


# ──────────────────────────────────────────────────────────────────────────────
# Internal SDE class
# ──────────────────────────────────────────────────────────────────────────────

class _StochasticLorenz(torch.nn.Module):
    """Diagonal-noise Itô SDE for the Lorenz system."""

    noise_type = "diagonal"
    sde_type   = "ito"

    def __init__(self, sigma: float, rho: float, beta: float, noise_scale: float):
        super().__init__()
        self.sigma       = float(sigma)
        self.rho         = float(rho)
        self.beta        = float(beta)
        self.noise_scale = float(noise_scale)

    def f(self, t, state):                      # drift
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.stack([dx, dy, dz], dim=1)

    def g(self, t, state):                      # diffusion
        ns = self.noise_scale
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        return torch.stack([ns * x, ns * y, ns * z], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _slug(**kwargs) -> str:
    """Build a filesystem-safe name from parameter key=value pairs."""
    return "__".join(f"{k}={v}" for k, v in kwargs.items())


def _save_params(folder: Path, params: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "params.json", "w") as fh:
        json.dump(params, fh, indent=2)


def _save_csv(folder: Path, t: np.ndarray, traj: np.ndarray) -> None:
    """Save trajectory as CSV: time, x, y, z."""
    path = folder / "trajectory.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "x", "y", "z"])
        for i, ti in enumerate(t):
            writer.writerow([ti, traj[i, 0], traj[i, 1], traj[i, 2]])


def _save_plot(folder: Path, traj: np.ndarray, axes: tuple[int, int] = (0, 1)) -> None:
    """Save a clean phase-plane PNG (no axes, no grid)."""
    a, b = axes
    labels = ["x", "y", "z"]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(traj[:, a], traj[:, b], lw=0.6, color="black")
    ax.axis("off")
    ax.set_aspect("equal", "box")
    plt.tight_layout(pad=0)
    fig.savefig(
        folder / f"phase_{labels[a]}{labels[b]}.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Public module class
# ──────────────────────────────────────────────────────────────────────────────

class LorenzSDE:
    """
    Parameters
    ----------
    T  : float  – total simulation time
    dt : float  – step size
    seed : int  – global RNG seed (set once at construction)
    """

    def __init__(self, T: float = 5.0, dt: float = 0.001, seed: int = 0):
        self.T    = float(T)
        self.dt   = float(dt)
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        t  = np.arange(0, T + dt, dt)
        self._t  = t
        self._ts = torch.tensor(t, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Core simulation
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        sigma:       float,
        rho:         float,
        beta:        float,
        noise_scale: float,
        y0:          Sequence[float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Simulate one trajectory.

        Returns
        -------
        np.ndarray of shape (n_steps, 3)  – columns are x, y, z
        """
        sde  = _StochasticLorenz(sigma, rho, beta, noise_scale)
        y0t  = torch.tensor([list(y0)], dtype=torch.float32)

        bm = BrownianInterval(
            t0=self._ts[0],
            t1=self._ts[-1],
            size=(1, 3),
            device=y0t.device,
            levy_area_approximation="none",
        )

        ys = torchsde.sdeint(sde, y0t, self._ts, bm=bm, method="euler")
        return ys.squeeze(1).detach().numpy()   # (n_steps, 3)

    # ------------------------------------------------------------------ #
    # Grid run over all parameter combinations
    # ------------------------------------------------------------------ #

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
        """
        Simulate every combination of the supplied parameter arrays.

        Directory layout
        ----------------
        out_dir/
          sigma=X__rho=Y__beta=Z__noise=N__y0=..../
            params.json
            trajectory.csv
            phase_xy.png

        Returns
        -------
        List of result dicts (params + output paths).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        combos = list(itertools.product(sigmas, rhos, betas, noise_scales, y0s))
        total  = len(combos)
        results: list[dict] = []

        print(f"[LorenzSDE] Running {total} combinations …")

        for idx, (sigma, rho, beta, ns, y0) in enumerate(combos, 1):
            y0_str = f"({','.join(str(v) for v in y0)})"
            name   = _slug(sigma=sigma, rho=rho, beta=beta, noise=ns, y0=y0_str)
            folder = out_dir / name
            folder.mkdir(parents=True, exist_ok=True)

            params = dict(sigma=sigma, rho=rho, beta=beta, noise_scale=ns, y0=list(y0))
            _save_params(folder, params)

            traj = self.simulate(sigma, rho, beta, ns, y0)

            _save_csv(folder, self._t, traj)
            _save_plot(folder, traj, plot_axes)

            results.append({**params, "folder": str(folder)})
            print(f"  [{idx}/{total}] {name}")

        print(f"[LorenzSDE] Done. Output in '{out_dir}'")
        return results

    # ------------------------------------------------------------------ #
    # Sensitivity analysis
    # ------------------------------------------------------------------ #

    def run_sensitivity(
        self,
        base_params: dict,
        ranges: dict[str, Sequence[float]],
        out_dir: str | Path = "lorenz_sensitivity",
        plot_axes: tuple[int, int] = (0, 1),
    ) -> None:
        """
        Vary one parameter at a time while keeping the others at base_params.

        Parameters
        ----------
        base_params : dict with keys sigma, rho, beta, noise_scale, y0
        ranges      : dict mapping each parameter name to a list of values to sweep
        out_dir     : root output directory

        Output layout
        -------------
        out_dir/
          <param_name>/
            overview.png          – all values overlaid on one phase plot
            <param>=<val>/
              params.json
              trajectory.csv
              phase_xy.png
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        bp = base_params                         # shorthand
        a, b = plot_axes
        labels = ["x", "y", "z"]

        for param_name, values in ranges.items():
            print(f"[sensitivity] Sweeping '{param_name}' over {values}")
            param_dir = out_dir / param_name
            param_dir.mkdir(parents=True, exist_ok=True)

            # ---- overview figure ----------------------------------------
            n_vals  = len(values)
            cmap    = plt.cm.get_cmap("plasma", n_vals)
            ov_fig, ov_ax = plt.subplots(figsize=(6, 6))
            ov_ax.set_title(f"Sensitivity: {param_name}", fontsize=13)
            ov_ax.set_xlabel(labels[a])
            ov_ax.set_ylabel(labels[b])
            ov_ax.grid(True, alpha=0.3)

            for ci, val in enumerate(values):
                p = {**bp, param_name: val}      # patch one parameter

                traj = self.simulate(
                    p["sigma"], p["rho"], p["beta"], p["noise_scale"], p["y0"]
                )

                # ---- per-value subfolder ----
                sub = param_dir / f"{param_name}={val}"
                sub.mkdir(parents=True, exist_ok=True)
                _save_params(sub, p)
                _save_csv(sub, self._t, traj)
                _save_plot(sub, traj, plot_axes)

                # ---- add to overview ----
                ov_ax.plot(
                    traj[:, a], traj[:, b],
                    lw=0.5, color=cmap(ci), label=f"{param_name}={val}"
                )

            ov_ax.legend(fontsize=7, loc="best")
            ov_fig.tight_layout()
            ov_fig.savefig(param_dir / "overview.png", dpi=150, bbox_inches="tight")
            plt.close(ov_fig)

            # ---- time-series overview ------------------------------------
            ts_fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            ts_fig.suptitle(f"Time series — sweeping '{param_name}'", fontsize=13)
            coord_labels = ["x", "y", "z"]

            for ci, val in enumerate(values):
                p    = {**bp, param_name: val}
                traj = self.simulate(
                    p["sigma"], p["rho"], p["beta"], p["noise_scale"], p["y0"]
                )
                for dim, ax in enumerate(axes):
                    ax.plot(self._t, traj[:, dim], lw=0.5,
                            color=cmap(ci), label=f"{val}" if dim == 0 else "")
                    ax.set_ylabel(coord_labels[dim])
                    ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("time")
            axes[0].legend(title=param_name, fontsize=7, ncol=min(n_vals, 6),
                           loc="upper right")
            ts_fig.tight_layout()
            ts_fig.savefig(param_dir / "timeseries_overview.png",
                           dpi=150, bbox_inches="tight")
            plt.close(ts_fig)

            print(f"  → saved to '{param_dir}'")

        print(f"[LorenzSDE] Sensitivity done. Output in '{out_dir}'")
