"""
run_lorenz.py
=============
Standalone runner.  Edit the CONFIGURATION section below, then run:

    python run_lorenz.py [--mode grid|sensitivity|both]

Modes
-----
grid         – simulate every combination in GRID_PARAMS, save CSV + PNG
sensitivity  – vary one parameter at a time (overview plots included)
both         – run sensitivity first, then the full grid
"""

import argparse
import sys
from pathlib import Path

# ── make sure the module is importable when run from the same directory ──────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sde.lorenz_sde import LorenzSDE


# ============================================================
# CONFIGURATION  – edit this section
# ============================================================

# --- Solver settings ----------------------------------------------------------
T   = 5.0          # total time
DT  = 0.001        # step size
SEED = 42

# --- Grid: all combinations of these arrays will be simulated -----------------
GRID_PARAMS = dict(
    sigmas       = [1.0, 5.0, 10.0],
    rhos         = [14.0, 28.0, 35.0],
    betas        = [2.0, 8/3, 4.0],
    noise_scales = [0.05, 0.15],
    y0s          = [(1.0, 1.0, 1.0), (5.0, 5.0, 5.0)],
)

# --- Sensitivity: base point + ranges to sweep --------------------------------
BASE_PARAMS = dict(
    sigma       = 10.0,
    rho         = 28.0,
    beta        = 8/3,
    noise_scale = 0.05,
    y0          = [1.0, 1.0, 1.0],
)

SENSITIVITY_RANGES = dict(
    sigma       = [1.0,  5.0, 10.0, 20.0],
    rho         = [10.0, 20.0, 28.0, 40.0],
    beta        = [1.0,  8/3,  4.0,  8.0],
    noise_scale = [0.0,  0.05, 0.15, 0.30],
)

# --- Output directories -------------------------------------------------------
GRID_OUT        = "./outputs/lorenz_grid"
SENSITIVITY_OUT = "./outputs/lorenz_sensitivity"

# --- Phase-plane axes to plot: 0=x,1=y,2=z -----------------------------------
PLOT_AXES = (0, 1)   # x vs y

# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Lorenz SDE runner")
    parser.add_argument(
        "--mode",
        choices=["grid", "sensitivity", "both"],
        default="both",
        help="Which run mode to execute (default: both)",
    )
    args = parser.parse_args()

    model = LorenzSDE(T=T, dt=DT, seed=SEED)

    if args.mode in ("sensitivity", "both"):
        print("\n" + "=" * 60)
        print("SENSITIVITY ANALYSIS")
        print("=" * 60)
        if not Path(SENSITIVITY_OUT).exists():
            Path(SENSITIVITY_OUT).mkdir(parents=True)
        model.run_sensitivity(
            base_params = BASE_PARAMS,
            ranges      = SENSITIVITY_RANGES,
            out_dir     = SENSITIVITY_OUT,
            plot_axes   = PLOT_AXES,
        )

    if args.mode in ("grid", "both"):
        print("\n" + "=" * 60)
        print("FULL GRID RUN")
        print("=" * 60)
        if not Path(GRID_OUT).exists():
            Path(GRID_OUT).mkdir(parents=True)
        results = model.run_grid(
            **GRID_PARAMS,
            out_dir    = GRID_OUT,
            plot_axes  = PLOT_AXES,
        )
        print(f"\nTotal trajectories generated: {len(results)}")


if __name__ == "__main__":
    main()
