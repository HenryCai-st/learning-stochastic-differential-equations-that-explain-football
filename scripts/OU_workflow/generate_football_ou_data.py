"""
Generate synthetic tracks for the single-model football OU baseline.

Inputs:
    - optional real football windows to bootstrap start/end conditions
    - OU parameter priors from src/sde/football_ou.py

Outputs:
    - data/football_ou_dataset/dataset.npz with OU tracks, parameters, y0, and
      target points.

Expected use:
    This is the older OU baseline workflow. It remains useful as a comparison
    showing why the final project moved to model voting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.sde.football_ou import (
    PARAMETER_HIGH,
    PARAMETER_LOW,
    PARAMETER_NAMES,
    sample_ou_parameters,
    simulate_position_ou_batch,
)


def load_condition_pool(path: str | Path | None, rng: np.random.Generator, n: int):
    """
    Choose start/target pairs for synthetic simulations.

    This is the football-specific replacement for Lorenz's fixed initial
    condition. The OU dynamics depend strongly on where a window starts and
    where it is trying to go, so synthetic training data should use a similar
    y0/target distribution to the real extracted windows.
    """
    if path is not None and Path(path).exists():
        real = np.load(path, allow_pickle=True)
        y0_pool = real["y0"].astype(np.float32)
        target_pool = real["target"].astype(np.float32)
        choice = rng.choice(len(y0_pool), size=n, replace=True)
        return y0_pool[choice], target_pool[choice], f"bootstrapped:{path}"

    y0 = np.column_stack([
        rng.uniform(0.0, 105.0, size=n),
        rng.uniform(0.0, 68.0, size=n),
    ]).astype(np.float32)
    disp = rng.normal(0.0, [15.0, 10.0], size=(n, 2)).astype(np.float32)
    target = y0 + disp
    target[:, 0] = np.clip(target[:, 0], 0.0, 105.0)
    target[:, 1] = np.clip(target[:, 1], 0.0, 68.0)
    return y0, target, "synthetic_uniform_pitch"


def main() -> None:
    """
    Generate the football OU baseline training dataset.

    Lorenz analogue:
        scripts/generate_data.py

    Football difference:
        each generated track stores y0 and target as conditioning variables,
        because C_phi must learn C(track, theta, y0, target), not just
        C(track, theta).
    """
    parser = argparse.ArgumentParser(description="Generate synthetic football OU tracks for ratio training.")
    parser.add_argument("--real-windows", default="data/real_football_windows.npz")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-tracks", type=int, default=20)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/football_ou_dataset")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    steps = int(round(args.T / args.dt))
    n_total = args.n_samples * args.n_tracks

    # Sample parameter groups first, then repeat each theta n_tracks times so
    # the dataset contains multiple stochastic realizations per parameter set.
    params_group = sample_ou_parameters(args.n_samples, rng)
    params = np.repeat(params_group, args.n_tracks, axis=0)
    group_ids = np.repeat(np.arange(args.n_samples, dtype=np.int32), args.n_tracks)

    # Bootstrap real y0/target pairs when available. This narrows the sim-real
    # gap compared with drawing starts and endpoints from a naive uniform box.
    y0, target, condition_source = load_condition_pool(args.real_windows, rng, n_total)

    # Vectorized Euler-Maruyama simulation for all synthetic windows.
    tracks = simulate_position_ou_batch(
        params=params,
        y0=y0,
        target=target,
        steps=steps,
        dt=args.dt,
        rng=rng,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "dataset.npz",
        tracks=tracks,
        parameters=params.astype(np.float32),
        y0=y0.astype(np.float32),
        target=target.astype(np.float32),
        group_ids=group_ids,
        parameter_names=PARAMETER_NAMES,
        parameter_low=PARAMETER_LOW,
        parameter_high=PARAMETER_HIGH,
        T=args.T,
        dt=args.dt,
        steps=steps,
        seed=args.seed,
        condition_source=condition_source,
    )
    print(json.dumps({"out": str(out_dir / "dataset.npz"), "tracks": int(n_total), "condition_source": condition_source}, indent=2))


if __name__ == "__main__":
    main()
