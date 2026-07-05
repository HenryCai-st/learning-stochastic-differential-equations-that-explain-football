"""
generate_data.py
================
CLI script to sample parameters (sigma, rho, beta, noise_scale) using stratified sampling
and generate N trajectories per parameter set using a vectorized NumPy Euler-Maruyama SDE simulator.
Saves a compressed .npz file containing trajectories, parameters, group IDs, and prior metadata.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


PARAMETER_NAMES = np.array(["sigma", "rho", "beta", "noise_scale"])

# Stage 1 prior for the single Lorenz SDE model. Keep this metadata explicit so
# training, normalization, and evaluation use the same parameter support.
PARAMETER_LOW = np.array([1.0, 0.5, 0.5, 0.01], dtype=np.float32)
PARAMETER_HIGH = np.array([20.0, 50.0, 5.0, 0.5], dtype=np.float32)
PARAMETER_LOG_SCALE = np.array([False, True, False, True])
RHO_REGIME_BOUNDS = np.array([0.5, 13.9, 24.74, 50.0], dtype=np.float32)


def sample_parameters(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample parameters using stratified sampling for rho to cover all dynamical regimes.

    Returns
    -------
    np.ndarray of shape (n_samples, 4) - columns: sigma, rho, beta, noise_scale
    """
    params = np.zeros((n_samples, 4), dtype=np.float32)

    # 1. Sample sigma: Uniform(1.0, 20.0)
    params[:, 0] = rng.uniform(PARAMETER_LOW[0], PARAMETER_HIGH[0], size=n_samples)

    # 2. Sample rho: stratified sampling over three regimes.
    # Regime 1: fixed points / decay (0.5 to 13.9)
    # Regime 2: limit cycles / transient chaos (13.9 to 24.74)
    # Regime 3: chaos (24.74 to 50.0)
    n_regime = n_samples // 3
    n_rem = n_samples % 3

    rho_r1 = rng.uniform(RHO_REGIME_BOUNDS[0], RHO_REGIME_BOUNDS[1], size=n_regime)
    rho_r2 = rng.uniform(RHO_REGIME_BOUNDS[1], RHO_REGIME_BOUNDS[2], size=n_regime)
    rho_r3 = rng.uniform(RHO_REGIME_BOUNDS[2], RHO_REGIME_BOUNDS[3], size=n_regime + n_rem)

    rho = np.concatenate([rho_r1, rho_r2, rho_r3])
    rng.shuffle(rho)
    params[:, 1] = rho

    # 3. Sample beta: Uniform(0.5, 5.0)
    params[:, 2] = rng.uniform(PARAMETER_LOW[2], PARAMETER_HIGH[2], size=n_samples)

    # 4. Sample noise_scale (epsilon): Log-Uniform(0.01, 0.5)
    log_noise = rng.uniform(np.log10(PARAMETER_LOW[3]), np.log10(PARAMETER_HIGH[3]), size=n_samples)
    params[:, 3] = 10 ** log_noise

    return params


def simulate_batch(
    sigma: float,
    rho: float,
    beta: float,
    noise_scale: float,
    y0_batch: np.ndarray,
    t_grid: np.ndarray,
    rng: np.random.Generator,
    clip_value: float = 100.0,
) -> np.ndarray:
    """
    Simulate a batch of trajectories for the same parameter set in parallel
    using the Euler-Maruyama scheme.

    Update formulas for Diagonal Noise Stochastic Lorenz System:
      dx = sigma * (y - x) * dt + noise_scale * x * dW_x
      dy = (x * (rho - z) - y) * dt + noise_scale * y * dW_y
      dz = (x * y - beta * z) * dt + noise_scale * z * dW_z
    """
    n_tracks = y0_batch.shape[0]
    steps = len(t_grid)
    dt = t_grid[1] - t_grid[0]
    sqrt_dt = np.sqrt(dt)

    tracks = np.zeros((n_tracks, steps, 3), dtype=np.float32)
    tracks[:, 0] = y0_batch

    current_state = y0_batch.astype(np.float32).copy()

    for i in range(1, steps):
        x = current_state[:, 0]
        y = current_state[:, 1]
        z = current_state[:, 2]

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        dw = rng.normal(0, sqrt_dt, size=(n_tracks, 3)).astype(np.float32)

        current_state[:, 0] += dx * dt + noise_scale * x * dw[:, 0]
        current_state[:, 1] += dy * dt + noise_scale * y * dw[:, 1]
        current_state[:, 2] += dz * dt + noise_scale * z * dw[:, 2]

        if np.isnan(current_state).any() or np.isinf(current_state).any():
            current_state = np.nan_to_num(
                current_state, nan=0.0, posinf=clip_value, neginf=-clip_value
            )
        current_state = np.clip(current_state, -clip_value, clip_value)

        tracks[:, i] = current_state

    return tracks


def rho_regime_labels(rho_values: np.ndarray) -> np.ndarray:
    """Map rho values to regime labels used only for diversity visualization."""
    labels = np.empty(len(rho_values), dtype=object)
    labels[rho_values < RHO_REGIME_BOUNDS[1]] = "fixed/decay"
    labels[(rho_values >= RHO_REGIME_BOUNDS[1]) & (rho_values < RHO_REGIME_BOUNDS[2])] = "transition"
    labels[rho_values >= RHO_REGIME_BOUNDS[2]] = "chaos"
    return labels


def plot_dataset_diversity(
    tracks: np.ndarray,
    parameters: np.ndarray,
    out_path: Path,
    max_tracks: int = 300,
    seed: int = 42,
) -> None:
    """
    Plot many generated trajectories in XY projection to show dataset diversity.

    The plot is intentionally simple for a report/presentation: each line is one
    rendered trajectory, colored by Lorenz rho regime.
    
    Saves both a combined plot and separate plots for each regime.
    """
    rng = np.random.default_rng(seed)
    n = len(tracks)
    if n == 0:
        raise ValueError("No trajectories available for diversity plotting.")

    chosen = rng.choice(n, size=min(max_tracks, n), replace=False)
    labels = rho_regime_labels(parameters[chosen, 1])
    colors = {
        "fixed/decay": "tab:blue",
        "transition": "tab:orange",
        "chaos": "tab:green",
    }

    # Create output directory
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- Combined plot (all regimes together) ---
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    for idx, label in zip(chosen, labels):
        xy = tracks[idx, :, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=colors[label], alpha=0.18, linewidth=0.8)

    # Add clean legend entries once per regime.
    for label, color in colors.items():
        ax.plot([], [], color=color, linewidth=2, label=label)

    ax.set_title(f"Generated Lorenz SDE trajectories: dataset diversity, {len(chosen)} tracks")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(title="rho regime", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    
    # --- Separate plots for each regime ---
    for regime_name, color in colors.items():
        # Get indices for this regime
        regime_indices = [i for i, label in enumerate(labels) if label == regime_name]
        
        if len(regime_indices) == 0:
            print(f"Warning: No trajectories found for regime '{regime_name}'")
            continue
        
        # Create figure for this regime
        fig, ax = plt.subplots(figsize=(8.0, 6.5))
        
        # Plot all trajectories in this regime
        for idx in regime_indices:
            xy = tracks[chosen[idx], :, :2]
            ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.25, linewidth=0.8)
        
        # Add a single legend entry for this regime
        ax.plot([], [], color=color, linewidth=2, label=regime_name)
        
        # Create title with regime name
        regime_display_name = regime_name.replace("/", " / ")  # Make "fixed/decay" more readable
        ax.set_title(f"Lorenz SDE trajectories: {regime_display_name} regime, {len(regime_indices)} tracks")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)
        ax.legend(title="rho regime", frameon=True)
        fig.tight_layout()
        
        # Save with regime name in filename
        regime_filename = out_path.stem + f"_{regime_name.replace('/', '_')}" + out_path.suffix
        regime_path = out_path.parent / regime_filename
        fig.savefig(regime_path, dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Lorenz SDE dataset")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of parameter sets to sample")
    parser.add_argument("--n_tracks", type=int, default=20, help="Number of trajectories per parameter set")
    parser.add_argument("--T", type=float, default=5.0, help="Total simulation time")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation time step")
    parser.add_argument("--y0_noise_std", type=float, default=0.05, help="Initial-state jitter per track")
    parser.add_argument("--clip_value", type=float, default=100.0, help="State clipping threshold for unstable simulations")
    parser.add_argument("--out_dir", type=str, default="./data/lorenz_dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--plot_diversity", action="store_true", help="Save an XY plot of generated trajectories to prove diversity")
    parser.add_argument("--max_plot_tracks", type=int, default=300, help="Maximum trajectories drawn in diversity plot")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating SDE dataset with {args.n_samples} samples, {args.n_tracks} tracks each...")
    print(f"Simulation settings: T={args.T}, dt={args.dt} ({int(args.T / args.dt)} steps)")

    params = sample_parameters(args.n_samples, rng)
    t_grid = np.arange(0, args.T + args.dt, args.dt)
    base_y0 = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    total_tracks = args.n_samples * args.n_tracks
    all_tracks = np.zeros((total_tracks, len(t_grid), 3), dtype=np.float32)
    all_params = np.zeros((total_tracks, 4), dtype=np.float32)
    param_group_ids = np.zeros(total_tracks, dtype=np.int32)

    for i in range(args.n_samples):
        sigma, rho, beta, noise_scale = params[i]
        y0_batch = base_y0 + rng.normal(0, args.y0_noise_std, size=(args.n_tracks, 3))
        tracks = simulate_batch(
            sigma,
            rho,
            beta,
            noise_scale,
            y0_batch,
            t_grid,
            rng,
            clip_value=args.clip_value,
        )

        start_idx = i * args.n_tracks
        end_idx = start_idx + args.n_tracks
        all_tracks[start_idx:end_idx] = tracks
        all_params[start_idx:end_idx] = params[i]
        param_group_ids[start_idx:end_idx] = i

    np.savez_compressed(
        out_dir / "dataset.npz",
        tracks=all_tracks,
        parameters=all_params,
        group_ids=param_group_ids,
        parameter_names=PARAMETER_NAMES,
        parameter_low=PARAMETER_LOW,
        parameter_high=PARAMETER_HIGH,
        parameter_log_scale=PARAMETER_LOG_SCALE,
        rho_regime_bounds=RHO_REGIME_BOUNDS,
        t_grid=t_grid.astype(np.float32),
        base_y0=base_y0,
        n_samples=args.n_samples,
        n_tracks=args.n_tracks,
        T=args.T,
        dt=args.dt,
        y0_noise_std=args.y0_noise_std,
        clip_value=args.clip_value,
        seed=args.seed,
    )
    print(f"Saved dataset to {out_dir / 'dataset.npz'}")

    if args.plot_diversity:
        diversity_path = out_dir / "dataset_diversity_xy.png"
        plot_dataset_diversity(
            all_tracks,
            all_params,
            diversity_path,
            max_tracks=args.max_plot_tracks,
            seed=args.seed + 999,
        )
        print(f"Saved diversity plot to {diversity_path}")


if __name__ == "__main__":
    main()
