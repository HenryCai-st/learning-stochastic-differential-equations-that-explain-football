"""
generate_data.py
================
CLI script to sample parameters (sigma, rho, beta, noise_scale) using stratified sampling
and generate N trajectories per parameter set using a vectorized NumPy Euler-Maruyama SDE simulator.
Saves each run as a compressed .npz file containing the parameters and trajectory arrays.
"""

import argparse
import sys
from pathlib import Path
import numpy as np


def sample_parameters(n_samples: int) -> np.ndarray:
    """
    Sample parameters using stratified sampling for rho to cover all dynamical regimes.
    
    Returns
    -------
    np.ndarray of shape (n_samples, 4) - columns: sigma, rho, beta, noise_scale
    """
    # Initialize parameter array
    params = np.zeros((n_samples, 4))
    
    # 1. Sample sigma: Uniform(1.0, 20.0)
    params[:, 0] = np.random.uniform(1.0, 20.0, size=n_samples)
    
    # 2. Sample rho: Stratified sampling over three regimes
    # Regime 1: Fixed points / decay (0.5 to 13.9)
    # Regime 2: Limit cycles / transient chaos (13.9 to 24.74)
    # Regime 3: Chaos (24.74 to 50.0)
    n_regime = n_samples // 3
    n_rem = n_samples % 3
    
    rho_r1 = np.random.uniform(0.5, 13.9, size=n_regime)
    rho_r2 = np.random.uniform(13.9, 24.74, size=n_regime)
    # The remaining samples go to the chaotic regime
    rho_r3 = np.random.uniform(24.74, 50.0, size=n_regime + n_rem)
    
    rho = np.concatenate([rho_r1, rho_r2, rho_r3])
    np.random.shuffle(rho)
    params[:, 1] = rho
    
    # 3. Sample beta: Uniform(0.5, 5.0)
    params[:, 2] = np.random.uniform(0.5, 5.0, size=n_samples)
    
    # 4. Sample noise_scale (epsilon): Log-Uniform(0.01, 0.5)
    # log10(0.01) = -2.0, log10(0.5) = -0.301
    log_noise = np.random.uniform(-2.0, -0.301, size=n_samples)
    params[:, 3] = 10 ** log_noise
    
    return params


def simulate_batch(
    sigma: float,
    rho: float,
    beta: float,
    noise_scale: float,
    y0_batch: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Simulates a batch of trajectories for the same parameter set in parallel
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
    
    # Initialize track array (n_tracks, steps, 3)
    tracks = np.zeros((n_tracks, steps, 3), dtype=np.float32)
    tracks[:, 0] = y0_batch
    
    current_state = y0_batch.astype(np.float32).copy()
    
    for i in range(1, steps):
        x = current_state[:, 0]
        y = current_state[:, 1]
        z = current_state[:, 2]
        
        # Compute drift terms
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        # Sample Brownian increments: dW ~ N(0, dt)
        dw = np.random.normal(0, sqrt_dt, size=(n_tracks, 3)).astype(np.float32)
        
        # Apply Euler-Maruyama update step
        current_state[:, 0] += dx * dt + noise_scale * x * dw[:, 0]
        current_state[:, 1] += dy * dt + noise_scale * y * dw[:, 1]
        current_state[:, 2] += dz * dt + noise_scale * z * dw[:, 2]
        
        # Handle nan/inf overflow in chaotic regimes by clipping/nan_to_num
        if np.isnan(current_state).any() or np.isinf(current_state).any():
            current_state = np.nan_to_num(current_state, nan=0.0, posinf=100.0, neginf=-100.0)
            
        tracks[:, i] = current_state
        
    return tracks


def main():
    parser = argparse.ArgumentParser(description="Generate Lorenz SDE dataset")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of parameter sets to sample")
    parser.add_argument("--n_tracks", type=int, default=10, help="Number of trajectories per parameter set")
    parser.add_argument("--T", type=float, default=5.0, help="Total simulation time")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation time step")
    parser.add_argument("--out_dir", type=str, default="./data/lorenz_dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating SDE dataset with {args.n_samples} samples, {args.n_tracks} tracks each...")
    print(f"Simulation settings: T={args.T}, dt={args.dt} ({int(args.T/args.dt)} steps)")
    
    # Sample parameters
    params = sample_parameters(args.n_samples)
    
    # Define time grid
    t_grid = np.arange(0, args.T + args.dt, args.dt)
    
    base_y0 = np.array([1.0, 1.0, 1.0])
    
    # Pre-allocate arrays
    total_tracks = args.n_samples * args.n_tracks
    all_tracks = np.zeros((total_tracks, len(t_grid), 3), dtype=np.float32)
    all_params = np.zeros((total_tracks, 4), dtype=np.float32)
    param_group_ids = np.zeros(total_tracks, dtype=np.int32)  # Which parameter set each track belongs to
    
    # Generate data
    for i in range(args.n_samples):
        sigma, rho, beta, noise_scale = params[i]
        y0_batch = base_y0 + np.random.normal(0, 0.05, size=(args.n_tracks, 3))
        tracks = simulate_batch(sigma, rho, beta, noise_scale, y0_batch, t_grid)
        
        start_idx = i * args.n_tracks
        end_idx = start_idx + args.n_tracks
        all_tracks[start_idx:end_idx] = tracks
        all_params[start_idx:end_idx] = params[i]  # Same params repeated
        param_group_ids[start_idx:end_idx] = i  # Group ID for positive pairing
    
    # Save single file
    np.savez_compressed(
        out_dir / "dataset.npz",
        tracks=all_tracks,
        parameters=all_params,
        group_ids=param_group_ids,
        n_samples=args.n_samples,
        n_tracks=args.n_tracks
    )


if __name__ == "__main__":
    main()
