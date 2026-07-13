"""
recover_posterior_mcmc.py
=========================
Posterior recovery with Random-Walk Metropolis-Hastings for the trained
ratio classifier C_phi(D, theta).

Target density:
    log p(theta | D) = log p(theta) + log r_phi(D, theta) + const.

The ratio classifier logit is used as log r_phi(D, theta). The sampler proposes
new physical parameters with a Gaussian random walk and rejects proposals outside
prior support.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

def find_project_root(start: Path) -> Path:
    """Find repository root from this nested workflow script."""
    for parent in [start, *start.parents]:
        if (parent / "src").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not locate project root containing src/ and scripts/.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.Lorenz_workflow.generate_data import PARAMETER_LOW, PARAMETER_HIGH, RHO_REGIME_BOUNDS, sample_parameters
from scripts.Lorenz_workflow.train_ratio_classifier import RatioClassifier
from src.legacy.lorenz.dataset import SDEDataset
from src.sbi.encoder import TrajectoryEncoder

PARAMETER_NAMES = ["sigma", "rho", "beta", "noise_scale"]


def load_ratio_classifier(checkpoint_path: Path, device: torch.device) -> RatioClassifier:
    """Load a trained Lorenz ratio classifier from disk."""
    encoder = TrajectoryEncoder(feature_dim=256)
    model = RatioClassifier(encoder, feature_dim=256, param_dim=4)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()
    return model


def log_prior_physical(theta: np.ndarray) -> float:
    """Log prior used by generate_data.py, up to constants irrelevant for MH."""
    theta = np.asarray(theta, dtype=np.float64)
    if np.any(theta < PARAMETER_LOW) or np.any(theta > PARAMETER_HIGH):
        return -np.inf

    sigma, rho, beta, noise = theta
    if noise <= 0.0:
        return -np.inf

    # sigma and beta are uniform, so their log-density is constant inside support.
    # rho is sampled with equal mass in three regimes, so include the regime width.
    if rho < RHO_REGIME_BOUNDS[1]:
        rho_width = RHO_REGIME_BOUNDS[1] - RHO_REGIME_BOUNDS[0]
    elif rho < RHO_REGIME_BOUNDS[2]:
        rho_width = RHO_REGIME_BOUNDS[2] - RHO_REGIME_BOUNDS[1]
    else:
        rho_width = RHO_REGIME_BOUNDS[3] - RHO_REGIME_BOUNDS[2]

    # noise_scale is log-uniform, p(epsilon) proportional to 1 / epsilon.
    return -np.log(float(rho_width)) - np.log(float(noise))


@torch.no_grad()
def log_ratio_for_theta(model, track_t: torch.Tensor, theta_phys: np.ndarray, dataset: SDEDataset, device: torch.device) -> float:
    """Evaluate the classifier log-ratio for one physical theta value."""
    theta_norm = dataset.normalizer.normalize(theta_phys[None]).astype(np.float32)
    theta_t = torch.from_numpy(theta_norm).to(device)
    return float(model(track_t, theta_t).item())


def random_walk_metropolis_hastings(
    model,
    track: np.ndarray,
    dataset: SDEDataset,
    n_steps: int,
    burn_in: int,
    proposal_scale: np.ndarray,
    seed: int,
    device: torch.device,
    initial_theta: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Run Random-Walk Metropolis-Hastings in physical parameter space."""
    rng = np.random.default_rng(seed)
    if initial_theta is None:
        current = sample_parameters(1, rng)[0].astype(np.float64)
    else:
        current = np.asarray(initial_theta, dtype=np.float64).copy()

    norm_track = (track - dataset.track_mean) / (dataset.track_std + 1e-8)
    track_t = torch.from_numpy(norm_track.T[None]).float().to(device)

    current_logp = log_prior_physical(current) + log_ratio_for_theta(model, track_t, current, dataset, device)
    chain = np.zeros((n_steps, 4), dtype=np.float32)
    logp = np.zeros(n_steps, dtype=np.float32)
    accepted = 0

    for t in range(n_steps):
        proposal = current + rng.normal(0.0, proposal_scale, size=4)
        proposal_log_prior = log_prior_physical(proposal)
        if np.isfinite(proposal_log_prior):
            proposal_logp = proposal_log_prior + log_ratio_for_theta(model, track_t, proposal, dataset, device)
            if np.log(rng.uniform()) < proposal_logp - current_logp:
                current = proposal
                current_logp = proposal_logp
                accepted += 1
        chain[t] = current
        logp[t] = current_logp

    posterior_samples = chain[burn_in:]
    posterior_logp = logp[burn_in:]
    map_idx = int(np.argmax(posterior_logp))
    return {
        "chain": chain,
        "logp": logp,
        "samples": posterior_samples,
        "posterior_mean_phys": posterior_samples.mean(axis=0),
        "map_phys": posterior_samples[map_idx],
        "acceptance_rate": accepted / max(1, n_steps),
    }


def posterior_metrics(pred_mean: np.ndarray, map_theta: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray | float]:
    """Compute posterior mean/MAP errors against true Lorenz parameters."""
    return {
        "mean_mae_phys": np.abs(pred_mean - target).mean(axis=0),
        "map_mae_phys": np.abs(map_theta - target).mean(axis=0),
        "overall_mean_mae_phys": float(np.abs(pred_mean - target).mean()),
        "overall_map_mae_phys": float(np.abs(map_theta - target).mean()),
    }


def evaluate_mcmc_recovery(model, loader, dataset, device, args) -> dict[str, np.ndarray | float]:
    """Run MCMC recovery over validation examples and aggregate metrics."""
    pred_mean, pred_map, target, acc = [], [], [], []
    proposal_scale = np.asarray(args.proposal_scale, dtype=np.float64)

    for i, (query_t, _, _, params_t) in enumerate(loader):
        if i >= args.max_eval_items:
            break
        # Each batch item is an independent observed trajectory; use the first item of each batch.
        track_norm = query_t[0].cpu().numpy().T
        track = track_norm * (dataset.track_std + 1e-8) + dataset.track_mean
        target_phys = dataset.normalizer.denormalize(params_t[0].cpu().numpy()[None])[0]
        result = random_walk_metropolis_hastings(
            model=model,
            track=track,
            dataset=dataset,
            n_steps=args.mcmc_steps,
            burn_in=args.burn_in,
            proposal_scale=proposal_scale,
            seed=args.seed + 1000 + i,
            device=device,
            initial_theta=target_phys if args.start_at_true else None,
        )
        pred_mean.append(result["posterior_mean_phys"])
        pred_map.append(result["map_phys"])
        target.append(target_phys)
        acc.append(result["acceptance_rate"])

    pred_mean = np.asarray(pred_mean)
    pred_map = np.asarray(pred_map)
    target = np.asarray(target)
    metrics = posterior_metrics(pred_mean, pred_map, target)
    metrics.update({
        "pred_mean_phys": pred_mean,
        "map_phys": pred_map,
        "target_phys": target,
        "acceptance_rate": np.asarray(acc),
        "mean_acceptance_rate": float(np.mean(acc)),
    })
    return metrics


def print_metrics(prefix: str, metrics: dict[str, np.ndarray | float]) -> None:
    """Print a compact posterior recovery report."""
    print(f"\n{prefix}")
    print(f"  overall posterior-mean MAE(phys): {metrics['overall_mean_mae_phys']:.4f}")
    print(f"  overall MAP MAE(phys):            {metrics['overall_map_mae_phys']:.4f}")
    print(f"  mean MH acceptance rate:          {metrics['mean_acceptance_rate']:.2%}")
    print("  MAE physical by parameter:")
    for name, mean_mae, map_mae in zip(PARAMETER_NAMES, metrics["mean_mae_phys"], metrics["map_mae_phys"]):
        print(f"    {name:11s} mean={mean_mae:.4f}  map={map_mae:.4f}")


def main():
    """Load data/checkpoint, run Lorenz MCMC recovery, and save metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/lorenz_dataset")
    parser.add_argument("--ratio_ckpt", type=str, default="./checkpoints/ratio_classifier_best.pt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--mcmc_steps", type=int, default=4000)
    parser.add_argument("--burn_in", type=int, default=1000)
    parser.add_argument("--proposal_scale", nargs=4, type=float, default=[0.7, 1.5, 0.15, 0.02])
    parser.add_argument("--max_eval_items", type=int, default=16)
    parser.add_argument("--start_at_true", action="store_true", help="Diagnostic only: initialize MH at true theta")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Posterior recovery with Random-Walk Metropolis-Hastings")

    dataset = SDEDataset(args.data_dir)
    val_size = int(len(dataset) * 0.2)
    if len(dataset) >= 4:
        val_size = max(2, val_size)
    train_size = len(dataset) - val_size
    _, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt_path = Path(args.ratio_ckpt)
    if not ckpt_path.exists():
        print(f"Ratio classifier checkpoint not found: {ckpt_path}")
        print("Run train_ratio_classifier.py first.")
        sys.exit(1)
    model = load_ratio_classifier(ckpt_path, device)
    print(f"Loaded ratio classifier from {ckpt_path}")

    metrics = evaluate_mcmc_recovery(model, val_loader, dataset, device, args)
    print_metrics("VAL MCMC", metrics)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "posterior_recovery_mcmc.npz"
    np.savez_compressed(out_path, **metrics, proposal_scale=np.asarray(args.proposal_scale), burn_in=args.burn_in, mcmc_steps=args.mcmc_steps)
    print(f"\nSaved MCMC posterior recovery report to {out_path}")


if __name__ == "__main__":
    main()
