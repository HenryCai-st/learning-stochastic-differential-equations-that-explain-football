"""
recover_posterior.py
==================
Posterior recovery step for the trained ratio classifier.

Historically this file trained a direct regressor q(theta | track).
The current workflow loads C_phi(D, theta), scores candidate theta values,
and approximate the posterior with normalized ratio weights:

  p(theta | D) ∝ p(theta) r_phi(D, theta)

When candidate parameters are sampled from the prior, the prior is represented by the
candidate sampling distribution, so posterior weights are softmax(logit C_phi(D, theta)).
This script evaluates parameter recovery on synthetic tracks and saves a compact report.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_data import sample_parameters
from train_ratio_classifier import RatioClassifier
from src.data.dataset import SDEDataset
from src.models.encoder import TrajectoryEncoder


PARAMETER_NAMES = ["sigma", "rho", "beta", "noise_scale"]


def unique_rows(values: np.ndarray) -> np.ndarray:
    """Return stable unique rows from a 2D array."""
    seen = set()
    rows = []
    for row in values:
        key = tuple(np.asarray(row).round(8).tolist())
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def build_candidate_bank(dataset: SDEDataset, source: str, n_candidates: int, seed: int) -> np.ndarray:
    """
    Build normalized candidate theta values used to approximate posterior weights.

    source="dataset" uses the empirical parameter support from generated data.
    source="prior" draws fresh parameters from the same prior as generate_data.py.
    """
    if source == "dataset":
        physical = unique_rows(dataset.parameters)
    elif source == "prior":
        rng = np.random.default_rng(seed)
        physical = sample_parameters(n_candidates, rng)
    else:
        raise ValueError(f"Unknown candidate source: {source}")

    normalized = dataset.normalizer.normalize(physical).astype(np.float32)
    return normalized


def load_ratio_classifier(checkpoint_path: Path, device: torch.device) -> RatioClassifier:
    encoder = TrajectoryEncoder(feature_dim=256)
    model = RatioClassifier(encoder, feature_dim=256, param_dim=4)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Defensive fallback for a raw state_dict checkpoint.
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def score_candidates(
    model: RatioClassifier,
    tracks: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Score every track against every theta candidate.

    Returns
    -------
    logits  : (B, K)
    weights : (B, K), normalized posterior weights over candidates
    """
    batch_size = tracks.size(0)
    n_candidates = candidates.size(0)

    tracks_rep = tracks[:, None].expand(batch_size, n_candidates, *tracks.shape[1:])
    tracks_rep = tracks_rep.reshape(batch_size * n_candidates, *tracks.shape[1:])
    theta_rep = candidates[None].expand(batch_size, n_candidates, candidates.size(1))
    theta_rep = theta_rep.reshape(batch_size * n_candidates, candidates.size(1))

    logits = model(tracks_rep, theta_rep).reshape(batch_size, n_candidates)
    weights = torch.softmax(logits / temperature, dim=1)
    return logits, weights


def posterior_metrics(
    pred_mean_norm: np.ndarray,
    map_norm: np.ndarray,
    target_norm: np.ndarray,
    normalizer,
) -> dict[str, np.ndarray | float]:
    pred_mean_phys = normalizer.denormalize(pred_mean_norm)
    map_phys = normalizer.denormalize(map_norm)
    target_phys = normalizer.denormalize(target_norm)

    return {
        "mean_mae_norm": np.abs(pred_mean_norm - target_norm).mean(axis=0),
        "map_mae_norm": np.abs(map_norm - target_norm).mean(axis=0),
        "mean_mae_phys": np.abs(pred_mean_phys - target_phys).mean(axis=0),
        "map_mae_phys": np.abs(map_phys - target_phys).mean(axis=0),
        "overall_mean_mae_norm": float(np.abs(pred_mean_norm - target_norm).mean()),
        "overall_map_mae_norm": float(np.abs(map_norm - target_norm).mean()),
    }


def evaluate_posterior_recovery(
    model: RatioClassifier,
    loader: DataLoader,
    candidate_bank: np.ndarray,
    dataset: SDEDataset,
    device: torch.device,
    temperature: float,
    append_true_theta: bool,
) -> dict[str, np.ndarray | float]:
    all_mean = []
    all_map = []
    all_target = []
    all_entropy = []
    all_ess = []
    all_true_weight = []

    base_candidates = torch.from_numpy(candidate_bank).float().to(device)

    for query_t, _, _, params_t in loader:
        tracks = query_t.to(device)
        targets = params_t.to(device)

        if append_true_theta:
            candidates = torch.cat([base_candidates, targets], dim=0)
        else:
            candidates = base_candidates

        logits, weights = score_candidates(model, tracks, candidates, temperature)
        mean = weights @ candidates
        map_idx = weights.argmax(dim=1)
        map_theta = candidates[map_idx]

        entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=1)
        ess = 1.0 / weights.square().sum(dim=1).clamp_min(1e-12)

        all_mean.append(mean.cpu().numpy())
        all_map.append(map_theta.cpu().numpy())
        all_target.append(targets.cpu().numpy())
        all_entropy.append(entropy.cpu().numpy())
        all_ess.append(ess.cpu().numpy())

        if append_true_theta:
            n_base = base_candidates.size(0)
            true_indices = torch.arange(targets.size(0), device=device) + n_base
            all_true_weight.append(weights[torch.arange(targets.size(0), device=device), true_indices].cpu().numpy())

    pred_mean_norm = np.concatenate(all_mean, axis=0)
    map_norm = np.concatenate(all_map, axis=0)
    target_norm = np.concatenate(all_target, axis=0)

    metrics = posterior_metrics(pred_mean_norm, map_norm, target_norm, dataset.normalizer)
    metrics.update({
        "entropy": np.concatenate(all_entropy, axis=0),
        "ess": np.concatenate(all_ess, axis=0),
        "mean_entropy": float(np.concatenate(all_entropy, axis=0).mean()),
        "mean_ess": float(np.concatenate(all_ess, axis=0).mean()),
        "pred_mean_norm": pred_mean_norm,
        "map_norm": map_norm,
        "target_norm": target_norm,
    })

    if all_true_weight:
        true_weight = np.concatenate(all_true_weight, axis=0)
        metrics["true_weight"] = true_weight
        metrics["mean_true_weight"] = float(true_weight.mean())

    return metrics


def print_metrics(prefix: str, metrics: dict[str, np.ndarray | float]) -> None:
    print(f"\n{prefix}")
    print(f"  overall posterior-mean MAE(norm): {metrics['overall_mean_mae_norm']:.4f}")
    print(f"  overall MAP MAE(norm):            {metrics['overall_map_mae_norm']:.4f}")
    print(f"  posterior mean ESS:               {metrics['mean_ess']:.2f}")
    print(f"  posterior mean entropy:           {metrics['mean_entropy']:.3f}")
    if "mean_true_weight" in metrics:
        print(f"  mean posterior weight on true θ:   {metrics['mean_true_weight']:.4f}")

    print("  MAE physical by parameter:")
    for name, mean_mae, map_mae in zip(PARAMETER_NAMES, metrics["mean_mae_phys"], metrics["map_mae_phys"]):
        print(f"    {name:11s} mean={mean_mae:.4f}  map={map_mae:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/lorenz_dataset")
    parser.add_argument("--ratio_ckpt", type=str, default="./checkpoints/ratio_classifier_best.pt")
    parser.add_argument("--candidate_source", type=str, default="dataset", choices=["dataset", "prior"])
    parser.add_argument("--n_candidates", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--append_true_theta", action="store_true", help="Append each track's true theta to the candidate bank for diagnostic recovery.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Posterior recovery from ratio classifier")

    dataset = SDEDataset(args.data_dir)
    candidate_bank = build_candidate_bank(dataset, args.candidate_source, args.n_candidates, args.seed + 17)
    print(f"Candidate source: {args.candidate_source} | candidates: {len(candidate_bank)}")

    val_size = int(len(dataset) * 0.2)
    if len(dataset) >= 4:
        val_size = max(2, val_size)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ckpt_path = Path(args.ratio_ckpt)
    if not ckpt_path.exists():
        print(f"Ratio classifier checkpoint not found: {ckpt_path}")
        print("Run scripts/train_ratio_classifier.py first.")
        sys.exit(1)

    model = load_ratio_classifier(ckpt_path, device)
    print(f"Loaded ratio classifier from {ckpt_path}")

    train_metrics = evaluate_posterior_recovery(
        model,
        train_loader,
        candidate_bank,
        dataset,
        device,
        args.temperature,
        args.append_true_theta,
    )
    val_metrics = evaluate_posterior_recovery(
        model,
        val_loader,
        candidate_bank,
        dataset,
        device,
        args.temperature,
        args.append_true_theta,
    )

    print_metrics("TRAIN", train_metrics)
    print_metrics("VAL", val_metrics)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "posterior_recovery.npz"
    np.savez_compressed(
        out_path,
        candidate_bank=candidate_bank,
        train_pred_mean_norm=train_metrics["pred_mean_norm"],
        train_map_norm=train_metrics["map_norm"],
        train_target_norm=train_metrics["target_norm"],
        val_pred_mean_norm=val_metrics["pred_mean_norm"],
        val_map_norm=val_metrics["map_norm"],
        val_target_norm=val_metrics["target_norm"],
        train_mean_mae_phys=train_metrics["mean_mae_phys"],
        val_mean_mae_phys=val_metrics["mean_mae_phys"],
        train_map_mae_phys=train_metrics["map_mae_phys"],
        val_map_mae_phys=val_metrics["map_mae_phys"],
        candidate_source=args.candidate_source,
        temperature=args.temperature,
        append_true_theta=args.append_true_theta,
    )
    print(f"\nSaved posterior recovery report to {out_path}")


if __name__ == "__main__":
    main()


