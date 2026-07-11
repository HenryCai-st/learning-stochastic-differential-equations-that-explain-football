"""
train_ratio_classifier.py
====================
Stage 1 training for SBI via ratio estimation.

This script trains a matched-vs-mismatched classifier C_phi(D, theta):
  positive pairs: trajectory D and the parameter theta that generated it
  negative pairs: trajectory D and an independently paired/shuffled theta

With balanced positive/negative batches, the classifier logit estimates a
likelihood-to-evidence log-ratio up to the usual class-prior convention:
  logit C_phi(D, theta) ~= log p(D | theta) - log p(D)

The trajectory encoder can still be viewed as a contrastive feature extractor,
but the core objective is BCE on matched-vs-mismatched pairs, not triplet or
SimCLR. This follows the project plan: infer a posterior over parameters before simulating predictive trajectories.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

def find_project_root(start: Path) -> Path:
    """Find repository root from this nested workflow script."""
    for parent in [start, *start.parents]:
        if (parent / "src").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not locate project root containing src/ and scripts/.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import SDEDataset
from src.models.encoder import TrajectoryEncoder


class RatioClassifier(nn.Module):
    """Classifier head for C_phi(track, theta)."""

    def __init__(self, encoder: TrajectoryEncoder, feature_dim: int = 256, param_dim: int = 4):
        """Create the trajectory encoder, theta encoder, and binary head."""
        super().__init__()
        self.encoder = encoder
        self.theta_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, tracks: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """tracks: (B, 3, steps), params: (B, 4) normalized -> logits: (B,)"""
        z_track = self.encoder(tracks)
        z_theta = self.theta_encoder(params)
        return self.classifier(torch.cat([z_track, z_theta], dim=1)).squeeze(-1)


def make_negative_params(params: torch.Tensor) -> torch.Tensor:
    """
    Pair every trajectory with another parameter vector from the same batch.
    A one-step roll is deterministic and preserves the empirical prior in-batch.
    """
    if params.size(0) < 2:
        raise ValueError("Batch size must be at least 2 for mismatched pairs.")
    return torch.roll(params, shifts=1, dims=0)


def batch_loss_and_metrics(model: RatioClassifier, tracks: torch.Tensor, params: torch.Tensor):
    """Compute matched/mismatched BCE loss and batch diagnostics."""
    neg_params = make_negative_params(params)

    pos_logits = model(tracks, params)
    neg_logits = model(tracks, neg_params)

    logits = torch.cat([pos_logits, neg_logits], dim=0)
    targets = torch.cat([
        torch.ones_like(pos_logits),
        torch.zeros_like(neg_logits),
    ], dim=0)

    loss = F.binary_cross_entropy_with_logits(logits, targets)
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        acc = ((probs >= 0.5).float() == targets).float().mean()
        pos_score = torch.sigmoid(pos_logits).mean()
        neg_score = torch.sigmoid(neg_logits).mean()
        log_ratio_gap = pos_logits.mean() - neg_logits.mean()

    return loss, {
        "acc": acc.item(),
        "pos_score": pos_score.item(),
        "neg_score": neg_score.item(),
        "log_ratio_gap": log_ratio_gap.item(),
    }


def run_epoch(model, loader, optimizer, device, training: bool):
    """Run one train or validation epoch for the Lorenz ratio classifier."""
    model.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()

    totals = {
        "loss": 0.0,
        "acc": 0.0,
        "pos_score": 0.0,
        "neg_score": 0.0,
        "log_ratio_gap": 0.0,
    }
    n = 0

    with ctx:
        for query_t, _, _, params_t in loader:
            if params_t.size(0) < 2:
                continue

            tracks = query_t.to(device)
            params = params_t.to(device)

            loss, metrics = batch_loss_and_metrics(model, tracks, params)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            totals["loss"] += loss.item()
            for key, value in metrics.items():
                totals[key] += value
            n += 1

    if n == 0:
        raise ValueError("No usable batches: ratio training needs batches with at least 2 items.")
    return {key: value / n for key, value in totals.items()}


def main():
    """Train the Lorenz matched-vs-mismatched ratio classifier."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/lorenz_dataset")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Matched-vs-mismatched ratio classifier")

    if args.batch_size < 2:
        raise ValueError("--batch_size must be at least 2 for mismatched-pair training")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = SDEDataset(args.data_dir)
    except FileNotFoundError:
        print("Dataset not found. Run scripts/generate_data.py first.")
        sys.exit(1)

    val_size = int(len(dataset) * 0.2)
    if len(dataset) >= 4:
        val_size = max(2, val_size)
    train_size = len(dataset) - val_size
    if train_size < 2:
        raise ValueError("Dataset is too small: need at least 2 training items for mismatched pairs.")
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    encoder = TrajectoryEncoder(feature_dim=256)
    model = RatioClassifier(encoder, feature_dim=256, param_dim=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    print(f"Training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, training=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, training=False)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.2%} "
            f"pos {train_metrics['pos_score']:.3f} neg {train_metrics['neg_score']:.3f} "
            f"gap {train_metrics['log_ratio_gap']:.3f} | "
            f"Val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.2%} "
            f"pos {val_metrics['pos_score']:.3f} neg {val_metrics['neg_score']:.3f} "
            f"gap {val_metrics['log_ratio_gap']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt = out_dir / "ratio_classifier_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "param_dim": 4,
                    "feature_dim": 256,
                    "data_dir": args.data_dir,
                    "seed": args.seed,
                    "objective": "balanced_bce_matched_vs_mismatched",
                    "best_val_loss": best_val_loss,
                },
                ckpt,
            )
            print(f"  -> Saved {ckpt}")

    print("Ratio-classifier training complete.")


if __name__ == "__main__":
    main()


