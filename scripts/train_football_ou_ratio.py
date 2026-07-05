from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.football_dataset import FootballOUDataset
from src.models.encoder import TrajectoryEncoder


class ConditionedRatioClassifier(nn.Module):
    """C_phi(track, theta, y0, target) for Phase A football OU SBI."""

    def __init__(
        self,
        encoder: TrajectoryEncoder,
        feature_dim: int = 256,
        param_dim: int = 2,
        condition_dim: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.theta_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 64 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, tracks: torch.Tensor, params: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Return one match/mismatch logit per item.

        tracks:    (B, 2, steps)
        params:    (B, 2), normalized (k, noise_scale)
        condition: (B, 4), normalized concat(y0, target)
        """
        z_track = self.encoder(tracks)
        z_theta = self.theta_encoder(params)
        z_cond = self.condition_encoder(condition)
        return self.classifier(torch.cat([z_track, z_theta, z_cond], dim=1)).squeeze(-1)


def make_negative_params(params: torch.Tensor) -> torch.Tensor:
    """
    Build mismatched examples by shuffling only theta.

    We keep y0/target attached to the real track. This asks the classifier:
    "Could this theta have produced this track under these start/target
    conditions?"
    """
    if params.size(0) < 2:
        raise ValueError("Batch size must be at least 2.")
    return torch.roll(params, shifts=1, dims=0)


def batch_loss_and_metrics(model, batch, device):
    """Balanced BCE loss: true pairs vs wrong-theta pairs."""
    tracks = batch["track"].to(device)
    params = batch["params"].to(device)
    condition = batch["condition"].to(device)
    neg_params = make_negative_params(params)

    pos_logits = model(tracks, params, condition)
    neg_logits = model(tracks, neg_params, condition)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    targets = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        acc = ((probs >= 0.5).float() == targets).float().mean()
        gap = pos_logits.mean() - neg_logits.mean()
    return loss, {"acc": acc.item(), "log_ratio_gap": gap.item()}


def run_epoch(model, loader, optimizer, device, training: bool):
    model.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()
    totals = {"loss": 0.0, "acc": 0.0, "log_ratio_gap": 0.0}
    n = 0
    with ctx:
        for batch in loader:
            if batch["params"].size(0) < 2:
                continue
            loss, metrics = batch_loss_and_metrics(model, batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            totals["loss"] += loss.item()
            totals["acc"] += metrics["acc"]
            totals["log_ratio_gap"] += metrics["log_ratio_gap"]
            n += 1
    if n == 0:
        raise ValueError("No usable batches.")
    return {k: v / n for k, v in totals.items()}


def main() -> None:
    """
    Train the football equivalent of Lorenz train_ratio_classifier.py.

    The learned logit is later used as a likelihood-ratio surrogate inside
    candidate scoring or random-walk Metropolis-Hastings.
    """
    parser = argparse.ArgumentParser(description="Train football OU conditioned ratio classifier.")
    parser.add_argument("--data-dir", default="data/football_ou_dataset")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FootballOUDataset(args.data_dir)
    val_size = max(2, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    encoder = TrajectoryEncoder(in_channels=2, feature_dim=256)
    model = ConditionedRatioClassifier(encoder, feature_dim=256, param_dim=2, condition_dim=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "football_ou_ratio_history.csv"
    best_path = out_dir / "football_ou_ratio_best.pt"
    best_val = float("inf")

    with open(history_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, ["epoch", "train_loss", "train_acc", "train_gap", "val_loss", "val_acc", "val_gap"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, train_loader, optimizer, device, True)
            val = run_epoch(model, val_loader, None, device, False)
            row = {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_acc": train["acc"],
                "train_gap": train["log_ratio_gap"],
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "val_gap": val["log_ratio_gap"],
            }
            writer.writerow(row)
            fh.flush()
            print(
                f"Epoch {epoch:03d} | train loss {train['loss']:.4f} acc {train['acc']:.2%} "
                f"gap {train['log_ratio_gap']:.3f} | val loss {val['loss']:.4f} "
                f"acc {val['acc']:.2%} gap {val['log_ratio_gap']:.3f}"
            )
            if val["loss"] < best_val:
                best_val = val["loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "track_mean": torch.as_tensor(dataset.track_mean, dtype=torch.float32),
                        "track_std": torch.as_tensor(dataset.track_std, dtype=torch.float32),
                        "param_dim": 2,
                        "condition_dim": 4,
                        "feature_dim": 256,
                        "data_dir": args.data_dir,
                    },
                    best_path,
                )
                print(f"  -> saved {best_path}")


if __name__ == "__main__":
    main()
