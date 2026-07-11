"""
Train the model-voting neural ratio classifier.

Inputs:
    - data/model_voting_dataset/dataset.npz from generate_model_voting_data.py

Outputs:
    - checkpoints/model_voting_ratio_best.pt
    - checkpoints/model_voting_ratio_history.csv with global and per-model
      validation metrics.

Expected use:
    Run this after synthetic model-voting data generation. The trained logit is
    later used as the likelihood-ratio surrogate for MCMC posterior inference.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.model_voting_dataset import ModelVotingDataset
from src.models.model_voting_ratio import ModelVotingRatioClassifier
from src.sde.model_voting import MODEL_NAMES


def roll_negative(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle candidate model/theta while preserving each track's condition."""
    return (
        torch.roll(batch["params"], shifts=1, dims=0),
        torch.roll(batch["param_mask"], shifts=1, dims=0),
        torch.roll(batch["model_id"], shifts=1, dims=0),
    )


def batch_loss_and_metrics(model: ModelVotingRatioClassifier, batch: dict[str, torch.Tensor], device: torch.device):
    """Compute positive/mismatched contrastive loss and global batch metrics."""
    track = batch["track"].to(device)
    params = batch["params"].to(device)
    param_mask = batch["param_mask"].to(device)
    model_id = batch["model_id"].to(device)
    condition = batch["condition"].to(device)
    neg_params, neg_mask, neg_model_id = [x.to(device) for x in roll_negative(batch)]

    pos_logits = model(track, params, param_mask, model_id, condition)
    neg_logits = model(track, neg_params, neg_mask, neg_model_id, condition)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    targets = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        acc = ((probs >= 0.5).float() == targets).float().mean().item()
        gap = (pos_logits.mean() - neg_logits.mean()).item()
    return loss, {"acc": acc, "log_ratio_gap": gap}


def run_epoch(model, loader, optimizer, device, training: bool) -> dict[str, float]:
    """Run one train or validation epoch and return averaged metrics."""
    model.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()
    total = {"loss": 0.0, "acc": 0.0, "log_ratio_gap": 0.0}
    n = 0
    with ctx:
        for batch in loader:
            if batch["track"].size(0) < 2:
                continue
            loss, metrics = batch_loss_and_metrics(model, batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total["loss"] += loss.item()
            total["acc"] += metrics["acc"]
            total["log_ratio_gap"] += metrics["log_ratio_gap"]
            n += 1
    if n == 0:
        raise ValueError("No usable batches; batch size must be at least 2.")
    return {key: value / n for key, value in total.items()}


@torch.no_grad()
def validation_metrics_by_model(model, loader, device) -> dict[str, float]:
    """Report whether matched/mismatched separation works for each model ID."""
    model.eval()
    totals: dict[str, dict[str, float]] = {
        name: {"acc": 0.0, "gap": 0.0, "n": 0.0}
        for name in MODEL_NAMES
    }
    for batch in loader:
        if batch["track"].size(0) < 2:
            continue
        track = batch["track"].to(device)
        params = batch["params"].to(device)
        param_mask = batch["param_mask"].to(device)
        model_id = batch["model_id"].to(device)
        condition = batch["condition"].to(device)
        neg_params, neg_mask, neg_model_id = [x.to(device) for x in roll_negative(batch)]
        pos_logits = model(track, params, param_mask, model_id, condition)
        neg_logits = model(track, neg_params, neg_mask, neg_model_id, condition)
        for mid, name in enumerate(MODEL_NAMES):
            rows = model_id == mid
            if not torch.any(rows):
                continue
            pos = pos_logits[rows]
            neg = neg_logits[rows]
            logits = torch.cat([pos, neg], dim=0)
            targets = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
            acc = ((torch.sigmoid(logits) >= 0.5).float() == targets).float().mean().item()
            gap = (pos.mean() - neg.mean()).item()
            totals[name]["acc"] += acc
            totals[name]["gap"] += gap
            totals[name]["n"] += 1
    out: dict[str, float] = {}
    for name, values in totals.items():
        n = max(values["n"], 1.0)
        out[f"val_acc_{name}"] = values["acc"] / n
        out[f"val_gap_{name}"] = values["gap"] / n
    return out


def main() -> None:
    """Load data, train the classifier, log history, and save the best checkpoint."""
    parser = argparse.ArgumentParser(description="Train model-voting SBI ratio classifier.")
    parser.add_argument("--data-dir", default="data/model_voting_dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ModelVotingDataset(args.data_dir)
    val_size = max(2, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    model = ModelVotingRatioClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "model_voting_ratio_history.csv"
    best_path = out_dir / "model_voting_ratio_best.pt"
    best_val = float("inf")

    fieldnames = ["epoch", "train_loss", "train_acc", "train_gap", "val_loss", "val_acc", "val_gap"]
    for name in MODEL_NAMES:
        fieldnames.extend([f"val_acc_{name}", f"val_gap_{name}"])

    with open(history_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, train_loader, optimizer, device, training=True)
            val = run_epoch(model, val_loader, optimizer, device, training=False)
            per_model = validation_metrics_by_model(model, val_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_acc": train["acc"],
                "train_gap": train["log_ratio_gap"],
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "val_gap": val["log_ratio_gap"],
            }
            row.update(per_model)
            writer.writerow(row)
            fh.flush()
            print(
                f"Epoch {epoch:03d} | train loss {train['loss']:.4f} acc {train['acc']:.2%} "
                f"gap {train['log_ratio_gap']:.3f} | val loss {val['loss']:.4f} "
                f"acc {val['acc']:.2%} gap {val['log_ratio_gap']:.3f}"
            )
            if val["loss"] < best_val:
                best_val = val["loss"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "track_mean": torch.as_tensor(dataset.track_mean, dtype=torch.float32),
                    "track_std": torch.as_tensor(dataset.track_std, dtype=torch.float32),
                    "model_names": dataset.model_names,
                    "data_dir": args.data_dir,
                    "best_val_loss": best_val,
                }, best_path)
                print(f"  -> saved {best_path}")


if __name__ == "__main__":
    main()
