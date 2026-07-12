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
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.synthetic.dataset import ModelVotingDataset
from src.sbi.artifacts import checkpoint_metadata, dataset_contract, write_run_metadata
from src.sbi.ratio_model import ModelVotingRatioClassifier
from src.sbi.training import run_epoch, validation_metrics_by_model
from src.simulators.model_voting import MODEL_NAMES


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
                    "artifact_metadata": checkpoint_metadata(dataset, args),
                }, best_path)
                print(f"  -> saved {best_path}")

    write_run_metadata(
        out_dir / "run_metadata.json",
        stage="ratio_classifier_training",
        args=args,
        inputs={"dataset": Path(args.data_dir) / "dataset.npz"},
        outputs={"checkpoint": best_path, "history": history_path},
        contract=dataset_contract(dataset),
        results={"best_val_loss": best_val},
    )


if __name__ == "__main__":
    main()
