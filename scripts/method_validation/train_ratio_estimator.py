"""Train the ratio estimator with independent synthetic train/validation splits."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sbi.artifacts import checkpoint_metadata, dataset_contract, write_run_metadata
from src.sbi.ratio_model import ModelVotingRatioClassifier
from src.sbi.training import run_epoch, validation_metrics_by_model
from src.simulators.model_voting import MODEL_NAMES
from src.synthetic.dataset import ModelVotingDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ratio estimator on the controlled synthetic benchmark.")
    parser.add_argument("--train-data", default="data/method_validation/train.npz")
    parser.add_argument("--validation-data", default="data/method_validation/validation.npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="checkpoints/method_validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ModelVotingDataset(args.train_data)
    validation_dataset = ModelVotingDataset(
        args.validation_data,
        track_mean=train_dataset.track_mean,
        track_std=train_dataset.track_std,
    )
    if dataset_contract(train_dataset)["model_names"] != dataset_contract(validation_dataset)["model_names"]:
        raise ValueError("Train and validation model families differ.")
    if train_dataset.steps != validation_dataset.steps or train_dataset.dt != validation_dataset.dt:
        raise ValueError("Train and validation trajectory contracts differ.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = ModelVotingRatioClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "ratio_estimator_history.csv"
    best_path = out_dir / "ratio_estimator_best.pt"
    best_validation_loss = float("inf")
    best_epoch = 0
    fieldnames = ["epoch", "train_loss", "train_acc", "train_gap", "val_loss", "val_acc", "val_gap"]
    for model_name in MODEL_NAMES:
        fieldnames.extend([f"val_acc_{model_name}", f"val_gap_{model_name}"])

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, train_loader, optimizer, device, training=True)
            validation = run_epoch(model, validation_loader, optimizer, device, training=False)
            per_model = validation_metrics_by_model(model, validation_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_acc": train["acc"],
                "train_gap": train["log_ratio_gap"],
                "val_loss": validation["loss"],
                "val_acc": validation["acc"],
                "val_gap": validation["log_ratio_gap"],
                **per_model,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"Epoch {epoch:03d} | train loss {train['loss']:.4f} acc {train['acc']:.2%} "
                f"gap {train['log_ratio_gap']:.3f} | val loss {validation['loss']:.4f} "
                f"acc {validation['acc']:.2%} gap {validation['log_ratio_gap']:.3f}"
            )
            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "track_mean": torch.as_tensor(train_dataset.track_mean),
                        "track_std": torch.as_tensor(train_dataset.track_std),
                        "model_names": train_dataset.model_names,
                        "train_data": args.train_data,
                        "validation_data": args.validation_data,
                        "best_val_loss": best_validation_loss,
                        "best_epoch": best_epoch,
                        "artifact_metadata": checkpoint_metadata(train_dataset, args),
                    },
                    best_path,
                )
                print(f"  -> saved {best_path}")

    results = {"best_validation_loss": best_validation_loss, "best_epoch": best_epoch, "device": str(device)}
    write_run_metadata(
        out_dir / "run_metadata.json",
        stage="controlled_synthetic_ratio_training",
        args=args,
        inputs={"train": args.train_data, "validation": args.validation_data},
        outputs={"checkpoint": best_path, "history": history_path},
        contract=dataset_contract(train_dataset),
        results=results,
    )
    print(f"Best validation loss {best_validation_loss:.6f} at epoch {best_epoch} on {device}.")


if __name__ == "__main__":
    main()
