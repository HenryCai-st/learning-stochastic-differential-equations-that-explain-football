from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lorenz_dataset import (  # noqa: E402
    LorenzPairDataset,
    LorenzTrajectoryDataset,
    make_subset,
)
from src.models.lorenz_models import (  # noqa: E402
    LorenzParameterRegressor,
    LorenzRatioEstimator,
    LorenzRegimeClassifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Lorenz demo models.")
    parser.add_argument("--dataset", default="lorenz_dataset.npz")
    parser.add_argument("--task", choices=["regime", "params", "ratio"], default="regime")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="outputs/lorenz_training")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_loaders(args: argparse.Namespace):
    train_base = LorenzTrajectoryDataset(
        args.dataset,
        split="train",
        split_seed=args.seed,
        max_points=args.max_points,
    )
    val_base = LorenzTrajectoryDataset(
        args.dataset,
        split="val",
        split_seed=args.seed,
        max_points=args.max_points,
        param_scaler=train_base.param_scaler,
    )

    if args.task == "ratio":
        train_data = LorenzPairDataset(train_base, seed=args.seed)
        val_data = LorenzPairDataset(val_base, seed=args.seed + 1)
    else:
        train_data = train_base
        val_data = val_base

    train_data = make_subset(train_data, args.limit_train, seed=args.seed)
    val_data = make_subset(val_data, args.limit_val, seed=args.seed + 1)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


def build_model_and_loss(task: str, device: str):
    if task == "regime":
        return LorenzRegimeClassifier().to(device), nn.CrossEntropyLoss()
    if task == "params":
        return LorenzParameterRegressor().to(device), nn.L1Loss()
    if task == "ratio":
        return LorenzRatioEstimator().to(device), nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown task: {task}")


def run_epoch(model, loader, loss_fn, optimizer, task: str, device: str) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for batch in loader:
        trajectory = batch["trajectory"].to(device)
        params = batch["params"].to(device)
        label = batch["label"].to(device)

        if task == "regime":
            output = model(trajectory)
            loss = loss_fn(output, label)
            prediction = output.argmax(dim=1)
            total_correct += (prediction == label).sum().item()
            total_items += label.numel()
        elif task == "params":
            output = model(trajectory)
            loss = loss_fn(output, params)
            total_items += params.shape[0]
        else:
            output = model(trajectory, params)
            loss = loss_fn(output, label)
            prediction = (torch.sigmoid(output) >= 0.5).float()
            total_correct += (prediction == label).sum().item()
            total_items += label.numel()

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * trajectory.shape[0]

    metrics = {"loss": total_loss / max(total_items, 1)}
    if task in {"regime", "ratio"}:
        metrics["accuracy"] = total_correct / max(total_items, 1)
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(args)
    model, loss_fn = build_model_and_loss(args.task, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history_path = out_dir / "history.csv"
    best_val = float("inf")

    with open(history_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, train_loader, loss_fn, optimizer, args.task, args.device)
            val = run_epoch(model, val_loader, loss_fn, None, args.task, args.device)

            row = {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_accuracy": train.get("accuracy", ""),
                "val_loss": val["loss"],
                "val_accuracy": val.get("accuracy", ""),
            }
            writer.writerow(row)
            fh.flush()

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train['loss']:.4f} "
                f"val_loss={val['loss']:.4f} "
                f"train_acc={train.get('accuracy', float('nan')):.3f} "
                f"val_acc={val.get('accuracy', float('nan')):.3f}"
            )

            if val["loss"] < best_val:
                best_val = val["loss"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "task": args.task,
                        "args": vars(args),
                    },
                    out_dir / "best.pt",
                )

    print(f"Saved training history to {history_path}")
    print(f"Saved best checkpoint to {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

