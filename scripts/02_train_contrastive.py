"""
scripts/02_train_contrastive.py
===============================
Part 2 of 4 — Contrastive Learning for the SDE

Trains the neural ratio estimator (NRE) using contrastive learning:
  - Positive pairs:  (θ, x_θ)   — matching parameter/trajectory
  - Negative pairs:  (θ, x_φ)   — mismatched

The trained model learns  logit ≈ log p(x|θ)/p(x),  the likelihood ratio.
Supports three training modes via --task:
  ratio   — BCEWithLogitsLoss on matched/mismatched pairs (main SBI task)
  contrastive — NT-Xent loss via ContrastiveRatioNet.nt_xent_loss
  regime  — CrossEntropy regime classifier (sanity check)
  params  — L1 regression on parameters (deterministic baseline)

Usage:
    python scripts/02_train_contrastive.py --task ratio --epochs 30
    python scripts/02_train_contrastive.py --task contrastive --epochs 30

Outputs:
    outputs/training/<task>/best.pt
    outputs/training/<task>/history.csv
    outputs/training/<task>/training_curves.png
"""

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

from src.data.lorenz_dataset import (
    LorenzPairDataset,
    LorenzTrajectoryDataset,
    make_subset,
)
from src.models.lorenz_models import (
    ContrastiveRatioNet,
    LorenzParameterRegressor,
    LorenzRatioEstimator,
    LorenzRegimeClassifier,
)
from src.utils.plotting import plot_training_curves


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train contrastive ratio estimator.")
    p.add_argument("--dataset",     default=str(ROOT / "data" / "data.npz"))
    p.add_argument("--task",
                   choices=["ratio", "contrastive", "regime", "params"],
                   default="ratio",
                   help="Training objective")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--max-points",  type=int,   default=512)
    p.add_argument("--limit-train", type=int,   default=None)
    p.add_argument("--limit-val",   type=int,   default=None)
    p.add_argument("--hidden-dim",  type=int,   default=128)
    p.add_argument("--temperature", type=float, default=0.1,
                   help="NT-Xent temperature (contrastive task only)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",     default=str(ROOT / "outputs" / "training"))
    p.add_argument("--seed",        type=int,   default=0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / loader builders
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(args: argparse.Namespace):
    train_base = LorenzTrajectoryDataset(
        args.dataset, split="train", split_seed=args.seed,
        max_points=args.max_points,
    )
    val_base = LorenzTrajectoryDataset(
        args.dataset, split="val", split_seed=args.seed,
        max_points=args.max_points,
        param_scaler=train_base.param_scaler,
    )

    if args.task in ("ratio", "contrastive"):
        train_data = LorenzPairDataset(train_base, seed=args.seed)
        val_data   = LorenzPairDataset(val_base,   seed=args.seed + 1)
    else:
        train_data = train_base
        val_data   = val_base

    train_data = make_subset(train_data, args.limit_train, seed=args.seed)
    val_data   = make_subset(val_data,   args.limit_val,   seed=args.seed + 1)

    return (
        DataLoader(train_data, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_data,   batch_size=args.batch_size, shuffle=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model / loss builders
# ─────────────────────────────────────────────────────────────────────────────

def build_model_and_loss(args: argparse.Namespace):
    H  = args.hidden_dim
    dev = args.device

    if args.task == "regime":
        return LorenzRegimeClassifier(hidden_dim=H).to(dev), nn.CrossEntropyLoss()
    if args.task == "params":
        return LorenzParameterRegressor(hidden_dim=H).to(dev), nn.L1Loss()
    if args.task == "ratio":
        return LorenzRatioEstimator(hidden_dim=H).to(dev), nn.BCEWithLogitsLoss()
    if args.task == "contrastive":
        return ContrastiveRatioNet(hidden_dim=H).to(dev), None  # loss computed in loop
    raise ValueError(f"Unknown task: {args.task}")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model, loader, loss_fn, optimizer,
    task: str, device: str, temperature: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss    = 0.0
    total_correct = 0
    total_items   = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            trajectory = batch["trajectory"].to(device)
            params     = batch["params"].to(device)
            label      = batch["label"].to(device)

            if task == "regime":
                output = model(trajectory)
                loss   = loss_fn(output, label)
                pred   = output.argmax(dim=1)
                total_correct += (pred == label).sum().item()
                total_items   += label.numel()

            elif task == "params":
                output = model(trajectory)
                loss   = loss_fn(output, params)
                total_items += params.shape[0]

            elif task == "ratio":
                output = model(trajectory, params)
                loss   = loss_fn(output, label)
                pred   = (torch.sigmoid(output) >= 0.5).float()
                total_correct += (pred == label).sum().item()
                total_items   += label.numel()

            elif task == "contrastive":
                z_t, z_p = model.project(trajectory, params)
                loss      = ContrastiveRatioNet.nt_xent_loss(z_t, z_p, temperature)
                total_items += trajectory.shape[0]

            total_loss += loss.item() * trajectory.shape[0]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    metrics = {"loss": total_loss / max(total_items, 1)}
    if task in ("regime", "ratio"):
        metrics["accuracy"] = total_correct / max(total_items, 1)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[02_train_contrastive] task={args.task}  device={args.device}  "
          f"epochs={args.epochs}")

    train_loader, val_loader = build_loaders(args)
    model, loss_fn           = build_model_and_loss(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history:  list[dict] = []
    best_val  = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, loss_fn, optimizer,
                       args.task, args.device, args.temperature)
        va = run_epoch(model, val_loader,   loss_fn, None,
                       args.task, args.device, args.temperature)

        row = {
            "epoch":          epoch,
            "train_loss":     tr["loss"],
            "train_accuracy": tr.get("accuracy", ""),
            "val_loss":       va["loss"],
            "val_accuracy":   va.get("accuracy", ""),
        }
        history.append(row)
        print(f"  epoch={epoch:03d}  train_loss={tr['loss']:.4f}  "
              f"val_loss={va['loss']:.4f}  "
              f"val_acc={va.get('accuracy', float('nan')):.3f}")

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {"model_state": model.state_dict(),
                 "task":        args.task,
                 "args":        vars(args)},
                out_dir / "best.pt",
            )

    # ── Write CSV history ─────────────────────────────────────────────────────
    csv_path = out_dir / "history.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"[02_train_contrastive] History  {csv_path}")

    # ── Plot training curves ──────────────────────────────────────────────────
    plot_training_curves(history, out_dir / "training_curves.png")
    print(f"[02_train_contrastive] Best checkpoint  {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
