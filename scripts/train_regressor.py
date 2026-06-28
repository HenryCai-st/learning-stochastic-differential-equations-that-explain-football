"""
train_regressor.py
==================
Train the probabilistic parameter regressor on top of a pretrained encoder.

Phase 2 of the pipeline:
  - Load encoder pretrained via contrastive learning (triplet by default)
  - Optionally freeze encoder weights
  - Train MLP head to output (mean, logvar) over normalized θ
  - Loss: Gaussian NLL (trains both accuracy and uncertainty calibration)
  - Metrics: NLL (objective), MAE in normalized space, MAE in physical space
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import SDEDataset
from src.models.encoder import TrajectoryEncoder, ProbabilisticRegressor


def run_epoch(regressor, loader, optimizer, normalizer, device, training: bool):
    """
    One training or validation epoch.
    Returns (mean_nll, mean_mae_normalized, mean_mae_physical)
    """
    regressor.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()

    total_nll, total_mae_norm, total_mae_phys, n = 0.0, 0.0, 0.0, 0

    with ctx:
        for query_t, _, _, params_t in loader:
            query_t  = query_t.to(device)   # (B, 3, steps)
            params_t = params_t.to(device)  # (B, 4) normalized

            mean, logvar = regressor(query_t)

            # Gaussian NLL in normalized space (our training objective)
            nll = ProbabilisticRegressor.gaussian_nll_loss(mean, logvar, params_t)

            if training:
                optimizer.zero_grad()
                nll.backward()
                optimizer.step()

            # MAE in normalized space (quick sanity check)
            mae_norm = (mean - params_t).abs().mean()

            # MAE in physical space (interpretable metric)
            # Detach and move to cpu for numpy-based denormalization
            mean_phys   = normalizer.denormalize(mean.detach().cpu().numpy())
            target_phys = normalizer.denormalize(params_t.detach().cpu().numpy())
            mae_phys    = abs(mean_phys - target_phys).mean()

            total_nll      += nll.item()
            total_mae_norm += mae_norm.item()
            total_mae_phys += mae_phys
            n += 1

    return total_nll / n, total_mae_norm / n, total_mae_phys / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      type=str,   default="./data/lorenz_dataset")
    parser.add_argument("--encoder_ckpt",  type=str,   default="./checkpoints/encoder_best_triplet.pt")
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder; only train the MLP head")
    parser.add_argument("--out_dir",       type=str,   default="./checkpoints")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Freeze encoder: {args.freeze_encoder}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    try:
        dataset = SDEDataset(args.data_dir)
    except FileNotFoundError:
        print("Dataset not found. Run scripts/generate_data.py first.")
        sys.exit(1)

    val_size   = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder = TrajectoryEncoder(feature_dim=256)

    ckpt_path = Path(args.encoder_ckpt)
    if ckpt_path.exists():
        encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded pretrained encoder from {ckpt_path}")
    else:
        print(f"Warning: encoder checkpoint not found at {ckpt_path}. Training from scratch.")

    # ProbabilisticRegressor handles freezing internally
    regressor = ProbabilisticRegressor(
        encoder,
        feature_dim=256,
        out_dim=4,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    n_trainable = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optim_params = [p for p in regressor.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(optim_params, lr=args.lr)
    # Halve LR when val NLL stops improving for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Normalizer needed for physical-space MAE
    normalizer = dataset.normalizer

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_nll = float("inf")
    print("Starting regressor training...")

    for epoch in range(1, args.epochs + 1):
        tr_nll, tr_mae_n, tr_mae_p = run_epoch(
            regressor, train_loader, optimizer, normalizer, device, training=True
        )
        val_nll, val_mae_n, val_mae_p = run_epoch(
            regressor, val_loader, optimizer, normalizer, device, training=False
        )

        scheduler.step(val_nll)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train NLL {tr_nll:.4f}  MAE(norm) {tr_mae_n:.4f}  MAE(phys) {tr_mae_p:.4f} | "
            f"Val   NLL {val_nll:.4f}  MAE(norm) {val_mae_n:.4f}  MAE(phys) {val_mae_p:.4f}"
        )

        # Save on best val NLL (objective includes uncertainty calibration)
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            ckpt = out_dir / "regressor_best.pt"
            torch.save(regressor.state_dict(), ckpt)
            print(f"  → Saved {ckpt}")

    print("Regressor training complete.")


if __name__ == "__main__":
    main()