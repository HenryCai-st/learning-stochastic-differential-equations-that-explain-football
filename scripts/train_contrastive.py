"""
train_contrastive.py
====================
Train the 1D CNN trajectory encoder using contrastive learning.

Supports three modes:
  triplet  (default) — uses explicit hard negatives from the dataset.
                       Best match for our dataloader; works at any batch size.
  infonce            — SimCLR-style; negatives come from the batch implicitly.
                       Needs large batch size (>=128) to work well.
  siamese            — binary BCE classifier. Sanity-check baseline.

Dataset returns: (query, positive, negative, params)  all shape (3, steps) or (4,)
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import SDEDataset
from src.models.encoder import TrajectoryEncoder, ProjectionHead
from src.models.contrastive import TripletLoss, InfoNCELoss, SiameseClassifier


def train_one_epoch(mode, encoder, proj_head, siamese_net, criterion, optimizer, loader, device):
    encoder.train()
    if mode == "infonce":
        proj_head.train()
    elif mode == "siamese":
        siamese_net.train()

    total_loss, total_acc, n = 0.0, 0.0, 0

    for query_t, positive_t, negative_t, _ in loader:
        query_t    = query_t.to(device)     # (B, 3, steps)
        positive_t = positive_t.to(device)
        negative_t = negative_t.to(device)

        optimizer.zero_grad()

        if mode == "triplet":
            z_q = encoder(query_t)
            z_p = encoder(positive_t)
            z_n = encoder(negative_t)
            loss = criterion(z_q, z_p, z_n)

        elif mode == "infonce":
            # InfoNCE only uses query and positive; batch provides negatives implicitly
            z_a = proj_head(encoder(query_t))
            z_b = proj_head(encoder(positive_t))
            loss = criterion(z_a, z_b)

        else:  # siamese
            loss, acc = siamese_net.compute_loss(query_t, positive_t)
            total_acc += acc

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n, (total_acc / n) if mode == "siamese" else None


@torch.no_grad()
def eval_one_epoch(mode, encoder, proj_head, siamese_net, criterion, loader, device):
    encoder.eval()
    if mode == "infonce":
        proj_head.eval()
    elif mode == "siamese":
        siamese_net.eval()

    total_loss, total_acc, n = 0.0, 0.0, 0

    for query_t, positive_t, negative_t, _ in loader:
        query_t    = query_t.to(device)
        positive_t = positive_t.to(device)
        negative_t = negative_t.to(device)

        if mode == "triplet":
            z_q = encoder(query_t)
            z_p = encoder(positive_t)
            z_n = encoder(negative_t)
            loss = criterion(z_q, z_p, z_n)

        elif mode == "infonce":
            z_a = proj_head(encoder(query_t))
            z_b = proj_head(encoder(positive_t))
            loss = criterion(z_a, z_b)

        else:  # siamese
            loss, acc = siamese_net.compute_loss(query_t, positive_t)
            total_acc += acc

        total_loss += loss.item()
        n += 1

    return total_loss / n, (total_acc / n) if mode == "siamese" else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,   default="./data/lorenz_dataset")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64,
                        help="Use >=128 for infonce; triplet works at any size")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--temp",       type=float, default=0.1,
                        help="Temperature for InfoNCE (ignored for triplet/siamese)")
    parser.add_argument("--margin",     type=float, default=1.0,
                        help="Margin for TripletLoss (ignored for infonce/siamese)")
    parser.add_argument("--mode",       type=str,   default="triplet",
                        choices=["triplet", "infonce", "siamese"])
    parser.add_argument("--out_dir",    type=str,   default="./checkpoints")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--hard_neg_window", type=int, default=5,
                        help="Rho-rank window for hard negative sampling in dataset")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode} | Batch: {args.batch_size}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    try:
        dataset = SDEDataset(args.data_dir, hard_neg_window=args.hard_neg_window)
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

    # ── Models ───────────────────────────────────────────────────────────────
    encoder     = TrajectoryEncoder(feature_dim=256).to(device)
    proj_head   = None
    siamese_net = None

    if args.mode == "triplet":
        criterion = TripletLoss(margin=args.margin)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    elif args.mode == "infonce":
        proj_head = ProjectionHead(input_dim=256, proj_dim=128).to(device)
        criterion = InfoNCELoss(temperature=args.temp)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(proj_head.parameters()), lr=args.lr
        )

    else:  # siamese
        siamese_net = SiameseClassifier(encoder, feature_dim=256).to(device)
        criterion   = None  # loss computed inside compute_loss
        optimizer   = torch.optim.Adam(siamese_net.parameters(), lr=args.lr)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"Training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            args.mode, encoder, proj_head, siamese_net, criterion, optimizer, train_loader, device
        )
        val_loss, val_acc = eval_one_epoch(
            args.mode, encoder, proj_head, siamese_net, criterion, val_loader, device
        )

        if args.mode == "siamese":
            print(f"Epoch {epoch:02d}/{args.epochs} | "
                  f"Train {tr_loss:.4f} (acc {tr_acc:.2%}) | "
                  f"Val {val_loss:.4f} (acc {val_acc:.2%})")
        else:
            print(f"Epoch {epoch:02d}/{args.epochs} | "
                  f"Train {tr_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = out_dir / f"encoder_best_{args.mode}.pt"
            torch.save(encoder.state_dict(), ckpt)
            print(f"  → Saved {ckpt}")

    print("Pretraining complete.")


if __name__ == "__main__":
    main()