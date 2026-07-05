"""
contrastive.py
==============
Contrastive loss functions for SDE trajectory encoder training.

Three options:
  - TripletLoss   : uses explicit (anchor, positive, hard_negative) triplets from
                    the dataset. Best match for our dataloader which already provides
                    hard negatives sorted by rho.
  - InfoNCELoss   : SimCLR-style, constructs negatives implicitly from the batch.
                    Requires large batch sizes (>=128) to work well. Does NOT use
                    the hard negative track from the dataset.
  - SiameseClassifier : binary same/different classifier. Simpler but embedding space
                    is not metric-structured, which hurts downstream parameter inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet margin loss: pulls (anchor, positive) together and pushes
    (anchor, negative) apart by at least `margin`.

    This directly uses the hard negative provided by the dataset (sampled
    from a nearby rho group), making it the natural fit for our dataloader.

    L = max(0, d(a, p) - d(a, n) + margin)

    A margin of 0.5–1.0 works well in normalized embedding space.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")

    def forward(
        self,
        z_anchor: torch.Tensor,    # (batch, emb_dim)
        z_positive: torch.Tensor,  # (batch, emb_dim)
        z_negative: torch.Tensor,  # (batch, emb_dim)
    ) -> torch.Tensor:
        # L2-normalize so distances are in [0, 2] regardless of emb_dim
        z_anchor   = F.normalize(z_anchor,   dim=1)
        z_positive = F.normalize(z_positive, dim=1)
        z_negative = F.normalize(z_negative, dim=1)
        return self.loss_fn(z_anchor, z_positive, z_negative)


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (SimCLR style).

    Each item i in the batch pairs with its positive at index i in z_b.
    All other 2*(batch-1) items act as negatives — so this needs large
    batches (>=128) to provide enough negatives.

    NOTE: does not use the hard negative track from the dataset; the batch
    itself provides all negatives implicitly. If you want to use hard
    negatives explicitly, use TripletLoss instead.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        z_a : (batch_size, emb_dim) — query embeddings
        z_b : (batch_size, emb_dim) — positive embeddings
        Returns scalar loss.
        """
        batch_size = z_a.size(0)

        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

        # Concatenate: (2*B, emb_dim)
        out = torch.cat([z_a, z_b], dim=0)

        # Full cosine similarity matrix: (2*B, 2*B)
        sim_matrix = torch.matmul(out, out.T) / self.temperature

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_a.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Positive target: for row i its positive is i+B (and vice versa)
        targets = torch.arange(2 * batch_size, device=z_a.device)
        targets[:batch_size] += batch_size
        targets[batch_size:] -= batch_size

        return F.cross_entropy(sim_matrix, targets)


class SiameseClassifier(nn.Module):
    """
    Binary same/different classifier on top of a shared encoder.

    Useful as a sanity-check baseline, but the embedding space is not
    explicitly metric-structured, which limits its use for downstream
    parameter inference compared to TripletLoss / InfoNCE.
    """

    def __init__(self, encoder: nn.Module, feature_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        # Input is concat([|z_a - z_b|, z_a * z_b]) → feature_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        """Returns similarity logits of shape (batch_size, 1)."""
        z_a = self.encoder(x_a)
        z_b = self.encoder(x_b)
        features = torch.cat([torch.abs(z_a - z_b), z_a * z_b], dim=1)
        return self.classifier(features)

    def compute_loss(
        self, x_a: torch.Tensor, x_b: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """
        Positive pairs: (x_a_i, x_b_i) → label 1
        Negative pairs: (x_a_i, x_b_{i+1 mod B}) → label 0  (roll by 1)
        Returns (bce_loss, accuracy).
        """
        pos_logits = self.forward(x_a, x_b)
        pos_targets = torch.ones_like(pos_logits)

        x_b_neg = torch.roll(x_b, shifts=1, dims=0)
        neg_logits = self.forward(x_a, x_b_neg)
        neg_targets = torch.zeros_like(neg_logits)

        logits  = torch.cat([pos_logits,  neg_logits],  dim=0)
        targets = torch.cat([pos_targets, neg_targets], dim=0)

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        acc  = ((logits > 0).float() == targets).float().mean().item()
        return loss, acc