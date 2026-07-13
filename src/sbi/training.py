"""Shared contrastive training helpers for model-voting ratio estimation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.sbi.ratio_model import ModelVotingRatioClassifier
from src.simulators.model_voting import MODEL_NAMES


def roll_negative(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle candidate model/theta while preserving each track's condition."""
    return (
        torch.roll(batch["params"], shifts=1, dims=0),
        torch.roll(batch["param_mask"], shifts=1, dims=0),
        torch.roll(batch["model_id"], shifts=1, dims=0),
    )


def batch_loss_and_metrics(
    model: ModelVotingRatioClassifier,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
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
        accuracy = ((probs >= 0.5).float() == targets).float().mean().item()
        gap = (pos_logits.mean() - neg_logits.mean()).item()
    return loss, {"acc": accuracy, "log_ratio_gap": gap}


def run_epoch(model, loader, optimizer, device, training: bool) -> dict[str, float]:
    """Run one train or validation epoch and return averaged metrics."""
    model.train(training)
    context = torch.enable_grad() if training else torch.no_grad()
    total = {"loss": 0.0, "acc": 0.0, "log_ratio_gap": 0.0}
    n_batches = 0
    with context:
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
            n_batches += 1
    if n_batches == 0:
        raise ValueError("No usable batches; batch size must be at least 2.")
    return {key: value / n_batches for key, value in total.items()}


@torch.no_grad()
def validation_metrics_by_model(model, loader, device) -> dict[str, float]:
    """Report matched/mismatched separation for each model family."""
    model.eval()
    totals = {name: {"acc": 0.0, "gap": 0.0, "n": 0.0} for name in MODEL_NAMES}
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
        for model_index, model_name in enumerate(MODEL_NAMES):
            rows = model_id == model_index
            if not torch.any(rows):
                continue
            pos = pos_logits[rows]
            neg = neg_logits[rows]
            logits = torch.cat([pos, neg])
            targets = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
            accuracy = ((torch.sigmoid(logits) >= 0.5).float() == targets).float().mean().item()
            totals[model_name]["acc"] += accuracy
            totals[model_name]["gap"] += (pos.mean() - neg.mean()).item()
            totals[model_name]["n"] += 1

    output: dict[str, float] = {}
    for model_name, values in totals.items():
        count = max(values["n"], 1.0)
        output[f"val_acc_{model_name}"] = values["acc"] / count
        output[f"val_gap_{model_name}"] = values["gap"] / count
    return output
