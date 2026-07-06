from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.model_voting_ratio import ModelVotingRatioClassifier
from src.sde.model_voting import (
    MODEL_NAMES,
    MODEL_TO_ID,
    normalize_padded_parameters,
    pad_parameters,
    sample_model_parameters,
)


def load_checkpoint(path: Path, device: torch.device) -> tuple[ModelVotingRatioClassifier, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run train_model_voting_ratio.py first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ModelVotingRatioClassifier()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def checkpoint_array(ckpt: dict, key: str) -> np.ndarray:
    value = ckpt[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    m = np.max(values)
    return float(m + np.log(np.mean(np.exp(values - m))))


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


@torch.no_grad()
def score_model_candidates(
    model: ModelVotingRatioClassifier,
    track_t: torch.Tensor,
    condition_t: torch.Tensor,
    model_name: str,
    n_candidates: int,
    rng: np.random.Generator,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    params = sample_model_parameters(model_name, n_candidates, rng)
    padded, mask = pad_parameters(model_name, params)
    params_norm = normalize_padded_parameters(model_name, padded)
    model_id = MODEL_TO_ID[model_name]
    outputs: list[np.ndarray] = []

    for start in range(0, n_candidates, batch_size):
        end = min(start + batch_size, n_candidates)
        n = end - start
        logits = model(
            track_t.repeat(n, 1, 1),
            torch.from_numpy(params_norm[start:end]).to(device),
            torch.from_numpy(mask[start:end]).to(device),
            torch.full((n,), model_id, dtype=torch.long, device=device),
            condition_t.repeat(n, 1),
        )
        outputs.append(logits.cpu().numpy())

    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synthetic known-model recovery for model-voting SBI.")
    parser.add_argument("--data-dir", default="data/model_voting_dataset_test")
    parser.add_argument("--checkpoint", default="checkpoints_test/model_voting_ratio_best.pt")
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-candidates", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="outputs/model_voting_synthetic_eval")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(Path(args.checkpoint), device)

    data_path = Path(args.data_dir) / "dataset.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run generate_model_voting_data.py first.")
    data = np.load(data_path, allow_pickle=True)
    tracks = data["tracks"].astype(np.float32)
    conditions = data["conditions"].astype(np.float32)
    true_model_ids = data["model_id"].astype(np.int64)
    model_names = [str(name) for name in data["model_names"].tolist()]

    track_mean = checkpoint_array(ckpt, "track_mean").astype(np.float32)
    track_std = checkpoint_array(ckpt, "track_std").astype(np.float32)
    track_std = np.where(track_std < 1e-8, 1.0, track_std)

    n_eval = min(args.n_eval, len(tracks))
    indices = rng.choice(len(tracks), size=n_eval, replace=False)
    confusion = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)), dtype=np.int64)
    rows: list[dict[str, object]] = []
    correct = 0
    top2_correct = 0
    true_vote_weights: list[float] = []
    vote_entropies: list[float] = []
    per_model_seen = np.zeros(len(MODEL_NAMES), dtype=np.int64)
    per_model_correct = np.zeros(len(MODEL_NAMES), dtype=np.int64)

    for sample_index in indices:
        track_norm = ((tracks[sample_index] - track_mean) / track_std).astype(np.float32)
        track_t = torch.from_numpy(track_norm.T[None]).float().to(device)
        condition_t = torch.from_numpy(conditions[sample_index][None]).float().to(device)

        scores = []
        for model_name in MODEL_NAMES:
            logits = score_model_candidates(
                model=model,
                track_t=track_t,
                condition_t=condition_t,
                model_name=model_name,
                n_candidates=args.n_candidates,
                rng=rng,
                device=device,
            )
            scores.append(logmeanexp(logits))

        scores_arr = np.asarray(scores, dtype=np.float64)
        votes = softmax(scores_arr)
        pred_id = int(np.argmax(votes))
        true_id = int(true_model_ids[sample_index])
        true_name = model_names[true_id]
        pred_name = MODEL_NAMES[pred_id]
        is_correct = pred_id == true_id
        top2 = set(np.argsort(-votes)[:2].tolist())

        correct += int(is_correct)
        top2_correct += int(true_id in top2)
        true_vote_weights.append(float(votes[true_id]))
        vote_entropies.append(entropy(votes))
        confusion[true_id, pred_id] += 1
        per_model_seen[true_id] += 1
        per_model_correct[true_id] += int(is_correct)

        row: dict[str, object] = {
            "sample_index": int(sample_index),
            "true_model": true_name,
            "pred_model": pred_name,
            "correct": int(is_correct),
        }
        for model_name, vote in zip(MODEL_NAMES, votes):
            row[f"vote_{model_name}"] = float(vote)
        for model_name, score in zip(MODEL_NAMES, scores_arr):
            row[f"score_{model_name}"] = float(score)
        rows.append(row)

    per_model_accuracy = {
        model_name: (None if per_model_seen[i] == 0 else float(per_model_correct[i] / per_model_seen[i]))
        for i, model_name in enumerate(MODEL_NAMES)
    }
    summary = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "n_eval": int(n_eval),
        "n_candidates": int(args.n_candidates),
        "seed": int(args.seed),
        "top1_model_accuracy": float(correct / max(1, n_eval)),
        "top2_model_accuracy": float(top2_correct / max(1, n_eval)),
        "mean_vote_weight_for_true_model": float(np.mean(true_vote_weights)),
        "mean_entropy_of_model_votes": float(np.mean(vote_entropies)),
        "per_model_accuracy": per_model_accuracy,
        "model_names": list(MODEL_NAMES),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = [
        "sample_index",
        "true_model",
        "pred_model",
        "correct",
        *[f"vote_{model_name}" for model_name in MODEL_NAMES],
        *[f"score_{model_name}" for model_name in MODEL_NAMES],
    ]
    with open(out_dir / "per_sample_votes.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "confusion_matrix.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true_by_pred", *MODEL_NAMES])
        for model_name, counts in zip(MODEL_NAMES, confusion):
            writer.writerow([model_name, *counts.tolist()])

    print(json.dumps(summary, indent=2))
    print(f"Saved synthetic model-voting evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
