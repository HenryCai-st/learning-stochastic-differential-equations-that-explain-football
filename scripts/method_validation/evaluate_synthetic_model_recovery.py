"""
Evaluate model-family recovery on fresh synthetic football-ball trajectories.

Inputs:
    - a trained model-voting ratio-classifier checkpoint
    - the generated training dataset, used only as a pool of pitch conditions

Outputs:
    - summary.json with recovery accuracy, confusion matrix, and model log score
    - confusion_matrix.png
    - cases.npz with true labels, predicted labels, and approximate weights

The trajectories and theta values are newly simulated with a separate seed;
they are not rows copied from the classifier training dataset. This evaluates
model selection, not MCMC parameter-interval calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sbi.artifacts import validate_checkpoint_contract, write_run_metadata
from src.sbi.evidence import logmeanexp, softmax
from src.sbi.scoring import load_checkpoint, normalize_track, score_params
from src.simulators.model_voting import (
    MODEL_NAMES,
    pitch_normalize_condition,
    sample_model_parameters,
    simulate_model_batch,
)


def plot_confusion_matrix(confusion: np.ndarray, out_path: Path) -> None:
    """Render row-normalized true-model versus selected-model frequencies."""
    row_sum = np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    normalized = confusion / row_sum
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    for row in range(len(MODEL_NAMES)):
        for col in range(len(MODEL_NAMES)):
            ax.text(col, row, f"{confusion[row, col]}\n{normalized[row, col]:.0%}", ha="center", va="center")
    ax.set_xticks(range(len(MODEL_NAMES)), MODEL_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(len(MODEL_NAMES)), MODEL_NAMES)
    ax.set_xlabel("selected model")
    ax.set_ylabel("true simulator")
    ax.set_title("Fresh synthetic model-recovery confusion matrix")
    fig.colorbar(image, ax=ax, label="row-normalized frequency")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Simulate fresh cases, estimate evidence weights, and report recovery."""
    parser = argparse.ArgumentParser(description="Evaluate model recovery on fresh synthetic ball tracks.")
    parser.add_argument("--checkpoint", default="checkpoints/model_voting_ratio_best.pt")
    parser.add_argument("--dataset", default="data/model_voting_dataset/dataset.npz")
    parser.add_argument("--n-cases", type=int, default=80)
    parser.add_argument("--n-evidence-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--out-dir", default="outputs/synthetic_model_recovery")
    args = parser.parse_args()

    if args.n_cases < len(MODEL_NAMES):
        raise ValueError(f"--n-cases must be at least {len(MODEL_NAMES)}.")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    dataset = np.load(args.dataset, allow_pickle=True)
    steps = int(dataset["steps"])
    dt = float(dataset["dt"])
    validate_checkpoint_contract(checkpoint, steps=steps, dt=dt)

    true_ids = np.arange(args.n_cases, dtype=np.int64) % len(MODEL_NAMES)
    rng.shuffle(true_ids)
    predicted_ids = np.zeros(args.n_cases, dtype=np.int64)
    weights = np.zeros((args.n_cases, len(MODEL_NAMES)), dtype=np.float32)

    for case_idx, true_id in enumerate(true_ids):
        condition_idx = int(rng.integers(0, len(dataset["y0"])))
        y0 = dataset["y0"][condition_idx].astype(np.float32)
        target = dataset["target"][condition_idx].astype(np.float32)
        change_points = dataset["change_points"][condition_idx].astype(np.int64)
        true_model = MODEL_NAMES[int(true_id)]
        true_theta = sample_model_parameters(true_model, 1, rng)
        track = simulate_model_batch(
            model_name=true_model,
            params=true_theta,
            y0=y0[None],
            target=target[None],
            change_points=change_points[None],
            steps=steps,
            dt=dt,
            rng=rng,
        )[0]
        condition = pitch_normalize_condition(y0, target, change_points, steps)
        track_t = torch.from_numpy(normalize_track(track, checkpoint).T[None]).float().to(device)
        condition_t = torch.from_numpy(condition[None]).float().to(device)

        log_evidence = []
        for candidate_model in MODEL_NAMES:
            theta = sample_model_parameters(candidate_model, args.n_evidence_samples, rng)
            logits = score_params(model, track_t, condition_t, candidate_model, theta, device)
            log_evidence.append(logmeanexp(logits))
        case_weights = softmax(np.asarray(log_evidence, dtype=np.float64))
        weights[case_idx] = case_weights.astype(np.float32)
        predicted_ids[case_idx] = int(np.argmax(case_weights))

    confusion = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)), dtype=np.int64)
    for true_id, predicted_id in zip(true_ids, predicted_ids):
        confusion[true_id, predicted_id] += 1

    true_weights = weights[np.arange(args.n_cases), true_ids]
    accuracy = float(np.mean(predicted_ids == true_ids))
    mean_log_score = float(np.mean(np.log(np.clip(true_weights, 1e-12, 1.0))))
    per_model_accuracy = {}
    for model_id, model_name in enumerate(MODEL_NAMES):
        rows = true_ids == model_id
        per_model_accuracy[model_name] = float(np.mean(predicted_ids[rows] == true_ids[rows]))

    summary = {
        "checkpoint": args.checkpoint,
        "dataset_conditions": args.dataset,
        "fresh_simulation_seed": args.seed,
        "n_cases": args.n_cases,
        "n_evidence_samples": args.n_evidence_samples,
        "accuracy": accuracy,
        "mean_true_model_weight": float(true_weights.mean()),
        "mean_model_log_score": mean_log_score,
        "per_model_accuracy": per_model_accuracy,
        "confusion_matrix_rows_true_columns_selected": confusion.tolist(),
        "status": "synthetic model-recovery diagnostic; does not establish real-data calibration",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "cases.npz",
        true_model_id=true_ids,
        selected_model_id=predicted_ids,
        model_weights=weights,
        model_names=np.asarray(MODEL_NAMES),
    )
    plot_confusion_matrix(confusion, out_dir / "confusion_matrix.png")
    write_run_metadata(
        out_dir / "run_metadata.json",
        stage="synthetic_model_recovery",
        args=args,
        inputs={"checkpoint": args.checkpoint, "dataset": args.dataset},
        outputs={"summary": out_dir / "summary.json", "cases": out_dir / "cases.npz"},
        contract={"steps": steps, "dt": dt},
        results=summary,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved fresh synthetic model-recovery outputs to {out_dir}")


if __name__ == "__main__":
    main()
