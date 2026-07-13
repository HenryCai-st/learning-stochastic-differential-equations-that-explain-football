"""Evaluate model-family recovery on an independent synthetic test split."""

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
from src.simulators.model_voting import MODEL_NAMES, sample_model_parameters


def plot_confusion_matrix(confusion: np.ndarray, out_path: Path) -> None:
    """Render row-normalized true-model versus selected-model frequencies."""
    row_sum = np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    normalized = confusion / row_sum
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    for row in range(len(MODEL_NAMES)):
        for column in range(len(MODEL_NAMES)):
            ax.text(
                column,
                row,
                f"{confusion[row, column]}\n{normalized[row, column]:.0%}",
                ha="center",
                va="center",
            )
    ax.set_xticks(range(len(MODEL_NAMES)), MODEL_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(len(MODEL_NAMES)), MODEL_NAMES)
    ax.set_xlabel("selected model")
    ax.set_ylabel("true simulator")
    ax.set_title("Independent synthetic model-recovery confusion matrix")
    fig.colorbar(image, ax=ax, label="row-normalized frequency")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def select_balanced_indices(model_ids: np.ndarray, n_cases: int, rng: np.random.Generator) -> np.ndarray:
    """Select an equal number of held-out rows from every model family."""
    if n_cases == 0:
        return np.arange(len(model_ids), dtype=np.int64)
    if n_cases < len(MODEL_NAMES) or n_cases % len(MODEL_NAMES) != 0:
        raise ValueError(f"--n-cases must be 0 or a positive multiple of {len(MODEL_NAMES)}.")
    per_model = n_cases // len(MODEL_NAMES)
    selected = []
    for model_id in range(len(MODEL_NAMES)):
        available = np.flatnonzero(model_ids == model_id)
        if len(available) < per_model:
            raise ValueError(f"Test split has only {len(available)} rows for {MODEL_NAMES[model_id]}.")
        selected.append(rng.choice(available, size=per_model, replace=False))
    indices = np.concatenate(selected).astype(np.int64)
    rng.shuffle(indices)
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model recovery on an independent synthetic test split.")
    parser.add_argument("--checkpoint", default="checkpoints/method_validation/ratio_estimator_best.pt")
    parser.add_argument("--test-data", default="data/method_validation/test.npz")
    parser.add_argument("--n-cases", type=int, default=0, help="0 evaluates the complete test split.")
    parser.add_argument("--n-evidence-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out-dir", default="outputs/method_validation/model_recovery")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    test_data = np.load(args.test_data, allow_pickle=True)
    steps = int(test_data["steps"])
    dt = float(test_data["dt"])
    validate_checkpoint_contract(checkpoint, steps=steps, dt=dt)

    all_true_ids = test_data["model_id"].astype(np.int64)
    selected_indices = select_balanced_indices(all_true_ids, args.n_cases, rng)
    true_ids = all_true_ids[selected_indices]
    predicted_ids = np.zeros(len(selected_indices), dtype=np.int64)
    weights = np.zeros((len(selected_indices), len(MODEL_NAMES)), dtype=np.float32)

    for case_number, row_index in enumerate(selected_indices):
        track = test_data["tracks"][row_index].astype(np.float32)
        condition = test_data["conditions"][row_index].astype(np.float32)
        track_tensor = torch.from_numpy(normalize_track(track, checkpoint).T[None]).to(device)
        condition_tensor = torch.from_numpy(condition[None]).to(device)

        log_evidence = []
        for candidate_model in MODEL_NAMES:
            theta = sample_model_parameters(candidate_model, args.n_evidence_samples, rng)
            logits = score_params(model, track_tensor, condition_tensor, candidate_model, theta, device)
            log_evidence.append(logmeanexp(logits))
        case_weights = softmax(np.asarray(log_evidence, dtype=np.float64))
        weights[case_number] = case_weights.astype(np.float32)
        predicted_ids[case_number] = int(np.argmax(case_weights))

    confusion = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)), dtype=np.int64)
    for true_id, predicted_id in zip(true_ids, predicted_ids):
        confusion[true_id, predicted_id] += 1

    true_weights = weights[np.arange(len(true_ids)), true_ids]
    accuracy = float(np.mean(predicted_ids == true_ids))
    per_model_accuracy = {
        model_name: float(np.mean(predicted_ids[true_ids == model_id] == model_id))
        for model_id, model_name in enumerate(MODEL_NAMES)
    }
    summary = {
        "checkpoint": args.checkpoint,
        "test_data": args.test_data,
        "test_split_seed": int(test_data["seed"]),
        "evaluation_seed": args.seed,
        "n_cases": int(len(selected_indices)),
        "n_evidence_samples": args.n_evidence_samples,
        "accuracy": accuracy,
        "mean_true_model_weight": float(true_weights.mean()),
        "mean_model_log_score": float(np.mean(np.log(np.clip(true_weights, 1e-12, 1.0)))),
        "per_model_accuracy": per_model_accuracy,
        "confusion_matrix_rows_true_columns_selected": confusion.tolist(),
        "status": "independent synthetic model-recovery benchmark; parameter and forecast displays are not included",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    cases_path = out_dir / "cases.npz"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        cases_path,
        test_row_index=selected_indices,
        true_model_id=true_ids,
        selected_model_id=predicted_ids,
        model_weights=weights,
        true_parameters=test_data["parameters"][selected_indices],
        parameter_mask=test_data["parameter_mask"][selected_indices],
        model_names=np.asarray(MODEL_NAMES),
    )
    plot_confusion_matrix(confusion, out_dir / "confusion_matrix.png")
    write_run_metadata(
        out_dir / "run_metadata.json",
        stage="independent_synthetic_model_recovery",
        args=args,
        inputs={"checkpoint": args.checkpoint, "test_data": args.test_data},
        outputs={"summary": summary_path, "cases": cases_path},
        contract={"steps": steps, "dt": dt, "n_test_rows": len(all_true_ids)},
        results=summary,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved independent synthetic model-recovery outputs to {out_dir}")


if __name__ == "__main__":
    main()
