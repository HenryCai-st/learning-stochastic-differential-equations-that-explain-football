"""Validate known-model parameter posteriors on independent synthetic cases."""

from __future__ import annotations

import argparse
import csv
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
from src.sbi.diagnostics import effective_sample_size, interval_summary, split_rhat
from src.sbi.mcmc import log_prior_batch, run_batched_model_mcmc
from src.sbi.scoring import load_checkpoint, normalize_track, score_aligned_params
from src.simulators.model_voting import MODEL_NAMES, MODEL_PARAMETER_NAMES, MODEL_SPECS, sample_model_parameters


def plot_coverage(rows: list[dict[str, object]], out_path: Path) -> None:
    labels = []
    values = {50: [], 80: [], 90: []}
    for model_name in MODEL_NAMES:
        for parameter in MODEL_PARAMETER_NAMES[model_name]:
            subset = [row for row in rows if row["model"] == model_name and row["parameter"] == parameter]
            labels.append(f"{model_name}\n{parameter}")
            for level in values:
                values[level].append(np.mean([float(row[f"covered_{level}"]) for row in subset]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for offset, (level, coverage) in zip((-0.22, 0.0, 0.22), values.items()):
        ax.bar(x + offset, coverage, width=0.2, label=f"{level}% interval")
        ax.axhline(level / 100.0, color="#777777", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("empirical coverage")
    ax.set_title("Known-model parameter posterior coverage")
    ax.legend(ncols=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_examples(payload: dict[str, np.ndarray], out_path: Path) -> None:
    max_parameters = max(spec.param_dim for spec in MODEL_SPECS.values())
    fig, axes = plt.subplots(len(MODEL_NAMES), max_parameters, figsize=(3.0 * max_parameters, 10.5))
    for model_index, model_name in enumerate(MODEL_NAMES):
        samples = payload[f"{model_name}_samples"][0].reshape(-1, MODEL_SPECS[model_name].param_dim)
        truth = payload[f"{model_name}_true_theta"][0]
        for dim in range(max_parameters):
            ax = axes[model_index, dim]
            if dim >= MODEL_SPECS[model_name].param_dim:
                ax.axis("off")
                continue
            ax.hist(samples[:, dim], bins=30, color="#4c78a8", alpha=0.85)
            ax.axvline(truth[dim], color="#d62728", linewidth=2, label="truth")
            ax.set_title(f"{model_name}: {MODEL_PARAMETER_NAMES[model_name][dim]}", fontsize=9)
            ax.grid(alpha=0.15)
    axes[0, 0].legend()
    fig.suptitle("Example known-model parameter posteriors", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def initialize_chains(
    model,
    tracks_t: torch.Tensor,
    conditions_t: torch.Tensor,
    model_name: str,
    n_candidates: int,
    n_chains: int,
    rng: np.random.Generator,
    device: torch.device,
) -> np.ndarray:
    """Choose high-target prior candidates independently for each case."""
    n_cases = len(tracks_t)
    candidates = sample_model_parameters(model_name, n_cases * n_candidates, rng).reshape(
        n_cases, n_candidates, MODEL_SPECS[model_name].param_dim
    )
    logits = score_aligned_params(
        model,
        tracks_t.repeat_interleave(n_candidates, dim=0),
        conditions_t.repeat_interleave(n_candidates, dim=0),
        model_name,
        candidates.reshape(-1, candidates.shape[-1]),
        device,
    ).reshape(n_cases, n_candidates)
    target = logits + log_prior_batch(model_name, candidates)
    best = np.argpartition(target, -n_chains, axis=1)[:, -n_chains:]
    return candidates[np.arange(n_cases)[:, None], best]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate known-model parameter posterior recovery.")
    parser.add_argument("--checkpoint", default="checkpoints/method_validation/ratio_estimator_best.pt")
    parser.add_argument("--test-data", default="data/method_validation/test.npz")
    parser.add_argument("--cases-per-model", type=int, default=25)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--init-candidates", type=int, default=64)
    parser.add_argument("--mcmc-steps", type=int, default=1200)
    parser.add_argument("--burn-in", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--out-dir", default="outputs/method_validation/parameter_recovery")
    args = parser.parse_args()
    if args.init_candidates < args.chains:
        raise ValueError("--init-candidates must be at least --chains.")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    test = np.load(args.test_data, allow_pickle=True)
    validate_checkpoint_contract(checkpoint, steps=int(test["steps"]), dt=float(test["dt"]))

    metric_rows: list[dict[str, object]] = []
    samples_payload: dict[str, np.ndarray] = {}
    model_summary: dict[str, object] = {}
    for model_id, model_name in enumerate(MODEL_NAMES):
        available = np.flatnonzero(test["model_id"] == model_id)
        if len(available) < args.cases_per_model:
            raise ValueError(f"Only {len(available)} test rows are available for {model_name}.")
        indices = rng.choice(available, size=args.cases_per_model, replace=False)
        tracks = np.stack([normalize_track(test["tracks"][index], checkpoint).T for index in indices])
        conditions = test["conditions"][indices].astype(np.float32)
        spec = MODEL_SPECS[model_name]
        true_theta = test["parameters"][indices, :spec.param_dim].astype(np.float32)
        tracks_t = torch.from_numpy(tracks).float().to(device)
        conditions_t = torch.from_numpy(conditions).float().to(device)
        initial = initialize_chains(
            model,
            tracks_t,
            conditions_t,
            model_name,
            args.init_candidates,
            args.chains,
            rng,
            device,
        )
        result = run_batched_model_mcmc(
            model=model,
            tracks_t=tracks_t,
            conditions_t=conditions_t,
            model_name=model_name,
            initial_theta=initial,
            n_steps=args.mcmc_steps,
            burn_in=args.burn_in,
            rng=rng,
            device=device,
        )
        samples = result["samples"]
        acceptance = result["acceptance_rate"]
        samples_payload[f"{model_name}_test_indices"] = indices
        samples_payload[f"{model_name}_true_theta"] = true_theta
        samples_payload[f"{model_name}_samples"] = samples
        samples_payload[f"{model_name}_acceptance_rate"] = acceptance

        for case_number, test_index in enumerate(indices):
            case_samples = samples[case_number]
            intervals = interval_summary(case_samples, true_theta[case_number])
            rhat = split_rhat(case_samples)
            ess = effective_sample_size(case_samples)
            for dim, parameter_name in enumerate(MODEL_PARAMETER_NAMES[model_name]):
                row: dict[str, object] = {
                    "model": model_name,
                    "parameter": parameter_name,
                    "test_index": int(test_index),
                    "true_value": float(true_theta[case_number, dim]),
                    "posterior_mean": float(intervals["mean"][dim]),
                    "bias": float(intervals["bias"][dim]),
                    "absolute_error": float(abs(intervals["bias"][dim])),
                    "rhat": float(rhat[dim]),
                    "ess": float(ess[dim]),
                    "acceptance_rate": float(acceptance[case_number].mean()),
                }
                for level in (50, 80, 90):
                    row[f"covered_{level}"] = int(intervals[f"covered_{level}"][dim])
                    row[f"width_{level}"] = float(intervals[f"width_{level}"][dim])
                metric_rows.append(row)

        model_rows = [row for row in metric_rows if row["model"] == model_name]
        model_summary[model_name] = {
            "n_cases": args.cases_per_model,
            "mean_absolute_error": float(np.mean([row["absolute_error"] for row in model_rows])),
            "coverage_50": float(np.mean([row["covered_50"] for row in model_rows])),
            "coverage_80": float(np.mean([row["covered_80"] for row in model_rows])),
            "coverage_90": float(np.mean([row["covered_90"] for row in model_rows])),
            "median_ess": float(np.median([row["ess"] for row in model_rows])),
            "max_rhat": float(np.max([row["rhat"] for row in model_rows])),
            "mean_acceptance_rate": float(np.mean(acceptance)),
        }
        print(f"Completed {model_name}: {model_summary[model_name]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "case_parameter_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    samples_path = out_dir / "posterior_samples.npz"
    np.savez_compressed(samples_path, model_names=np.asarray(MODEL_NAMES), **samples_payload)
    plot_coverage(metric_rows, out_dir / "coverage_by_parameter.png")
    plot_examples(samples_payload, out_dir / "example_posteriors.png")

    overall = {
        "n_cases": args.cases_per_model * len(MODEL_NAMES),
        "coverage_50": float(np.mean([row["covered_50"] for row in metric_rows])),
        "coverage_80": float(np.mean([row["covered_80"] for row in metric_rows])),
        "coverage_90": float(np.mean([row["covered_90"] for row in metric_rows])),
        "median_ess": float(np.median([row["ess"] for row in metric_rows])),
        "fraction_rhat_below_1_05": float(np.mean([row["rhat"] < 1.05 for row in metric_rows])),
    }
    summary = {
        "checkpoint": args.checkpoint,
        "test_data": args.test_data,
        "conditioning": "true_model_known",
        "overall": overall,
        "by_model": model_summary,
        "interpretation": "Parameter coverage is conditional on the true model family.",
        "diagnostic_status": (
            "convergence_acceptable"
            if overall["fraction_rhat_below_1_05"] >= 0.9
            else "convergence_not_established"
        ),
        "caution": "Coverage must not be treated as reliable where R-hat or ESS diagnostics fail.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_metadata(
        out_dir / "run_metadata.json",
        stage="known_model_parameter_recovery",
        args=args,
        inputs={"checkpoint": args.checkpoint, "test_data": args.test_data},
        outputs={"summary": summary_path, "metrics": csv_path, "samples": samples_path},
        contract={"steps": int(test["steps"]), "dt": float(test["dt"])},
        results=summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
