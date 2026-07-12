"""Evaluate controlled posterior predictive forecasts and simple baselines."""

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
from src.sbi.evidence import logmeanexp, softmax
from src.sbi.forecasting import ade_fde, deterministic_baselines, energy_score, radial_coverage, simulate_future_batch
from src.sbi.scoring import load_checkpoint, normalize_track, score_params
from src.simulators.model_voting import MAX_PARAM_DIM, MODEL_NAMES, MODEL_SPECS, sample_model_parameters
from src.simulators.ou import PITCH_LENGTH, PITCH_WIDTH


def balanced_indices(model_ids: np.ndarray, cases_per_model: int, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for model_id, model_name in enumerate(MODEL_NAMES):
        available = np.flatnonzero(model_ids == model_id)
        if len(available) < cases_per_model:
            raise ValueError(f"Only {len(available)} rows are available for {model_name}.")
        selected.append(rng.choice(available, size=cases_per_model, replace=False))
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def sample_case_paths(
    model,
    checkpoint: dict,
    prefix: np.ndarray,
    condition: np.ndarray,
    target: np.ndarray,
    change_points: np.ndarray,
    future_steps: int,
    dt: float,
    n_evidence_samples: int,
    n_paths: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    track_t = torch.from_numpy(normalize_track(prefix, checkpoint).T[None]).float().to(device)
    condition_t = torch.from_numpy(condition[None].astype(np.float32)).to(device)
    pools: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    log_evidence = []
    for model_name in MODEL_NAMES:
        theta = sample_model_parameters(model_name, n_evidence_samples, rng)
        logits = score_params(model, track_t, condition_t, model_name, theta, device)
        pools[model_name] = (theta, softmax(logits))
        log_evidence.append(logmeanexp(logits))
    model_weights = softmax(np.asarray(log_evidence))
    model_ids = rng.choice(len(MODEL_NAMES), size=n_paths, p=model_weights)
    paths = np.empty((n_paths, future_steps, 2), dtype=np.float32)
    parameters = np.zeros((n_paths, MAX_PARAM_DIM), dtype=np.float32)
    for model_id, model_name in enumerate(MODEL_NAMES):
        rows = np.flatnonzero(model_ids == model_id)
        if len(rows) == 0:
            continue
        theta_pool, posterior_weights = pools[model_name]
        theta = theta_pool[rng.choice(len(theta_pool), size=len(rows), p=posterior_weights)]
        future, used_theta = simulate_future_batch(
            model_name=model_name,
            theta=theta,
            start=prefix[-1],
            target=target,
            observed_steps=len(prefix),
            observed_change_points=change_points,
            future_steps=future_steps,
            dt=dt,
            rng=rng,
        )
        paths[rows] = future
        parameters[rows, :used_theta.shape[1]] = used_theta
    return paths, model_ids.astype(np.int64), parameters, model_weights.astype(np.float32)


def plot_examples(
    prefixes: np.ndarray,
    suffixes: np.ndarray,
    paths: np.ndarray,
    true_ids: np.ndarray,
    rows: list[dict[str, object]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for model_id, ax in enumerate(axes.flat):
        candidates = np.flatnonzero(true_ids == model_id)
        interior = (
            (prefixes[candidates, :, 0] > 2.0).all(axis=1)
            & (prefixes[candidates, :, 0] < PITCH_LENGTH - 2.0).all(axis=1)
            & (prefixes[candidates, :, 1] > 2.0).all(axis=1)
            & (prefixes[candidates, :, 1] < PITCH_WIDTH - 2.0).all(axis=1)
            & (suffixes[candidates, :, 0] > 2.0).all(axis=1)
            & (suffixes[candidates, :, 0] < PITCH_LENGTH - 2.0).all(axis=1)
            & (suffixes[candidates, :, 1] > 2.0).all(axis=1)
            & (suffixes[candidates, :, 1] < PITCH_WIDTH - 2.0).all(axis=1)
        )
        if np.any(interior):
            candidates = candidates[interior]
        errors = np.asarray([float(rows[index]["sbi_ade"]) for index in candidates])
        row = int(candidates[np.argmin(np.abs(errors - np.median(errors)))])
        for path in paths[row, :60]:
            ax.plot(path[:, 0], path[:, 1], color="#54a24b", alpha=0.12, linewidth=0.8)
        ax.plot(prefixes[row, :, 0], prefixes[row, :, 1], color="#4c78a8", linewidth=2.5, label="prefix")
        ax.plot(
            np.r_[prefixes[row, -1, 0], suffixes[row, :, 0]],
            np.r_[prefixes[row, -1, 1], suffixes[row, :, 1]],
            color="#e45756",
            linewidth=2.2,
            linestyle="--",
            label="truth",
        )
        visible = np.concatenate([prefixes[row], suffixes[row], paths[row, :60].reshape(-1, 2)])
        low = visible.min(axis=0) - 2.0
        high = visible.max(axis=0) + 2.0
        if high[0] - low[0] < 10.0:
            midpoint = (high[0] + low[0]) / 2.0
            low[0], high[0] = midpoint - 5.0, midpoint + 5.0
        if high[1] - low[1] < 10.0:
            midpoint = (high[1] + low[1]) / 2.0
            low[1], high[1] = midpoint - 5.0, midpoint + 5.0
        ax.set_xlim(max(0.0, low[0]), min(PITCH_LENGTH, high[0]))
        ax.set_ylim(max(0.0, low[1]), min(PITCH_WIDTH, high[1]))
        ax.set_aspect("equal")
        ax.set_title(MODEL_NAMES[model_id])
        ax.grid(alpha=0.15)
    axes[0, 0].legend()
    fig.suptitle("Interior median-error posterior predictive examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(rows: list[dict[str, object]], out_path: Path) -> None:
    methods = ("sbi", "stationary", "last_velocity", "damped_velocity")
    x = np.arange(len(methods))
    ade = [np.mean([float(row[f"{method}_ade"]) for row in rows]) for method in methods]
    fde = [np.mean([float(row[f"{method}_fde"]) for row in rows]) for method in methods]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, ade, width=0.36, label="ADE")
    ax.bar(x + 0.18, fde, width=0.36, label="FDE")
    ax.set_xticks(x, [name.replace("_", "\n") for name in methods])
    ax.set_ylabel("metres")
    ax.set_title("Aggregate controlled forecast error")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(rows: list[dict[str, object]], out_path: Path) -> None:
    levels = np.asarray([0.5, 0.8, 0.9])
    empirical = np.asarray(
        [np.mean([float(row[f"coverage_{int(level * 100)}"]) for row in rows]) for level in levels]
    )
    endpoint = np.asarray(
        [np.mean([float(row[f"endpoint_covered_{int(level * 100)}"]) for row in rows]) for level in levels]
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(levels, levels, color="#555555", linestyle="--", label="ideal")
    ax.plot(levels, empirical, marker="o", linewidth=2, label="all time points")
    ax.plot(levels, endpoint, marker="s", linewidth=2, label="endpoints")
    ax.set_xlim(0.45, 0.95)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("nominal radial coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Posterior predictive calibration")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize_method(rows: list[dict[str, object]], method: str) -> dict[str, float]:
    return {
        "mean_ADE_m": float(np.mean([row[f"{method}_ade"] for row in rows])),
        "median_ADE_m": float(np.median([row[f"{method}_ade"] for row in rows])),
        "mean_FDE_m": float(np.mean([row[f"{method}_fde"] for row in rows])),
        "median_FDE_m": float(np.median([row[f"{method}_fde"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate controlled synthetic posterior predictive forecasts.")
    parser.add_argument("--checkpoint", default="checkpoints/method_validation/ratio_estimator_best.pt")
    parser.add_argument("--forecast-data", default="data/method_validation/forecast_test.npz")
    parser.add_argument("--cases-per-model", type=int, default=25)
    parser.add_argument("--n-evidence-samples", type=int, default=1024)
    parser.add_argument("--n-paths", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--out-dir", default="outputs/method_validation/forecast_evaluation")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    data = np.load(args.forecast_data, allow_pickle=True)
    prefix_steps = int(data["prefix_steps"])
    future_steps = int(data["future_steps"])
    dt = float(data["dt"])
    validate_checkpoint_contract(checkpoint, steps=prefix_steps, dt=dt)
    selected = balanced_indices(data["model_id"], args.cases_per_model, rng)
    prefixes = data["prefix_tracks"][selected].astype(np.float32)
    suffixes = data["suffix_tracks"][selected].astype(np.float32)
    true_ids = data["model_id"][selected].astype(np.int64)
    all_paths = np.empty((len(selected), args.n_paths, future_steps, 2), dtype=np.float32)
    all_path_ids = np.empty((len(selected), args.n_paths), dtype=np.int64)
    all_parameters = np.zeros((len(selected), args.n_paths, MAX_PARAM_DIM), dtype=np.float32)
    all_weights = np.empty((len(selected), len(MODEL_NAMES)), dtype=np.float32)
    rows: list[dict[str, object]] = []

    for case_number, source_index in enumerate(selected):
        paths, path_ids, parameters, model_weights = sample_case_paths(
            model=model,
            checkpoint=checkpoint,
            prefix=prefixes[case_number],
            condition=data["conditions"][source_index],
            target=data["target"][source_index],
            change_points=data["change_points"][source_index],
            future_steps=future_steps,
            dt=dt,
            n_evidence_samples=args.n_evidence_samples,
            n_paths=args.n_paths,
            rng=rng,
            device=device,
        )
        all_paths[case_number] = paths
        all_path_ids[case_number] = path_ids
        all_parameters[case_number] = parameters
        all_weights[case_number] = model_weights
        predictive_mean = paths.mean(axis=0)
        sbi_ade, sbi_fde = ade_fde(predictive_mean, suffixes[case_number])
        baselines = deterministic_baselines(prefixes[case_number], future_steps, dt)
        coverage = radial_coverage(paths, suffixes[case_number])
        row: dict[str, object] = {
            "source_index": int(source_index),
            "true_model": MODEL_NAMES[int(true_ids[case_number])],
            "selected_model": MODEL_NAMES[int(np.argmax(model_weights))],
            "model_correct": int(np.argmax(model_weights) == true_ids[case_number]),
            "sbi_ade": sbi_ade,
            "sbi_fde": sbi_fde,
            "energy_score": energy_score(paths, suffixes[case_number], rng),
        }
        for name, prediction in baselines.items():
            row[f"{name}_ade"], row[f"{name}_fde"] = ade_fde(prediction, suffixes[case_number])
        for level in (50, 80, 90):
            row[f"coverage_{level}"] = coverage[str(level)]["time_fraction"]
            row[f"endpoint_covered_{level}"] = coverage[str(level)]["endpoint_covered"]
            row[f"radius_{level}"] = coverage[str(level)]["mean_radius"]
        for model_id, model_name in enumerate(MODEL_NAMES):
            row[f"weight_{model_name}"] = float(model_weights[model_id])
        rows.append(row)

    methods = ("sbi", "stationary", "last_velocity", "damped_velocity")
    by_model = {}
    for model_name in MODEL_NAMES:
        subset = [row for row in rows if row["true_model"] == model_name]
        by_model[model_name] = {method: summarize_method(subset, method) for method in methods}
    summary = {
        "checkpoint": args.checkpoint,
        "forecast_data": args.forecast_data,
        "n_cases": len(rows),
        "prefix_steps": prefix_steps,
        "future_steps": future_steps,
        "model_recovery_accuracy": float(np.mean([row["model_correct"] for row in rows])),
        "methods": {method: summarize_method(rows, method) for method in methods},
        "mean_energy_score": float(np.mean([row["energy_score"] for row in rows])),
        "predictive_coverage": {
            str(level): {
                "time_points": float(np.mean([row[f"coverage_{level}"] for row in rows])),
                "endpoints": float(np.mean([row[f"endpoint_covered_{level}"] for row in rows])),
                "mean_radius_m": float(np.mean([row[f"radius_{level}"] for row in rows])),
            }
            for level in (50, 80, 90)
        },
        "by_true_model": by_model,
        "posterior_method": "prior importance resampling from learned likelihood-ratio estimates",
        "future_policy": "piecewise continues latest observed segment; no unobserved future turn",
        "comparison_status": "SBI does not beat the last-velocity baseline on aggregate mean ADE/FDE.",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "case_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    samples_path = out_dir / "posterior_predictive_samples.npz"
    np.savez_compressed(
        samples_path,
        source_indices=selected,
        prefix_tracks=prefixes,
        suffix_tracks=suffixes,
        paths=all_paths,
        path_model_ids=all_path_ids,
        path_parameters=all_parameters,
        model_weights=all_weights,
        true_model_ids=true_ids,
        model_names=np.asarray(MODEL_NAMES),
    )
    plot_examples(prefixes, suffixes, all_paths, true_ids, rows, out_dir / "example_forecasts.png")
    plot_comparison(rows, out_dir / "baseline_comparison.png")
    plot_calibration(rows, out_dir / "coverage_calibration.png")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_metadata(
        out_dir / "run_metadata.json",
        stage="controlled_synthetic_forecast_evaluation",
        args=args,
        inputs={"checkpoint": args.checkpoint, "forecast_data": args.forecast_data},
        outputs={"summary": summary_path, "metrics": csv_path, "samples": samples_path},
        contract={"prefix_steps": prefix_steps, "future_steps": future_steps, "dt": dt},
        results=summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
