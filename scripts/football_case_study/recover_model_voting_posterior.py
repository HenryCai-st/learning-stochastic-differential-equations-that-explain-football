"""
Recover the model-voting posterior with random-walk Metropolis-Hastings.

Inputs:
    - data/real_football_windows.npz from extract_football_windows.py
    - checkpoints/model_voting_ratio_best.pt from train_model_voting_ratio.py

Outputs:
    - outputs/model_voting_posterior/summary.json
    - outputs/model_voting_posterior/posterior_chains.npz with per-model chains,
      posterior samples, model vote weights, observed prefix, and future suffix.

Expected use:
    Run this after training the model-voting ratio classifier. If prefix_tracks
    are present, inference uses only the observed prefix and does not leak the
    future suffix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.football.segmentation import detect_change_points, fixed_even_change_points
from src.sbi.artifacts import validate_checkpoint_contract, write_run_metadata
from src.sbi.evidence import logmeanexp, softmax
from src.sbi.mcmc import run_model_mcmc
from src.sbi.scoring import load_checkpoint, normalize_track, score_params
from src.simulators.model_voting import (
    MODEL_NAMES,
    pitch_normalize_condition,
    sample_model_parameters,
)


def load_observed_window(path: Path, window_index: int, dt: float, max_segments: int, min_segment_len: int):
    """Load one real window and build prefix-only inference condition metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Real windows not found: {path}. Run extract_football_windows.py first.")
    real = np.load(path, allow_pickle=True)
    real_dt = float(real["dt"]) if "dt" in real.files else dt
    full_track = real["tracks"][window_index].astype(np.float32)
    if "prefix_tracks" in real.files and "suffix_tracks" in real.files:
        track = real["prefix_tracks"][window_index].astype(np.float32)
        suffix = real["suffix_tracks"][window_index].astype(np.float32)
        y0 = track[0].astype(np.float32)
        target = track[-1].astype(np.float32)
        prediction_y0 = track[-1].astype(np.float32)
        # Keep the OU condition consistent between inference and prediction.
        # The last observed position is the no-leak equilibrium for an OU
        # stop/settle hypothesis; moving futures are represented by the
        # velocity-based candidate models instead.
        prediction_target = target.copy()
        protocol = "prefix_suffix"
    else:
        track = full_track
        suffix = np.empty((0, 2), dtype=np.float32)
        y0 = real["y0"][window_index].astype(np.float32)
        target = real["target"][window_index].astype(np.float32)
        prediction_y0 = y0
        prediction_target = target
        protocol = "full_window_reconstruction"
    change_points = detect_change_points(
        track,
        dt=real_dt,
        max_segments=max_segments,
        min_segment_len=min_segment_len,
    )
    if len(change_points) < max_segments - 1:
        change_points = fixed_even_change_points(
            len(track),
            max_segments=max_segments,
            min_segment_len=min_segment_len,
        )
    padded_cps = np.zeros(max_segments - 1, dtype=np.int64)
    padded_cps[:min(len(change_points), max_segments - 1)] = change_points[:max_segments - 1]
    condition = pitch_normalize_condition(y0, target, padded_cps, len(track))
    return track, full_track, suffix, y0, target, prediction_y0, prediction_target, padded_cps, condition, real_dt, protocol


def main() -> None:
    """Run per-model MCMC chains and save posterior samples/vote weights."""
    parser = argparse.ArgumentParser(description="Recover p(model, theta | observed track) with model-voting MCMC.")
    parser.add_argument("--real-windows", default="data/real_football_windows.npz")
    parser.add_argument("--checkpoint", default="checkpoints/model_voting_ratio_best.pt")
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--n-init-candidates", type=int, default=2048)
    parser.add_argument(
        "--n-evidence-samples",
        type=int,
        default=4096,
        help="Prior samples used to estimate each model's marginal evidence ratio.",
    )
    parser.add_argument("--mcmc-steps", type=int, default=3000)
    parser.add_argument("--burn-in", type=int, default=800)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--min-segment-len", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="outputs/model_voting_posterior")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(Path(args.checkpoint), device)

    (
        track,
        full_track,
        suffix,
        y0,
        target,
        prediction_y0,
        prediction_target,
        change_points,
        condition,
        dt,
        protocol,
    ) = load_observed_window(
        Path(args.real_windows),
        args.window_index,
        args.dt,
        args.max_segments,
        args.min_segment_len,
    )
    validate_checkpoint_contract(ckpt, steps=len(track), dt=dt)
    track_norm = normalize_track(track, ckpt)
    track_t = torch.from_numpy(track_norm.T[None]).float().to(device)
    condition_t = torch.from_numpy(condition[None]).float().to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "window_index": args.window_index,
        "dt": dt,
        "protocol": protocol,
        "y0": y0.tolist(),
        "target": target.tolist(),
        "prediction_y0": prediction_y0.tolist(),
        "prediction_target": prediction_target.tolist(),
        "change_points": change_points.tolist(),
        "model_weight_method": "equal model priors with prior Monte Carlo integration of exp(classifier log-ratio)",
        "model_weight_status": "approximate; synthetic recovery and multi-window calibration are still required",
        "models": {},
    }
    log_evidence_ratios = []
    save_payload: dict[str, np.ndarray] = {
        "observed": track,
        "full_observed": full_track,
        "future_suffix": suffix,
        "y0": y0,
        "target": target,
        "prediction_y0": prediction_y0,
        "prediction_target": prediction_target,
        "change_points": change_points,
        "condition": condition,
        "protocol": np.asarray(protocol),
        "model_weight_method": np.asarray("prior_mc_ratio_evidence_equal_model_priors"),
        "model_weight_status": np.asarray("approximate_uncalibrated"),
        "dt": np.asarray(dt, dtype=np.float32),
    }

    for model_name in MODEL_NAMES:
        # If the classifier estimates log p(x | model, theta) / p(x), then
        # averaging exp(log-ratio) over theta ~ p(theta | model) estimates
        # p(x | model) / p(x). This prior integral is comparable across models;
        # a posterior-chain mean log density is not.
        n_prior = max(args.n_init_candidates, args.n_evidence_samples)
        prior_candidates = sample_model_parameters(model_name, n_prior, rng)
        prior_logits = score_params(model, track_t, condition_t, model_name, prior_candidates, device)
        evidence_logits = prior_logits[:args.n_evidence_samples]
        log_evidence_ratio = logmeanexp(evidence_logits)
        log_evidence_ratios.append(log_evidence_ratio)

        init_logits = prior_logits[:args.n_init_candidates]
        best_idx = int(np.argmax(init_logits))
        result = run_model_mcmc(
            model=model,
            track_t=track_t,
            condition_t=condition_t,
            model_name=model_name,
            initial_theta=prior_candidates[best_idx],
            n_steps=args.mcmc_steps,
            burn_in=args.burn_in,
            rng=rng,
            device=device,
        )
        summary["models"][model_name] = {
            "init_best_logit": float(init_logits[best_idx]),
            "log_evidence_ratio_prior_mc": float(log_evidence_ratio),
            "n_evidence_samples": int(args.n_evidence_samples),
            "acceptance_rate": float(result["acceptance_rate"]),
            "mean_logp": float(result["mean_logp"]),
            "max_logp": float(result["max_logp"]),
            "posterior_mean": np.asarray(result["posterior_mean"]).tolist(),
            "map": np.asarray(result["map"]).tolist(),
        }
        save_payload[f"{model_name}_chain"] = np.asarray(result["chain"], dtype=np.float32)
        save_payload[f"{model_name}_logp"] = np.asarray(result["logp"], dtype=np.float32)
        save_payload[f"{model_name}_samples"] = np.asarray(result["samples"], dtype=np.float32)

    # Equal model-family priors are used. With non-uniform model priors, add
    # log p(model) to each log evidence ratio before this normalization.
    vote_weights = softmax(np.asarray(log_evidence_ratios, dtype=np.float64))
    summary["model_vote_weights"] = {
        model_name: float(weight)
        for model_name, weight in zip(MODEL_NAMES, vote_weights)
    }
    save_payload["model_vote_weights"] = vote_weights.astype(np.float32)
    save_payload["log_evidence_ratios"] = np.asarray(log_evidence_ratios, dtype=np.float32)
    save_payload["model_names"] = np.asarray(MODEL_NAMES)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "posterior_chains.npz", **save_payload)
    write_run_metadata(
        out_dir / "run_metadata.json",
        stage="football_posterior_recovery",
        args=args,
        inputs={"real_windows": args.real_windows, "checkpoint": args.checkpoint},
        outputs={"summary": out_dir / "summary.json", "posterior": out_dir / "posterior_chains.npz"},
        contract={"steps": len(track), "dt": dt, "protocol": protocol},
        results={"model_vote_weights": summary["model_vote_weights"]},
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved model-voting posterior outputs to {out_dir}")


if __name__ == "__main__":
    main()
