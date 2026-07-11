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

from src.data.segmentation import detect_change_points, fixed_even_change_points
from src.models.model_voting_ratio import ModelVotingRatioClassifier
from src.sde.model_voting import (
    MAX_PARAM_DIM,
    MODEL_NAMES,
    MODEL_SPECS,
    MODEL_TO_ID,
    normalize_padded_parameters,
    pad_parameters,
    pitch_normalize_condition,
    sample_model_parameters,
)


def load_checkpoint(path: Path, device: torch.device) -> tuple[ModelVotingRatioClassifier, dict]:
    """Load the trained model-voting ratio classifier and checkpoint metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run train_model_voting_ratio.py first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ModelVotingRatioClassifier()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def checkpoint_array(ckpt: dict, key: str) -> np.ndarray:
    """Read checkpoint arrays saved as either tensors or NumPy arrays."""
    value = ckpt[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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


def normalize_track(track: np.ndarray, ckpt: dict) -> np.ndarray:
    """Normalize an observed track with training-set checkpoint statistics."""
    track_mean = checkpoint_array(ckpt, "track_mean").astype(np.float32)
    track_std = checkpoint_array(ckpt, "track_std").astype(np.float32)
    track_std = np.where(track_std < 1e-8, 1.0, track_std)
    return ((track - track_mean) / track_std).astype(np.float32)


@torch.no_grad()
def score_params(
    model: ModelVotingRatioClassifier,
    track_t: torch.Tensor,
    condition_t: torch.Tensor,
    model_name: str,
    params: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Evaluate classifier logits for many candidate parameters of one model."""
    padded, mask = pad_parameters(model_name, params.astype(np.float32))
    params_norm = normalize_padded_parameters(model_name, padded)
    model_id = MODEL_TO_ID[model_name]
    outputs: list[np.ndarray] = []
    for start in range(0, len(params), batch_size):
        end = start + batch_size
        n = len(params_norm[start:end])
        logits = model(
            track_t.repeat(n, 1, 1),
            torch.from_numpy(params_norm[start:end]).to(device),
            torch.from_numpy(mask[start:end]).to(device),
            torch.full((n,), model_id, dtype=torch.long, device=device),
            condition_t.repeat(n, 1),
        )
        outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs)


def log_prior(model_name: str, theta: np.ndarray) -> float:
    """Evaluate the model-specific log prior up to an irrelevant constant."""
    spec = MODEL_SPECS[model_name]
    theta = np.asarray(theta, dtype=np.float64)
    if len(theta) != spec.param_dim:
        return -np.inf
    if np.any(theta < spec.low) or np.any(theta > spec.high):
        return -np.inf
    logp = 0.0
    for value, is_log in zip(theta, spec.log_scale):
        if is_log:
            if value <= 0.0:
                return -np.inf
            logp -= float(np.log(value))
    return logp


def proposal_scale_for_model(model_name: str) -> np.ndarray:
    """Choose a simple Gaussian proposal scale for one model's parameter space."""
    spec = MODEL_SPECS[model_name]
    width = spec.high - spec.low
    scale = 0.04 * width
    scale = np.where(spec.log_scale, np.maximum(scale, 0.08), scale)
    return scale.astype(np.float64)


def run_model_mcmc(
    model: ModelVotingRatioClassifier,
    track_t: torch.Tensor,
    condition_t: torch.Tensor,
    model_name: str,
    initial_theta: np.ndarray,
    n_steps: int,
    burn_in: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    """Run random-walk Metropolis-Hastings for one candidate model family."""
    proposal_scale = proposal_scale_for_model(model_name)

    def log_target(theta: np.ndarray) -> float:
        """Unnormalized log posterior target for one proposed theta."""
        lp = log_prior(model_name, theta)
        if not np.isfinite(lp):
            return -np.inf
        logit = score_params(
            model=model,
            track_t=track_t,
            condition_t=condition_t,
            model_name=model_name,
            params=theta[None].astype(np.float32),
            device=device,
            batch_size=1,
        )[0]
        return float(lp + logit)

    current = initial_theta.astype(np.float64).copy()
    current_logp = log_target(current)
    chain = np.zeros((n_steps, len(current)), dtype=np.float32)
    logp = np.zeros(n_steps, dtype=np.float32)
    accepted = 0

    for step in range(n_steps):
        proposal = current + rng.normal(0.0, proposal_scale, size=len(current))
        proposal_logp = log_target(proposal)
        if np.log(rng.uniform()) < proposal_logp - current_logp:
            current = proposal
            current_logp = proposal_logp
            accepted += 1
        chain[step] = current
        logp[step] = current_logp

    samples = chain[burn_in:]
    sample_logp = logp[burn_in:]
    return {
        "chain": chain,
        "logp": logp,
        "samples": samples,
        "map": samples[int(np.argmax(sample_logp))],
        "posterior_mean": samples.mean(axis=0),
        "acceptance_rate": accepted / max(1, n_steps),
        "mean_logp": float(sample_logp.mean()),
        "max_logp": float(sample_logp.max()),
    }


def softmax(values: np.ndarray) -> np.ndarray:
    """Convert per-model evidence scores into normalized vote weights."""
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def logmeanexp(values: np.ndarray) -> float:
    """Stable log of the arithmetic mean of exp(values)."""
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return maximum + float(np.log(np.mean(np.exp(values - maximum))))


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
    print(json.dumps(summary, indent=2))
    print(f"Saved model-voting posterior outputs to {out_dir}")


if __name__ == "__main__":
    main()
