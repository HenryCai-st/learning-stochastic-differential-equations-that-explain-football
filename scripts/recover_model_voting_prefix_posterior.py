from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.segmentation import detect_change_points, fixed_even_change_points
from src.models.model_voting_ratio import ModelVotingRatioClassifier
from src.sde.football_ou import PITCH_LENGTH, PITCH_WIDTH
from src.sde.model_voting import (
    MODEL_NAMES,
    MODEL_SPECS,
    MODEL_TO_ID,
    normalize_padded_parameters,
    pad_parameters,
    pitch_normalize_condition,
    sample_model_parameters,
)


def load_checkpoint(path: Path, device: torch.device) -> tuple[ModelVotingRatioClassifier, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}.")
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


def prefix_projected_target(
    prefix: np.ndarray,
    dt: float,
    suffix_steps: int,
    strategy: str,
    tail_steps: int = 8,
) -> np.ndarray:
    """Project a target from prefix-only information, clipped to the pitch."""
    if strategy == "prefix_end":
        target = prefix[-1].copy()
    elif strategy == "average":
        horizon = dt * max(suffix_steps, 1)
        velocity = (prefix[-1] - prefix[0]) / (dt * max(len(prefix) - 1, 1))
        target = prefix[-1] + velocity * horizon
    elif strategy == "recent":
        n = min(max(tail_steps, 2), len(prefix) - 1)
        velocity = (prefix[-1] - prefix[-1 - n]) / (dt * n)
        horizon = dt * max(suffix_steps, 1)
        target = prefix[-1] + velocity * horizon
    else:
        raise ValueError(f"Unknown target strategy: {strategy}")
    target[0] = np.clip(target[0], 0.0, PITCH_LENGTH)
    target[1] = np.clip(target[1], 0.0, PITCH_WIDTH)
    return target.astype(np.float32)


def normalize_track(track: np.ndarray, ckpt: dict) -> np.ndarray:
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
    padded, mask = pad_parameters(model_name, params.astype(np.float32))
    params_norm = normalize_padded_parameters(model_name, padded)
    model_id = MODEL_TO_ID[model_name]
    outputs: list[np.ndarray] = []
    for start in range(0, len(params), batch_size):
        end = min(start + batch_size, len(params))
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


def log_prior(model_name: str, theta: np.ndarray) -> float:
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
    proposal_scale = proposal_scale_for_model(model_name)

    def log_target(theta: np.ndarray) -> float:
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
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover p(model, theta | prefix) without suffix leakage.")
    parser.add_argument("--windows", default="data/real_football_prefix_suffix_windows.npz")
    parser.add_argument("--checkpoint", default="checkpoints/model_voting_ratio_best.pt")
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--n-init-candidates", type=int, default=2048)
    parser.add_argument("--mcmc-steps", type=int, default=3000)
    parser.add_argument("--burn-in", type=int, default=800)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--min-segment-len", type=int, default=12)
    parser.add_argument("--target-strategy", choices=["average", "recent", "prefix_end"], default="average")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="outputs/prefix_suffix_posterior")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(Path(args.checkpoint), device)

    data = np.load(args.windows, allow_pickle=True)
    prefix = data["prefix_tracks"][args.window_index].astype(np.float32)
    suffix = data["suffix_tracks"][args.window_index].astype(np.float32)
    full = data["full_tracks"][args.window_index].astype(np.float32)
    dt = float(data["dt"])
    suffix_steps = int(data["suffix_steps"])
    y0 = prefix[0].astype(np.float32)
    prefix_end = prefix[-1].astype(np.float32)
    target_for_inference = prefix_projected_target(prefix, dt, suffix_steps, args.target_strategy)
    target_for_evaluation = data["target_for_evaluation"][args.window_index].astype(np.float32)

    change_points = detect_change_points(
        prefix,
        dt=dt,
        max_segments=args.max_segments,
        min_segment_len=args.min_segment_len,
    )
    if len(change_points) < args.max_segments - 1:
        change_points = fixed_even_change_points(
            len(prefix),
            max_segments=args.max_segments,
            min_segment_len=args.min_segment_len,
        )
    padded_cps = np.zeros(args.max_segments - 1, dtype=np.int64)
    padded_cps[:min(len(change_points), args.max_segments - 1)] = change_points[:args.max_segments - 1]

    condition = pitch_normalize_condition(y0, target_for_inference, padded_cps, len(prefix))
    track_norm = normalize_track(prefix, ckpt)
    track_t = torch.from_numpy(track_norm.T[None]).float().to(device)
    condition_t = torch.from_numpy(condition[None]).float().to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "window_index": args.window_index,
        "dt": dt,
        "prefix_steps": int(len(prefix)),
        "suffix_steps": int(len(suffix)),
        "y0": y0.tolist(),
        "prefix_end": prefix_end.tolist(),
        "target_for_inference": target_for_inference.tolist(),
        "target_strategy": args.target_strategy,
        "target_for_evaluation": target_for_evaluation.tolist(),
        "change_points": padded_cps.tolist(),
        "models": {},
    }
    evidence_scores = []
    save_payload: dict[str, np.ndarray] = {
        "prefix": prefix,
        "suffix": suffix,
        "full_track": full,
        "y0": y0,
        "prefix_end": prefix_end,
        "target_for_inference": target_for_inference,
        "target_strategy": np.asarray(args.target_strategy),
        "target_for_evaluation": target_for_evaluation,
        "change_points": padded_cps,
        "condition": condition,
        "dt": np.asarray(dt, dtype=np.float32),
        "prefix_steps": np.asarray(len(prefix), dtype=np.int32),
        "suffix_steps": np.asarray(len(suffix), dtype=np.int32),
    }

    for model_name in MODEL_NAMES:
        candidates = sample_model_parameters(model_name, args.n_init_candidates, rng)
        logits = score_params(model, track_t, condition_t, model_name, candidates, device)
        best_idx = int(np.argmax(logits))
        result = run_model_mcmc(
            model=model,
            track_t=track_t,
            condition_t=condition_t,
            model_name=model_name,
            initial_theta=candidates[best_idx],
            n_steps=args.mcmc_steps,
            burn_in=args.burn_in,
            rng=rng,
            device=device,
        )
        evidence_scores.append(float(result["mean_logp"]))
        summary["models"][model_name] = {
            "init_best_logit": float(logits[best_idx]),
            "acceptance_rate": float(result["acceptance_rate"]),
            "mean_logp": float(result["mean_logp"]),
            "max_logp": float(result["max_logp"]),
            "posterior_mean": np.asarray(result["posterior_mean"]).tolist(),
            "map": np.asarray(result["map"]).tolist(),
        }
        save_payload[f"{model_name}_chain"] = np.asarray(result["chain"], dtype=np.float32)
        save_payload[f"{model_name}_logp"] = np.asarray(result["logp"], dtype=np.float32)
        save_payload[f"{model_name}_samples"] = np.asarray(result["samples"], dtype=np.float32)

    vote_weights = softmax(np.asarray(evidence_scores, dtype=np.float64))
    summary["model_vote_weights"] = {
        model_name: float(weight)
        for model_name, weight in zip(MODEL_NAMES, vote_weights)
    }
    save_payload["model_vote_weights"] = vote_weights.astype(np.float32)
    save_payload["model_names"] = np.asarray(MODEL_NAMES)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "posterior_chains.npz", **save_payload)
    print(json.dumps(summary, indent=2))
    print(f"Saved prefix posterior outputs to {out_dir}")


if __name__ == "__main__":
    main()
