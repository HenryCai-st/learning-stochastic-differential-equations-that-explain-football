"""
Generate mixed-model synthetic football trajectories for model-voting SBI.

Inputs:
    - optional real football windows from extract_football_windows.py
    - priors for Brownian, constant velocity, OU-to-target, and piecewise
      velocity models

Outputs:
    - data/model_voting_dataset/dataset.npz with tracks, model IDs,
      normalized/padded parameters, masks, conditions, y0/target, and segments.

Expected use:
    Run this before train_model_voting_ratio.py. If the real windows contain
    prefix_tracks, synthetic training tracks automatically match that prefix
    length so training and inference use the same observed duration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.football.segmentation import detect_change_points, fixed_even_change_points
from src.simulators.model_voting import (
    MAX_PARAM_DIM,
    MODEL_NAMES,
    MODEL_TO_ID,
    normalize_padded_parameters,
    pad_parameters,
    pitch_normalize_condition,
    sample_model_parameters,
    simulate_model_batch,
)
from src.sbi.artifacts import data_artifact_metadata, write_run_metadata


def load_real_condition_pool(path: str | Path, dt: float, max_segments: int, min_segment_len: int):
    """Load real-window start/end/segment conditions for synthetic bootstrapping."""
    real = np.load(path, allow_pickle=True)
    if "prefix_tracks" in real.files:
        tracks = real["prefix_tracks"].astype(np.float32)
        y0 = real["prefix_y0"].astype(np.float32) if "prefix_y0" in real.files else tracks[:, 0]
        target = real["prefix_target"].astype(np.float32) if "prefix_target" in real.files else tracks[:, -1]
        source = "real_prefix_bootstrap"
    else:
        tracks = real["tracks"].astype(np.float32)
        y0 = real["y0"].astype(np.float32)
        target = real["target"].astype(np.float32)
        source = "real_window_bootstrap"
    change_points = []
    for track in tracks:
        cps = detect_change_points(
            track,
            dt=dt,
            max_segments=max_segments,
            min_segment_len=min_segment_len,
        )
        if len(cps) < max_segments - 1:
            cps = fixed_even_change_points(len(track), max_segments=max_segments, min_segment_len=min_segment_len)
        padded = np.zeros(max_segments - 1, dtype=np.int64)
        padded[:min(len(cps), max_segments - 1)] = cps[:max_segments - 1]
        change_points.append(padded)
    return y0, target, np.asarray(change_points, dtype=np.int64), source, len(tracks[0])


def synthetic_condition_pool(n: int, steps: int, rng: np.random.Generator, max_segments: int, min_segment_len: int):
    """Create fallback random start/end/segment conditions when no real data exists."""
    y0 = np.column_stack([
        rng.uniform(0.0, 105.0, size=n),
        rng.uniform(0.0, 68.0, size=n),
    ]).astype(np.float32)
    displacement = rng.normal(0.0, [18.0, 12.0], size=(n, 2)).astype(np.float32)
    target = y0 + displacement
    target[:, 0] = np.clip(target[:, 0], 0.0, 105.0)
    target[:, 1] = np.clip(target[:, 1], 0.0, 68.0)
    cps = np.tile(
        fixed_even_change_points(steps, max_segments=max_segments, min_segment_len=min_segment_len),
        (n, 1),
    )
    return y0, target, cps.astype(np.int64), "synthetic_condition_pool"


def sample_conditions(n: int, args, rng: np.random.Generator):
    """Sample condition rows from real windows when possible, otherwise synthetic ones."""
    real_path = Path(args.real_windows)
    if real_path.exists():
        y0_pool, target_pool, cps_pool, source, source_steps = load_real_condition_pool(
            real_path,
            dt=args.dt,
            max_segments=args.max_segments,
            min_segment_len=args.min_segment_len,
        )
        if args.auto_prefix_steps and source == "real_prefix_bootstrap":
            args.steps = source_steps
            args.T = args.steps * args.dt
        idx = rng.choice(len(y0_pool), size=n, replace=True)
        return y0_pool[idx], target_pool[idx], cps_pool[idx], source
    return synthetic_condition_pool(n, args.steps, rng, args.max_segments, args.min_segment_len)


def main() -> None:
    """Generate balanced synthetic tracks for every candidate model family."""
    parser = argparse.ArgumentParser(description="Generate mixed-model synthetic football tracks for model voting.")
    parser.add_argument("--real-windows", default="data/real_football_windows.npz")
    parser.add_argument("--out-dir", default="data/model_voting_dataset")
    parser.add_argument("--n-per-model", type=int, default=1000)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--min-segment-len", type=int, default=12)
    parser.add_argument(
        "--no-auto-prefix-steps",
        action="store_true",
        help="Disable automatic use of prefix_steps from real windows.",
    )
    args = parser.parse_args()
    args.steps = int(round(args.T / args.dt))
    args.auto_prefix_steps = not args.no_auto_prefix_steps

    rng = np.random.default_rng(args.seed)
    all_tracks = []
    all_params = []
    all_param_mask = []
    all_params_norm = []
    all_model_id = []
    all_conditions = []
    all_y0 = []
    all_target = []
    all_change_points = []
    condition_sources = {}

    for model_name in MODEL_NAMES:
        y0, target, change_points, condition_source = sample_conditions(args.n_per_model, args, rng)
        condition_sources[model_name] = condition_source
        raw_params = sample_model_parameters(model_name, args.n_per_model, rng)
        padded, mask = pad_parameters(model_name, raw_params)
        params_norm = normalize_padded_parameters(model_name, padded)

        tracks = simulate_model_batch(
            model_name=model_name,
            params=raw_params,
            y0=y0,
            target=target,
            change_points=change_points,
            steps=args.steps,
            dt=args.dt,
            rng=rng,
        )

        conditions = np.stack([
            pitch_normalize_condition(y0[i], target[i], change_points[i], args.steps)
            for i in range(args.n_per_model)
        ])

        all_tracks.append(tracks)
        all_params.append(padded)
        all_param_mask.append(mask)
        all_params_norm.append(params_norm)
        all_model_id.append(np.full(args.n_per_model, MODEL_TO_ID[model_name], dtype=np.int64))
        all_conditions.append(conditions)
        all_y0.append(y0)
        all_target.append(target)
        all_change_points.append(change_points)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "n_tracks": int(args.n_per_model * len(MODEL_NAMES)),
        "steps": int(args.steps),
        "track_channels": 2,
        "dt": float(args.dt),
        "T": float(args.T),
        "seed": int(args.seed),
        "model_names": list(MODEL_NAMES),
        "max_param_dim": int(MAX_PARAM_DIM),
        "condition_sources": condition_sources,
    }
    artifact_metadata = data_artifact_metadata(
        artifact_type="model_voting_synthetic_dataset",
        args=args,
        contract=contract,
        inputs={"real_windows": args.real_windows},
    )
    np.savez_compressed(
        out_dir / "dataset.npz",
        tracks=np.concatenate(all_tracks).astype(np.float32),
        parameters=np.concatenate(all_params).astype(np.float32),
        parameter_mask=np.concatenate(all_param_mask).astype(np.float32),
        parameters_norm=np.concatenate(all_params_norm).astype(np.float32),
        model_id=np.concatenate(all_model_id).astype(np.int64),
        conditions=np.concatenate(all_conditions).astype(np.float32),
        y0=np.concatenate(all_y0).astype(np.float32),
        target=np.concatenate(all_target).astype(np.float32),
        change_points=np.concatenate(all_change_points).astype(np.int64),
        model_names=np.asarray(MODEL_NAMES),
        max_param_dim=MAX_PARAM_DIM,
        T=args.T,
        dt=args.dt,
        steps=args.steps,
        seed=args.seed,
        condition_sources=json.dumps(condition_sources),
        artifact_metadata_json=np.asarray(json.dumps(artifact_metadata)),
    )
    write_run_metadata(
        out_dir / "run_metadata.json",
        stage="model_voting_data_generation",
        args=args,
        inputs={"real_windows": args.real_windows},
        outputs={"dataset": out_dir / "dataset.npz"},
        contract=contract,
    )
    print(json.dumps({
        "out": str(out_dir / "dataset.npz"),
        "models": list(MODEL_NAMES),
        "tracks": int(args.n_per_model * len(MODEL_NAMES)),
        "steps": args.steps,
        "condition_sources": condition_sources,
    }, indent=2))


if __name__ == "__main__":
    main()
