from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.segmentation import detect_change_points, fixed_even_change_points
from src.sde.model_voting import (
    MAX_PARAM_DIM,
    MODEL_NAMES,
    MODEL_TO_ID,
    normalize_padded_parameters,
    pad_parameters,
    pitch_normalize_condition,
    sample_model_parameters,
    simulate_model_batch,
)


def load_real_condition_pool(path: str | Path, dt: float, max_segments: int, min_segment_len: int):
    real = np.load(path, allow_pickle=True)
    tracks = real["tracks"].astype(np.float32)
    y0 = real["y0"].astype(np.float32)
    target = real["target"].astype(np.float32)
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
    return y0, target, np.asarray(change_points, dtype=np.int64), "real_window_bootstrap"


def synthetic_condition_pool(n: int, steps: int, rng: np.random.Generator, max_segments: int, min_segment_len: int):
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
    real_path = Path(args.real_windows)
    if real_path.exists():
        y0_pool, target_pool, cps_pool, source = load_real_condition_pool(
            real_path,
            dt=args.dt,
            max_segments=args.max_segments,
            min_segment_len=args.min_segment_len,
        )
        idx = rng.choice(len(y0_pool), size=n, replace=True)
        return y0_pool[idx], target_pool[idx], cps_pool[idx], source
    return synthetic_condition_pool(n, args.steps, rng, args.max_segments, args.min_segment_len)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mixed-model synthetic football tracks for model voting.")
    parser.add_argument("--real-windows", default="data/real_football_windows.npz")
    parser.add_argument("--out-dir", default="data/model_voting_dataset")
    parser.add_argument("--n-per-model", type=int, default=1000)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--min-segment-len", type=int, default=12)
    args = parser.parse_args()
    args.steps = int(round(args.T / args.dt))

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
