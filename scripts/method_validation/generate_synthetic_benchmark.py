"""Generate independent train, validation, and test artifacts for Part I."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sbi.artifacts import data_artifact_metadata, write_run_metadata
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
from src.synthetic.conditions import generate_condition_pool


def generate_split(
    split: str,
    n_per_model: int,
    steps: int,
    dt: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate one balanced split with conditions shared across model families."""
    rng = np.random.default_rng(seed)
    y0_pool, target_pool, change_point_pool = generate_condition_pool(n_per_model, steps, rng)

    tracks_all = []
    parameters_all = []
    masks_all = []
    parameters_norm_all = []
    model_ids_all = []
    conditions_all = []
    y0_all = []
    targets_all = []
    change_points_all = []

    for model_name in MODEL_NAMES:
        y0 = y0_pool.copy()
        target = target_pool.copy()
        change_points = change_point_pool.copy()
        raw_parameters = sample_model_parameters(model_name, n_per_model, rng)
        padded, mask = pad_parameters(model_name, raw_parameters)
        tracks = simulate_model_batch(
            model_name=model_name,
            params=raw_parameters,
            y0=y0,
            target=target,
            change_points=change_points,
            steps=steps,
            dt=dt,
            rng=rng,
        )
        conditions = np.stack(
            [
                pitch_normalize_condition(y0[i], target[i], change_points[i], steps)
                for i in range(n_per_model)
            ]
        )

        tracks_all.append(tracks)
        parameters_all.append(padded)
        masks_all.append(mask)
        parameters_norm_all.append(normalize_padded_parameters(model_name, padded))
        model_ids_all.append(np.full(n_per_model, MODEL_TO_ID[model_name], dtype=np.int64))
        conditions_all.append(conditions)
        y0_all.append(y0)
        targets_all.append(target)
        change_points_all.append(change_points)

    payload = {
        "tracks": np.concatenate(tracks_all).astype(np.float32),
        "parameters": np.concatenate(parameters_all).astype(np.float32),
        "parameter_mask": np.concatenate(masks_all).astype(np.float32),
        "parameters_norm": np.concatenate(parameters_norm_all).astype(np.float32),
        "model_id": np.concatenate(model_ids_all).astype(np.int64),
        "conditions": np.concatenate(conditions_all).astype(np.float32),
        "y0": np.concatenate(y0_all).astype(np.float32),
        "target": np.concatenate(targets_all).astype(np.float32),
        "change_points": np.concatenate(change_points_all).astype(np.int64),
    }
    permutation = rng.permutation(len(payload["tracks"]))
    return {key: value[permutation] for key, value in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a controlled synthetic model-voting benchmark.")
    parser.add_argument("--out-dir", default="data/method_validation")
    parser.add_argument("--n-train-per-model", type=int, default=1000)
    parser.add_argument("--n-validation-per-model", type=int, default=100)
    parser.add_argument("--n-test-per-model", type=int, default=100)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    steps = int(round(args.T / args.dt))
    split_sizes = {
        "train": args.n_train_per_model,
        "validation": args.n_validation_per_model,
        "test": args.n_test_per_model,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    split_summary: dict[str, object] = {}

    for offset, (split, n_per_model) in enumerate(split_sizes.items()):
        split_seed = args.seed + offset
        payload = generate_split(split, n_per_model, steps, args.dt, split_seed)
        contract = {
            "split": split,
            "n_tracks": int(len(payload["tracks"])),
            "n_per_model": int(n_per_model),
            "steps": steps,
            "track_channels": 2,
            "dt": float(args.dt),
            "T": float(args.T),
            "seed": int(split_seed),
            "model_names": list(MODEL_NAMES),
            "max_param_dim": MAX_PARAM_DIM,
            "condition_source": "controlled_synthetic",
            "condition_policy": "central_uniform_start_random_target_fixed_change_points",
        }
        metadata = data_artifact_metadata(
            artifact_type="controlled_synthetic_model_voting_split",
            args=args,
            contract=contract,
        )
        output_path = out_dir / f"{split}.npz"
        np.savez_compressed(
            output_path,
            **payload,
            model_names=np.asarray(MODEL_NAMES),
            max_param_dim=MAX_PARAM_DIM,
            T=args.T,
            dt=args.dt,
            steps=steps,
            seed=split_seed,
            split=np.asarray(split),
            condition_sources=np.asarray("controlled_synthetic"),
            artifact_metadata_json=np.asarray(json.dumps(metadata)),
        )
        outputs[split] = output_path
        split_summary[split] = contract

    write_run_metadata(
        out_dir / "run_metadata.json",
        stage="controlled_synthetic_benchmark_generation",
        args=args,
        outputs=outputs,
        contract={"splits": split_summary},
    )
    print(json.dumps({"out_dir": str(out_dir), "splits": split_summary}, indent=2))


if __name__ == "__main__":
    main()
