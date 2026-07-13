"""Extend independent synthetic test prefixes with held-out future suffixes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sbi.artifacts import data_artifact_metadata, write_run_metadata
from src.sbi.forecasting import simulate_future_batch
from src.simulators.model_voting import MODEL_NAMES, MODEL_SPECS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate held-out suffixes for controlled forecast validation.")
    parser.add_argument("--test-data", default="data/method_validation/test.npz")
    parser.add_argument("--future-T", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--out", default="data/method_validation/forecast_test.npz")
    args = parser.parse_args()

    source = np.load(args.test_data, allow_pickle=True)
    dt = float(source["dt"])
    future_steps = int(round(args.future_T / dt))
    rng = np.random.default_rng(args.seed)
    prefixes = source["tracks"].astype(np.float32)
    suffixes = np.empty((len(prefixes), future_steps, 2), dtype=np.float32)
    simulation_parameters = source["parameters"].astype(np.float32).copy()

    for model_id, model_name in enumerate(MODEL_NAMES):
        rows = np.flatnonzero(source["model_id"] == model_id)
        dim = MODEL_SPECS[model_name].param_dim
        for row in rows:
            future, used_theta = simulate_future_batch(
                model_name=model_name,
                theta=source["parameters"][row:row + 1, :dim],
                start=prefixes[row, -1],
                target=source["target"][row],
                observed_steps=prefixes.shape[1],
                observed_change_points=source["change_points"][row],
                future_steps=future_steps,
                dt=dt,
                rng=rng,
            )
            suffixes[row] = future[0]
            simulation_parameters[row, :dim] = used_theta[0]

    contract = {
        "n_cases": len(prefixes),
        "prefix_steps": prefixes.shape[1],
        "future_steps": future_steps,
        "dt": dt,
        "prefix_T": float(source["T"]),
        "future_T": args.future_T,
        "source_test_seed": int(source["seed"]),
        "forecast_seed": args.seed,
        "future_policy": "continue_same_dynamics_piecewise_latest_segment_no_unobserved_turn",
    }
    metadata = data_artifact_metadata(
        artifact_type="controlled_synthetic_prefix_suffix_forecast_split",
        args=args,
        contract=contract,
        inputs={"test_data": args.test_data},
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        prefix_tracks=prefixes,
        suffix_tracks=suffixes,
        model_id=source["model_id"],
        parameters=source["parameters"],
        simulation_parameters=simulation_parameters,
        parameter_mask=source["parameter_mask"],
        conditions=source["conditions"],
        y0=source["y0"],
        target=source["target"],
        change_points=source["change_points"],
        model_names=source["model_names"],
        prefix_steps=prefixes.shape[1],
        future_steps=future_steps,
        dt=dt,
        prefix_T=float(source["T"]),
        future_T=args.future_T,
        source_test_seed=int(source["seed"]),
        seed=args.seed,
        artifact_metadata_json=np.asarray(json.dumps(metadata)),
    )
    write_run_metadata(
        out_path.with_suffix(".run.json"),
        stage="controlled_synthetic_forecast_generation",
        args=args,
        inputs={"test_data": args.test_data},
        outputs={"forecast_test": out_path},
        contract=contract,
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
