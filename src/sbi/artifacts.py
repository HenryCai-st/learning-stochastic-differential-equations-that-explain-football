"""Metadata contracts for datasets, checkpoints, and experiment runs."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.simulators.model_voting import CONDITION_DIM, MAX_PARAM_DIM, MODEL_NAMES, MODEL_SPECS


ARTIFACT_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def git_commit() -> str | None:
    """Return the repository commit when Git is available."""
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_dirty() -> bool | None:
    """Report whether tracked files differ from the recorded commit."""
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def simulator_contract() -> dict[str, Any]:
    return {
        "model_names": list(MODEL_NAMES),
        "models": {
            name: {
                "param_dim": spec.param_dim,
                "low": spec.low.tolist(),
                "high": spec.high.tolist(),
                "log_scale": spec.log_scale.tolist(),
            }
            for name, spec in MODEL_SPECS.items()
        },
    }


def runtime_contract() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
    }


def file_descriptor(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    descriptor: dict[str, Any] = {"path": str(resolved)}
    if resolved.exists():
        stat = resolved.stat()
        descriptor.update({"exists": True, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    else:
        descriptor["exists"] = False
    return descriptor


def dataset_contract(data: Any) -> dict[str, Any]:
    """Build a stable contract from an NPZ-like object or loaded dataset."""
    def scalar(key: str, default: Any = None) -> Any:
        if hasattr(data, key):
            return _json_safe(getattr(data, key))
        if hasattr(data, "files") and key in data.files:
            return _json_safe(np.asarray(data[key]).item())
        return default

    tracks = data.tracks if hasattr(data, "tracks") else data["tracks"]
    model_names = data.model_names if hasattr(data, "model_names") else data["model_names"].tolist()
    return {
        "steps": int(tracks.shape[1]),
        "track_channels": int(tracks.shape[2]),
        "n_tracks": int(tracks.shape[0]),
        "dt": scalar("dt"),
        "T": scalar("T"),
        "seed": scalar("seed"),
        "condition_dim": CONDITION_DIM,
        "max_param_dim": MAX_PARAM_DIM,
        "model_names": list(model_names),
        "condition_sources": scalar("condition_sources"),
    }


def checkpoint_metadata(dataset: Any, args: Namespace) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "model_voting_ratio_checkpoint",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "dataset": dataset_contract(dataset),
        "simulators": simulator_contract(),
        "training": _json_safe(vars(args)),
        "runtime": runtime_contract(),
    }


def data_artifact_metadata(
    *,
    artifact_type: str,
    args: Namespace,
    contract: dict[str, Any],
    inputs: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Describe a generated NPZ artifact without changing its numeric payload."""
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "arguments": _json_safe(vars(args)),
        "inputs": {name: file_descriptor(value) for name, value in (inputs or {}).items()},
        "data_contract": _json_safe(contract),
        "simulators": simulator_contract(),
    }


def validate_checkpoint_contract(
    checkpoint: dict,
    *,
    steps: int,
    dt: float,
    model_names: list[str] | tuple[str, ...] = MODEL_NAMES,
) -> None:
    """Reject silent checkpoint/data protocol mismatches."""
    metadata = checkpoint.get("artifact_metadata")
    if metadata is None:
        raise ValueError("Checkpoint has no artifact_metadata; retrain it with the current pipeline.")
    contract = metadata.get("dataset", {})
    errors: list[str] = []
    if int(metadata.get("schema_version", -1)) != ARTIFACT_SCHEMA_VERSION:
        errors.append(
            f"schema checkpoint={metadata.get('schema_version')} current={ARTIFACT_SCHEMA_VERSION}"
        )
    if int(contract.get("steps", -1)) != int(steps):
        errors.append(f"steps checkpoint={contract.get('steps')} input={steps}")
    checkpoint_dt = contract.get("dt")
    if checkpoint_dt is not None and not np.isclose(float(checkpoint_dt), float(dt)):
        errors.append(f"dt checkpoint={checkpoint_dt} input={dt}")
    if list(contract.get("model_names", [])) != list(model_names):
        errors.append("model_names differ")
    if int(contract.get("condition_dim", -1)) != CONDITION_DIM:
        errors.append(f"condition_dim checkpoint={contract.get('condition_dim')} current={CONDITION_DIM}")
    if int(contract.get("max_param_dim", -1)) != MAX_PARAM_DIM:
        errors.append(f"max_param_dim checkpoint={contract.get('max_param_dim')} current={MAX_PARAM_DIM}")
    if metadata.get("simulators") != simulator_contract():
        errors.append("simulator priors or parameter schema differ")
    if errors:
        raise ValueError("Checkpoint/data contract mismatch: " + "; ".join(errors))


def write_run_metadata(
    path: str | Path,
    *,
    stage: str,
    args: Namespace,
    inputs: dict[str, str | Path] | None = None,
    outputs: dict[str, str | Path] | None = None,
    contract: dict[str, Any] | None = None,
    results: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "run_metadata",
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "arguments": _json_safe(vars(args)),
        "inputs": {name: file_descriptor(value) for name, value in (inputs or {}).items()},
        "outputs": {name: str(value) for name, value in (outputs or {}).items()},
        "data_contract": _json_safe(contract or {}),
        "simulators": simulator_contract(),
        "runtime": runtime_contract(),
        "results": _json_safe(results or {}),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
