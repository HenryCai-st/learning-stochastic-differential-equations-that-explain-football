from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Split real football windows into prefix and held-out suffix tracks.")
    parser.add_argument("--input", default="data/real_football_windows.npz")
    parser.add_argument("--prefix-seconds", type=float, default=2.0)
    parser.add_argument("--out", default="data/real_football_prefix_suffix_windows.npz")
    args = parser.parse_args()

    source_path = Path(args.input)
    if not source_path.exists():
        raise FileNotFoundError(f"Input windows not found: {source_path}. Run extract_football_windows.py first.")

    data = np.load(source_path, allow_pickle=True)
    tracks = data["tracks"].astype(np.float32)
    dt = float(data["dt"])
    prefix_steps = int(round(args.prefix_seconds / dt))
    if prefix_steps < 2:
        raise ValueError("prefix_steps must be at least 2.")
    if prefix_steps >= tracks.shape[1]:
        raise ValueError("prefix must be shorter than the full window.")

    prefix_tracks = tracks[:, :prefix_steps]
    suffix_tracks = tracks[:, prefix_steps:]
    prefix_end = prefix_tracks[:, -1]
    suffix_end = tracks[:, -1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        prefix_tracks=prefix_tracks,
        suffix_tracks=suffix_tracks,
        full_tracks=tracks,
        y0=tracks[:, 0],
        prefix_end=prefix_end,
        suffix_end=suffix_end,
        target_for_evaluation=suffix_end,
        dt=dt,
        prefix_steps=prefix_steps,
        suffix_steps=suffix_tracks.shape[1],
        full_steps=tracks.shape[1],
        meta=data["meta"] if "meta" in data.files else np.arange(len(tracks)),
        source=str(source_path),
    )
    print(json.dumps({
        "out": str(out),
        "windows": int(len(tracks)),
        "dt": dt,
        "prefix_steps": int(prefix_steps),
        "suffix_steps": int(suffix_tracks.shape[1]),
    }, indent=2))


if __name__ == "__main__":
    main()
