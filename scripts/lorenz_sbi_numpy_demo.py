from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PARAM_NAMES = ("sigma", "rho", "beta", "epsilon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runnable NumPy-only Lorenz SBI demo."
    )
    parser.add_argument("--dataset", default="lorenz_dataset.npz")
    parser.add_argument("--out-dir", default="outputs/lorenz_sbi_demo")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--future-samples", type=int, default=20)
    return parser.parse_args()


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    trajectories = raw["trajectories"]
    if trajectories.dtype == object:
        trajectories = np.stack(
            [np.asarray(item, dtype=np.float32) for item in trajectories]
        )
    else:
        trajectories = trajectories.astype(np.float32)
    params = raw["params"].astype(np.float32)
    labels = raw["labels"].astype(np.int64)
    return trajectories, params, labels


def split_indices(n_items: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_items)
    train_end = int(0.8 * n_items)
    val_end = int(0.9 * n_items)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def summarize_trajectory(traj: np.ndarray, max_points: int) -> np.ndarray:
    if len(traj) > max_points:
        pick = np.linspace(0, len(traj) - 1, max_points).astype(np.int64)
        traj = traj[pick]

    steps = np.diff(traj, axis=0)
    step_norm = np.linalg.norm(steps, axis=1)
    displacement = traj[-1] - traj[0]
    centered = traj - traj.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(covariance)

    return np.array(
        [
            traj[:, 0].mean(),
            traj[:, 1].mean(),
            traj[:, 0].std(),
            traj[:, 1].std(),
            traj[:, 0].min(),
            traj[:, 1].min(),
            traj[:, 0].max(),
            traj[:, 1].max(),
            displacement[0],
            displacement[1],
            step_norm.sum(),
            step_norm.mean(),
            step_norm.std(),
            step_norm.max(),
            eigvals[0],
            eigvals[1],
        ],
        dtype=np.float64,
    )


def transform_params(params: np.ndarray) -> np.ndarray:
    transformed = params.astype(np.float64).copy()
    transformed[:, 1] = np.log1p(transformed[:, 1])
    transformed[:, 3] = np.log1p(transformed[:, 3])
    return transformed


def standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / std


def make_pairs(
    param_features: np.ndarray,
    traj_features: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    local_params = param_features[indices]
    local_trajs = traj_features[indices]

    shuffled = indices.copy()
    while True:
        rng.shuffle(shuffled)
        if np.all(shuffled != indices):
            break

    positive = pair_design_matrix(local_params, local_trajs)
    negative = pair_design_matrix(local_params, traj_features[shuffled])
    x = np.vstack([positive, negative])
    y = np.concatenate([np.ones(len(indices)), np.zeros(len(indices))])

    order = rng.permutation(len(y))
    return x[order], y[order]


def pair_design_matrix(param_features: np.ndarray, traj_features: np.ndarray) -> np.ndarray:
    """
    Build compatibility features for a parameter/trajectory pair.

    A plain concatenation has identical positive and negative marginals, so a
    linear classifier cannot learn much. Interaction terms let this NumPy demo
    approximate the role of a neural trajectory encoder plus parameter encoder.
    """

    interactions = (
        param_features[:, :, None] * traj_features[:, None, :]
    ).reshape(len(param_features), -1)
    return np.concatenate([param_features, traj_features, interactions], axis=1)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def binary_metrics(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    logits = x @ weights[:-1] + weights[-1]
    prob = sigmoid(logits)
    eps = 1e-8
    loss = -np.mean(y * np.log(prob + eps) + (1.0 - y) * np.log(1.0 - prob + eps))
    acc = np.mean((prob >= 0.5) == y)
    return {"loss": float(loss), "accuracy": float(acc)}


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    weights = np.zeros(x_train.shape[1] + 1, dtype=np.float64)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        logits = x_train @ weights[:-1] + weights[-1]
        prob = sigmoid(logits)
        error = prob - y_train

        grad_w = (x_train.T @ error) / len(y_train) + l2 * weights[:-1]
        grad_b = error.mean()

        weights[:-1] -= lr * grad_w
        weights[-1] -= lr * grad_b

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            train = binary_metrics(x_train, y_train, weights)
            val = binary_metrics(x_val, y_val, weights)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train["loss"],
                    "train_accuracy": train["accuracy"],
                    "val_loss": val["loss"],
                    "val_accuracy": val["accuracy"],
                }
            )

    return weights, history


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = values / max(temperature, 1e-8)
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    return exp / np.maximum(exp.sum(), 1e-12)


def score_candidates(
    weights: np.ndarray,
    candidate_param_features: np.ndarray,
    observed_traj_feature: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.repeat(observed_traj_feature[None, :], len(candidate_param_features), axis=0)
    x = pair_design_matrix(candidate_param_features, observed)
    x = standardize(x, x_mean, x_std)
    logits = x @ weights[:-1] + weights[-1]
    posterior = softmax(logits)
    return logits, posterior


def simulate_lorenz_xy(
    params: np.ndarray,
    seed: int,
    steps: int = 300,
    dt: float = 0.01,
    y0: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    sigma, rho, beta, epsilon = [float(v) for v in params]
    rng = np.random.default_rng(seed)
    xyz = np.zeros((steps, 3), dtype=np.float64)
    xyz[0] = np.asarray(y0, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)

    for i in range(steps - 1):
        x, y, z = xyz[i]
        drift = np.array(
            [
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
            ],
            dtype=np.float64,
        )
        noise = epsilon * sqrt_dt * rng.standard_normal(3)
        xyz[i + 1] = xyz[i] + drift * dt + noise
        xyz[i + 1] = np.clip(xyz[i + 1], -1e4, 1e4)

    return xyz[:, :2]


def normalize_path(path: np.ndarray) -> np.ndarray:
    minimum = path.min(axis=0, keepdims=True)
    maximum = path.max(axis=0, keepdims=True)
    return (path - minimum) / np.maximum(maximum - minimum, 1e-8)


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def write_top_candidates_csv(
    path: Path,
    candidate_indices: np.ndarray,
    params: np.ndarray,
    logits: np.ndarray,
    posterior: np.ndarray,
    top_order: np.ndarray,
) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "dataset_index", *PARAM_NAMES, "logit", "posterior_weight"])
        for rank, item in enumerate(top_order, 1):
            idx = int(candidate_indices[item])
            writer.writerow(
                [
                    rank,
                    idx,
                    *[float(v) for v in params[idx]],
                    float(logits[item]),
                    float(posterior[item]),
                ]
            )


def svg_header(width: int = 800, height: int = 520) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def svg_footer() -> str:
    return "</svg>"


def write_loss_svg(path: Path, history: list[dict[str, float]]) -> None:
    width, height = 800, 480
    pad = 60
    epochs = np.array([row["epoch"] for row in history], dtype=np.float64)
    train = np.array([row["train_loss"] for row in history], dtype=np.float64)
    val = np.array([row["val_loss"] for row in history], dtype=np.float64)
    y_min = min(train.min(), val.min())
    y_max = max(train.max(), val.max())
    y_span = max(y_max - y_min, 1e-8)

    def points(values: np.ndarray) -> str:
        xs = pad + (epochs - epochs.min()) / max(epochs.max() - epochs.min(), 1e-8) * (width - 2 * pad)
        ys = height - pad - (values - y_min) / y_span * (height - 2 * pad)
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    lines = svg_header(width, height)
    lines += [
        '<text x="400" y="32" text-anchor="middle" font-size="20">Lorenz SBI classifier loss</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#444"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#444"/>',
        f'<polyline points="{points(train)}" fill="none" stroke="#1f77b4" stroke-width="2"/>',
        f'<polyline points="{points(val)}" fill="none" stroke="#d62728" stroke-width="2"/>',
        '<text x="650" y="72" fill="#1f77b4" font-size="14">train loss</text>',
        '<text x="650" y="94" fill="#d62728" font-size="14">val loss</text>',
        f'<text x="{pad}" y="{height-20}" font-size="12">epoch</text>',
        f'<text x="10" y="{pad}" font-size="12">loss</text>',
        svg_footer(),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_posterior_svg(
    path: Path,
    candidate_params: np.ndarray,
    posterior: np.ndarray,
    top_order: np.ndarray,
    observed_params: np.ndarray,
) -> None:
    width, height = 800, 520
    pad = 70
    rho = candidate_params[:, 1]
    epsilon = candidate_params[:, 3]
    rho_min, rho_max = rho.min(), rho.max()
    eps_min, eps_max = epsilon.min(), epsilon.max()

    def sx(value: float) -> float:
        return pad + (value - rho_min) / max(rho_max - rho_min, 1e-8) * (width - 2 * pad)

    def sy(value: float) -> float:
        return height - pad - (value - eps_min) / max(eps_max - eps_min, 1e-8) * (height - 2 * pad)

    lines = svg_header(width, height)
    lines += [
        '<text x="400" y="32" text-anchor="middle" font-size="20">Posterior candidate scores</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#444"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#444"/>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="14">rho</text>',
        '<text x="20" y="260" transform="rotate(-90 20 260)" text-anchor="middle" font-size="14">epsilon</text>',
    ]

    scaled = posterior / max(posterior.max(), 1e-12)
    for p, weight in zip(candidate_params, scaled):
        r = 2.0 + 7.0 * np.sqrt(weight)
        opacity = 0.12 + 0.75 * weight
        lines.append(
            f'<circle cx="{sx(float(p[1])):.1f}" cy="{sy(float(p[3])):.1f}" r="{r:.2f}" fill="#4c78a8" opacity="{opacity:.3f}"/>'
        )

    for rank, item in enumerate(top_order[:10], 1):
        p = candidate_params[item]
        lines.append(
            f'<circle cx="{sx(float(p[1])):.1f}" cy="{sy(float(p[3])):.1f}" r="6" fill="none" stroke="#f58518" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{sx(float(p[1])) + 7:.1f}" y="{sy(float(p[3])) - 7:.1f}" font-size="10" fill="#f58518">{rank}</text>'
        )

    lines.append(
        f'<path d="M {sx(float(observed_params[1]))-7:.1f} {sy(float(observed_params[3])):.1f} L {sx(float(observed_params[1]))+7:.1f} {sy(float(observed_params[3])):.1f} M {sx(float(observed_params[1])):.1f} {sy(float(observed_params[3]))-7:.1f} L {sx(float(observed_params[1])):.1f} {sy(float(observed_params[3]))+7:.1f}" stroke="#d62728" stroke-width="3"/>'
    )
    lines.append('<text x="590" y="72" font-size="13" fill="#d62728">red cross = true observed params</text>')
    lines.append('<text x="590" y="92" font-size="13" fill="#f58518">orange = top candidates</text>')
    lines.append(svg_footer())
    path.write_text("\n".join(lines), encoding="utf-8")


def write_future_paths_svg(path: Path, observed: np.ndarray, futures: list[np.ndarray]) -> None:
    width, height = 760, 520
    pad = 55

    all_paths = [normalize_path(observed)] + [normalize_path(f) for f in futures]

    def polyline(points: np.ndarray) -> str:
        xs = pad + points[:, 0] * (width - 2 * pad)
        ys = height - pad - points[:, 1] * (height - 2 * pad)
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    lines = svg_header(width, height)
    lines += [
        '<text x="380" y="32" text-anchor="middle" font-size="20">Future paths from likely parameters</text>',
        f'<rect x="{pad}" y="{pad}" width="{width-2*pad}" height="{height-2*pad}" fill="#f7f7f7" stroke="#444"/>',
    ]
    for f in all_paths[1:]:
        lines.append(
            f'<polyline points="{polyline(f)}" fill="none" stroke="#54a24b" stroke-width="1.2" opacity="0.35"/>'
        )
    obs = all_paths[0]
    lines.append(
        f'<polyline points="{polyline(obs)}" fill="none" stroke="#1f77b4" stroke-width="2.5" opacity="0.9"/>'
    )
    lines.append('<text x="590" y="72" font-size="13" fill="#1f77b4">blue = observed track</text>')
    lines.append('<text x="590" y="92" font-size="13" fill="#54a24b">green = sampled futures</text>')
    lines.append(svg_footer())
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories, params, labels = load_npz(args.dataset)
    train_idx, val_idx, test_idx = split_indices(len(params), args.seed)

    traj_features = np.stack(
        [summarize_trajectory(traj, args.max_points) for traj in trajectories]
    )
    param_features = transform_params(params)

    x_train_raw, y_train = make_pairs(param_features, traj_features, train_idx, rng)
    x_val_raw, y_val = make_pairs(param_features, traj_features, val_idx, rng)
    x_mean, x_std = standardize_fit(x_train_raw)
    x_train = standardize(x_train_raw, x_mean, x_std)
    x_val = standardize(x_val_raw, x_mean, x_std)

    weights, history = train_logistic_regression(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
    )

    observed_index = int(val_idx[0])
    logits, posterior = score_candidates(
        weights=weights,
        candidate_param_features=param_features,
        observed_traj_feature=traj_features[observed_index],
        x_mean=x_mean,
        x_std=x_std,
    )
    top_order = np.argsort(-posterior)[: args.top_k]

    future_paths: list[np.ndarray] = []
    for i, candidate_item in enumerate(top_order[: args.future_samples]):
        future_paths.append(
            simulate_lorenz_xy(
                params=params[candidate_item],
                seed=args.seed + 1000 + i,
                steps=400,
            )
        )

    write_history_csv(out_dir / "history.csv", history)
    write_top_candidates_csv(
        out_dir / "top_candidates.csv",
        candidate_indices=np.arange(len(params)),
        params=params,
        logits=logits,
        posterior=posterior,
        top_order=top_order,
    )
    write_loss_svg(out_dir / "training_loss.svg", history)
    write_posterior_svg(
        out_dir / "posterior_scores.svg",
        candidate_params=params,
        posterior=posterior,
        top_order=top_order,
        observed_params=params[observed_index],
    )
    write_future_paths_svg(
        out_dir / "future_paths.svg",
        observed=trajectories[observed_index],
        futures=future_paths,
    )

    summary = {
        "dataset": str(args.dataset),
        "n_samples": int(len(params)),
        "train_pairs": int(len(y_train)),
        "val_pairs": int(len(y_val)),
        "observed_index": observed_index,
        "observed_params": {
            name: float(value) for name, value in zip(PARAM_NAMES, params[observed_index])
        },
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "top_candidate_index": int(top_order[0]),
        "top_candidate_params": {
            name: float(value) for name, value in zip(PARAM_NAMES, params[top_order[0]])
        },
        "outputs": [
            "history.csv",
            "top_candidates.csv",
            "training_loss.svg",
            "posterior_scores.svg",
            "future_paths.svg",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
