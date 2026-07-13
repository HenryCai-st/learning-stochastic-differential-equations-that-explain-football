"""Multi-chain posterior diagnostics used by controlled validation."""

from __future__ import annotations

import numpy as np


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Compute split R-hat for arrays shaped (chains, draws, parameters)."""
    values = np.asarray(chains, dtype=np.float64)
    n_chains, n_draws, _ = values.shape
    half = n_draws // 2
    if n_chains < 2 or half < 2:
        return np.full(values.shape[2], np.nan)
    split = np.concatenate([values[:, :half], values[:, -half:]], axis=0)
    chain_means = split.mean(axis=1)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half * chain_means.var(axis=0, ddof=1)
    variance = ((half - 1) / half) * within + between / half
    return np.sqrt(np.divide(variance, within, out=np.ones_like(variance), where=within > 0))


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Estimate bulk ESS with an initial-positive autocorrelation sequence."""
    values = np.asarray(chains, dtype=np.float64)
    n_chains, n_draws, n_params = values.shape
    total = n_chains * n_draws
    output = np.empty(n_params, dtype=np.float64)
    for dim in range(n_params):
        centered = values[:, :, dim] - values[:, :, dim].mean(axis=1, keepdims=True)
        variance = np.mean(centered * centered)
        if variance <= 1e-15:
            output[dim] = float(total)
            continue
        rho_sum = 0.0
        previous_pair = np.inf
        for lag in range(1, n_draws - 1, 2):
            rho_a = np.mean(centered[:, :-lag] * centered[:, lag:]) / variance
            next_lag = lag + 1
            rho_b = np.mean(centered[:, :-next_lag] * centered[:, next_lag:]) / variance
            pair = min(float(rho_a + rho_b), previous_pair)
            if pair <= 0:
                break
            rho_sum += pair
            previous_pair = pair
        output[dim] = min(float(total), total / max(1.0 + 2.0 * rho_sum, 1.0))
    return output


def interval_summary(samples: np.ndarray, true_theta: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize posterior location, intervals, and truth inclusion."""
    flat = np.asarray(samples).reshape(-1, samples.shape[-1])
    truth = np.asarray(true_theta)
    result = {
        "mean": flat.mean(axis=0),
        "median": np.median(flat, axis=0),
        "bias": flat.mean(axis=0) - truth,
    }
    for level in (0.5, 0.8, 0.9):
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(flat, [tail, 1.0 - tail], axis=0)
        key = str(int(level * 100))
        result[f"low_{key}"] = low
        result[f"high_{key}"] = high
        result[f"width_{key}"] = high - low
        result[f"covered_{key}"] = (truth >= low) & (truth <= high)
    return result
