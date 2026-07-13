"""Numerically stable model-evidence helpers."""

from __future__ import annotations

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    """Convert comparable log-evidence values into normalized weights."""
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def logmeanexp(values: np.ndarray) -> float:
    """Stable log of the arithmetic mean of exp(values)."""
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return maximum + float(np.log(np.mean(np.exp(values - maximum))))
