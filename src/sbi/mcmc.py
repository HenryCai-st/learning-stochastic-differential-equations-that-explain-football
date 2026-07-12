"""Model-specific priors and random-walk Metropolis-Hastings."""

from __future__ import annotations

import numpy as np
import torch

from src.sbi.ratio_model import ModelVotingRatioClassifier
from src.sbi.scoring import score_params
from src.simulators.model_voting import MODEL_SPECS


def log_prior(model_name: str, theta: np.ndarray) -> float:
    """Evaluate the model-specific log prior up to an irrelevant constant."""
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
    """Choose the existing Gaussian proposal scale for one model."""
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
    """Run the existing random-walk Metropolis-Hastings implementation."""
    proposal_scale = proposal_scale_for_model(model_name)

    def log_target(theta: np.ndarray) -> float:
        prior = log_prior(model_name, theta)
        if not np.isfinite(prior):
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
        return float(prior + logit)

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
