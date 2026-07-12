"""
src/legacy/inference/mcmc.py
=====================
Historical Random-Walk Metropolis-Hastings prototype. This module is retained
for reference only and is not imported by the active model-voting pipeline.

The sampler is decoupled from the model/data: it takes a generic
``log_target_fn`` callable so it can be reused with any likelihood surrogate.

Public API
----------
rwmh_mcmc(log_target_fn, theta_init, n_steps, step_size, rng)
    -> (chain, accept_rate)

make_log_target(weights, obs_traj_feat, x_mean, x_std, prior_bounds)
    -> callable  (builds the log-target closure for the trained classifier)

log_prior(theta_raw, prior_bounds)
    -> float
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Prior
# ─────────────────────────────────────────────────────────────────────────────

def log_prior(theta_raw: np.ndarray, prior_bounds: np.ndarray) -> float:
    """
    Log of a uniform prior over the parameter box.

    Returns 0 if theta_raw is inside the box, −∞ otherwise.

    Parameters
    ----------
    theta_raw    : (4,) raw parameter vector
    prior_bounds : (4, 2)  array of [[lo, hi], ...]
    """
    lo, hi = prior_bounds[:, 0], prior_bounds[:, 1]
    if np.any(theta_raw < lo) or np.any(theta_raw > hi):
        return -np.inf
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Log-target factory
# ─────────────────────────────────────────────────────────────────────────────

def make_log_target(
    weights:       np.ndarray,       # (D,)  trained classifier weights (bias is last)
    obs_traj_feat: np.ndarray,       # (F_t,) feature vector of the observed trajectory
    x_mean:        np.ndarray,       # (D-1,) standardisation mean
    x_std:         np.ndarray,       # (D-1,) standardisation std
    prior_bounds:  np.ndarray,       # (4, 2)
):
    """
    Return a closure  log_target(theta_raw) -> float.

    The log-target combines:
      - the trained classifier's logit as a log-likelihood surrogate
        (justified by the likelihood-ratio trick from NRE / SNRE literature)
      - the log-prior (uniform box)

    The RWMH sampler then explores  p(θ | x_obs)  ∝  exp(log_target(θ)).
    """
    try:
        from src.utils.features import pair_design_matrix, standardize, transform_params
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This legacy inference prototype requires the removed "
            "src.utils.features module and is retained for reference only."
        ) from exc

    obs_feat_1d = obs_traj_feat.astype(np.float64).reshape(1, -1)

    def log_target(theta_raw: np.ndarray) -> float:
        lp = log_prior(theta_raw, prior_bounds)
        if not np.isfinite(lp):
            return -np.inf

        # Build the design matrix row for this (theta, x_obs) pair
        pf_raw = transform_params(theta_raw[None, :]).astype(np.float64)  # (1, 4)
        x_raw  = pair_design_matrix(pf_raw, obs_feat_1d)
        x_std_ = standardize(x_raw, x_mean, x_std)
        logit  = float(x_std_ @ weights[:-1] + weights[-1])

        return logit + lp

    return log_target


# ─────────────────────────────────────────────────────────────────────────────
# Random-Walk Metropolis-Hastings
# ─────────────────────────────────────────────────────────────────────────────

def rwmh_mcmc(
    log_target_fn,
    theta_init:  np.ndarray,
    n_steps:     int,
    step_size:   float,
    rng:         np.random.Generator,
) -> tuple[np.ndarray, float]:
    """
    Random-Walk Metropolis-Hastings sampler.

    Proposal distribution:  θ' = θ + step_size · N(0, I)

    Because the proposal is symmetric, the Metropolis-Hastings acceptance ratio
    reduces to:

        α = min(1, exp( log_target(θ') − log_target(θ) ))

    Parameters
    ----------
    log_target_fn : callable(theta: np.ndarray) -> float
        Log of the (unnormalised) target density; return −∞ to reject.
    theta_init    : (dim,)  initial parameter vector (raw space)
    n_steps       : total number of MCMC proposals (including burn-in)
    step_size     : Gaussian proposal std per dimension
    rng           : seeded numpy Generator

    Returns
    -------
    chain         : np.ndarray (n_steps, dim)  — full Markov chain
    accept_rate   : float  — fraction of proposals accepted
    """
    dim   = len(theta_init)
    chain = np.empty((n_steps, dim), dtype=np.float64)
    theta = theta_init.copy().astype(np.float64)
    log_p = log_target_fn(theta)
    n_acc = 0

    for i in range(n_steps):
        theta_prop = theta + step_size * rng.standard_normal(dim)
        log_p_prop = log_target_fn(theta_prop)

        log_alpha = log_p_prop - log_p
        if np.log(rng.uniform()) < log_alpha:
            theta = theta_prop
            log_p = log_p_prop
            n_acc += 1

        chain[i] = theta

    return chain, n_acc / n_steps
