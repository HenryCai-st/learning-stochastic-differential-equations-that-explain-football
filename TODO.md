# Current TODO: Probabilistic SDE Track Prediction

## 1. Finish Lorenz Demo First

- [x] Use existing `lorenz_dataset.npz`.
- [x] Add runnable matched/mismatched SBI demo.
- [x] Save `history.csv`.
- [x] Add loss visualization as `training_loss.svg`.
- [ ] Repair PyTorch environment.
- [ ] Train `regime` classifier as a quick sanity check.
- [ ] Train `params` regressor as a deterministic baseline.
- [ ] Train neural `ratio` classifier as the main SBI/contrastive-learning demo.
- [ ] Save `best.pt` for each PyTorch run.

## 2. Lorenz Posterior-Scoring Demo

- [x] Pick one validation trajectory as the observed track.
- [x] Sample candidate parameter sets from the dataset.
- [x] Score `(candidate parameter, observed track)` with the demo ratio estimator.
- [x] Sort candidates by score.
- [x] Visualize likely parameter regions as `posterior_scores.svg`.
- [x] Simulate future trajectories from the best-scoring candidates.
- [x] Visualize future paths as `future_paths.svg`.

## 3. Dataset And Dataloader

- [x] Keep trajectory sequence input as `(time, coordinates)`.
- [x] Convert to PyTorch format `(channels, time)` inside the dataloader.
- [x] Normalize trajectories to `[-1, 1]` in the PyTorch dataloader.
- [x] Normalize parameters with training-set statistics in the PyTorch dataloader.
- [x] Use `log1p` transforms in the NumPy demo for wide/noise-like parameters.
- [ ] For multiple SDE model families, concatenate model one-hot encoding with the parameter vector.

## 4. Model Architecture

- [x] Define PyTorch 1D CNN trajectory encoder.
- [x] Define PyTorch parameter encoder inside the ratio estimator.
- [x] Define PyTorch ratio-estimator head outputting one logit.
- [x] Implement NumPy interaction-feature classifier as a runnable fallback demo.
- [x] Positive label: parameter set and trajectory match.
- [x] Negative label: parameter set and trajectory do not match.

## 5. Switch To Football Random-Walk Model

- [ ] Start with correlated random walk (CRW), not full social force.
- [ ] Parameters: `(kappa, v0, sigma_theta, sigma_v)`.
- [ ] Use different initial positions and velocities.
- [ ] Generate synthetic CRW datasets from parameter priors.
- [ ] Reuse the same dataloader and ratio-estimator training logic.

## 6. Future-Path Prediction

- [x] For one observed Lorenz track, infer a posterior-like distribution over parameters.
- [x] Sample plausible Lorenz parameters from the inferred distribution.
- [x] Simulate many possible future Lorenz paths.
- [x] Visualize future paths.
- [ ] Repeat with football CRW.
- [ ] Visualize football future position as a distribution or heatmap.

## 7. Presentation Checklist

- [x] State the problem: predict tracks with SDEs.
- [x] Explain why the output must be probabilistic.
- [x] Introduce Bayesian inference: `p(theta | track)`.
- [x] Explain simulation-based inference and classifier-based ratio learning.
- [x] Show the architecture: trajectory encoder, parameter encoder, binary head.
- [x] Generate visualizations: observed path, candidate paths, parameter distribution, future path distribution.
