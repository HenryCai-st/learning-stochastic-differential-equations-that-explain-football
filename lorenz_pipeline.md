# Lorenz-to-Football SDE Workflow

This document is the working pipeline for the project. The final goal is to
predict future football tracks as a probability distribution, not as one fixed
deterministic path.

The Lorenz system is the first controlled demo. It lets us finish the complete
machine-learning pipeline before replacing the simulator with a football
movement SDE such as correlated random walk or social force.

## Core Problem

An SDE does not produce one fixed trajectory from one parameter setting. The
same parameters can lead to different endings because of the Wiener noise.
Therefore, predicting only one parameter vector or one future path is
misleading.

The project target is:

```text
observed track -> probability distribution over SDE parameters
               -> probability distribution over future tracks
```

In Bayesian notation:

```text
p(theta | x_observed) proportional to p(x_observed | theta) p(theta)
```

where `theta` is the SDE parameter vector and `x_observed` is the observed
trajectory.

## Updated Pipeline

1. Simulate SDE trajectories from known parameters.
   - Lorenz demo: already available through `lorenz_dataset.npz`.
   - Football later: use CRW/social-force simulators.

2. Sample many parameter sets from a prior distribution.
   - Lorenz example: `(sigma, rho, beta, epsilon)`.
   - Football CRW example: `(kappa, v0, sigma_theta, sigma_v)`.
   - Use different initial states, because real tracking data also starts from
     different positions and velocities.

3. Build matched and mismatched pairs.
   - Positive pair: `(theta_A, track_A)` where `track_A` was generated from
     `theta_A`.
   - Negative pair: `(theta_A, track_B)` where `track_B` was generated from a
     different parameter set.

4. Train a binary classifier.
   - Input: trajectory encoding plus parameter encoding.
   - Output: probability that the pair is matched.
   - This is the contrastive-learning / classifier-based SBI step.
   - The classifier learns a compatibility score between parameters and tracks.

5. Use the trained classifier as a posterior scorer.
   - For one observed track, sample many candidate parameter sets.
   - Score each `(candidate theta, observed track)` pair.
   - Normalize or rank the scores to approximate likely parameter regions.

6. Simulate future paths.
   - Draw plausible parameters from the inferred distribution.
   - Simulate many future trajectories.
   - Report possible future locations as a distribution or heatmap.

## Files

- `lorenz_dataset.npz`
  - existing generated dataset
  - contains `trajectories`, `params`, and `labels`
  - current trajectory shape is `(samples, time, xy)`

- `src/data/lorenz_dataset.py`
  - `LorenzTrajectoryDataset`: loads `.npz` into PyTorch tensors
  - outputs trajectories as `(channels, time)`
  - normalizes trajectory coordinates to `[-1, 1]`
  - normalizes parameters with min-max scaling
  - supports train/val/test splits
  - `LorenzPairDataset`: creates matched and mismatched parameter/trajectory pairs for SBI

- `src/models/lorenz_models.py`
  - `TrajectoryEncoder1D`: shared 1D CNN encoder for trajectories
  - `LorenzRegimeClassifier`: predicts fixed-point vs chaotic regime
  - `LorenzParameterRegressor`: predicts normalized parameters
  - `LorenzRatioEstimator`: matched/mismatched classifier for simulation-based inference

- `scripts/train_lorenz.py`
  - common training script for the three demo tasks

## Data Shape

For the Lorenz demo, the current sequence input is:

```text
trajectory: (time, 2)
```

The dataloader converts it to PyTorch CNN format:

```text
trajectory: (channels, time)
```

For one training batch:

```text
trajectory batch: (batch, 2, time)
params batch:     (batch, 4)
label batch:      (batch,)
```

For future model-selection experiments with multiple SDE families, extend the
parameter input with a model ID:

```text
model_onehot = [1, 0, 0, 0]  # Brownian
model_onehot = [0, 1, 0, 0]  # OU velocity
model_onehot = [0, 0, 1, 0]  # CRW
model_onehot = [0, 0, 0, 1]  # social force
```

Then concatenate:

```text
parameter_input = concat(model_onehot, normalized_parameters)
```

or pass the model ID through an embedding/MLP before concatenation.

## Environment

The repository currently has a `.venv`, but it points to a missing Python
installation on this machine. Recreate it before training:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If `py` is not available, install Python 3.11 or 3.12 first, then recreate the
virtual environment. PyTorch support for very new Python versions can lag, so
3.11 or 3.12 is safer than 3.13.

## Training Commands

Regime classifier:

```powershell
python scripts\train_lorenz.py --task regime --epochs 20
```

Parameter regression:

```powershell
python scripts\train_lorenz.py --task params --epochs 30
```

SBI ratio estimator:

```powershell
python scripts\train_lorenz.py --task ratio --epochs 30
```

This is the most important task for the final project because it matches the
group workflow:

```text
(parameter set A, track B) -> false
(parameter set B, track B) -> true
```

Quick smoke test:

```powershell
python scripts\train_lorenz.py --task regime --epochs 1 --limit-train 96 --limit-val 32 --device cpu
```

Training outputs are written under:

```text
outputs/lorenz_training/<task>/
```

Each run saves:

- `history.csv`
- `best.pt`

## Runnable NumPy Demo

Because the current `.venv` points to a missing Python installation and the
bundled runtime does not include PyTorch, a NumPy-only demo is also available:

```powershell
& "C:\Users\liuyo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" scripts\lorenz_sbi_numpy_demo.py --epochs 500 --out-dir outputs\lorenz_sbi_demo
```

This demo completes the project-shaped workflow without PyTorch:

```text
load Lorenz dataset
-> build matched/mismatched pairs
-> train binary compatibility classifier
-> score candidate parameters for one observed track
-> simulate future paths from high-scoring candidates
```

Latest demo result:

```text
train accuracy = 0.7919
val accuracy   = 0.7400
```

Generated outputs:

```text
outputs/lorenz_sbi_demo/history.csv
outputs/lorenz_sbi_demo/top_candidates.csv
outputs/lorenz_sbi_demo/training_loss.svg
outputs/lorenz_sbi_demo/posterior_scores.svg
outputs/lorenz_sbi_demo/future_paths.svg
outputs/lorenz_sbi_demo/summary.json
```

## Recommended Walkthrough

Start with `regime`.

This tests the easiest question: can the model distinguish fixed-point and
chaotic trajectories from the trajectory alone?

Then run `params`.

This tests direct inverse prediction:

```text
trajectory -> (sigma, rho, beta, epsilon)
```

This is useful, but it is not fully Bayesian because it predicts one parameter
vector.

Finally run `ratio`.

This matches the project PDF more closely:

```text
(parameter, trajectory) -> matched or mismatched
```

The classifier score can later be used as a compatibility score over many
candidate parameter samples. That gives a distribution of plausible parameters
instead of one deterministic answer.

## Practical Milestones

Milestone 1: Lorenz sanity check.

- Load `lorenz_dataset.npz`.
- Train `regime` for 1-5 epochs.
- Confirm loss decreases and accuracy is above chance.

Milestone 2: Lorenz direct inverse baseline.

- Train `params`.
- Report L1 error for normalized parameters.
- Use this as a simple baseline, not the final probabilistic method.

Milestone 3: Lorenz contrastive SBI.

- Train `ratio`.
- For one validation trajectory, sample many candidate parameter vectors.
- Score all candidates with the ratio estimator.
- Visualize the best candidates or score distribution.

Milestone 4: Football CRW replacement.

- Replace Lorenz trajectories with CRW trajectories.
- Keep the same dataloader pattern.
- Keep the same pair classifier training objective.

Milestone 5: Future-path prediction.

- Infer likely parameters for an observed partial track.
- Simulate many continuations.
- Visualize future position probability, for example as a density map.

## How This Transfers To Football

When switching to the football random-walk model, keep the same structure:

- replace Lorenz simulator with CRW or social-force simulator
- keep trajectories as `(time, coordinates)`
- keep parameter vectors as a fixed-size array
- reuse the dataset pattern
- reuse the ratio-estimator training task

The football version should first infer parameters for the correlated random
walk model:

```text
(kappa, v0, sigma_theta, sigma_v)
```

After that works, add model selection between Brownian, OU velocity, CRW, and
social-force models.

## Presentation Outline

1. State the problem: predict football/player tracks using SDEs.
2. Explain why SDE prediction is probabilistic: random noise makes trajectories
   diverge even with the same parameters.
3. Introduce the Bayesian target: infer `p(theta | observed track)`.
4. State the practical goal: predict possible future locations, not one point.
5. Explain simulation-based inference:
   - simulate many trajectories from known parameters
   - train a classifier on matched/mismatched pairs
   - use classifier scores to identify plausible parameters
6. Show the model architecture:
   - trajectory encoder
   - parameter encoder
   - binary classifier head
   - output shape: one match logit/probability
7. Show visualization:
   - observed track
   - candidate simulated tracks
   - inferred parameter distribution
   - future location distribution
