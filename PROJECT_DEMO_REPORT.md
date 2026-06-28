# Project Demo Report: Simulation-Based Inference for SDE Tracks

## 1. Project Goal

The project goal is to predict football/player tracks with stochastic
differential equations (SDEs). Because an SDE contains random noise, the same
parameter setting can generate different trajectories. Therefore, the correct
target is not one fixed future path, but a probability distribution over future
paths.

The intended inference chain is:

```text
observed track
-> probability distribution over SDE parameters
-> probability distribution over future tracks
```

In Bayesian form:

```text
p(theta | x_observed) proportional to p(x_observed | theta) p(theta)
```

where:

- `theta` is the SDE parameter vector
- `x_observed` is the observed trajectory
- `p(theta | x_observed)` is the posterior distribution we want

## 2. Why We Start With Lorenz

The final project should use football movement models, for example correlated
random walk (CRW) or social-force SDEs. However, Lorenz is a useful controlled
demo because:

- we already have a generated dataset
- the parameters are known
- trajectories are stochastic
- the inverse problem is similar: infer likely parameters from a trajectory

Lorenz is not the final scientific model. It is the pipeline test before
switching to football tracks.

## 3. Completed Demo

A runnable NumPy-only demo was added:

```text
scripts/lorenz_sbi_numpy_demo.py
```

It avoids the currently broken PyTorch environment and demonstrates the core
project idea end-to-end:

1. Load `lorenz_dataset.npz`.
2. Extract summary features from each trajectory.
3. Transform Lorenz parameters `(sigma, rho, beta, epsilon)`.
4. Build positive and negative pairs:
   - positive: `(theta_A, track_A)`
   - negative: `(theta_A, track_B)`
5. Train a binary classifier to identify matched pairs.
6. Pick one validation trajectory as the observed track.
7. Score candidate parameters with the classifier.
8. Treat scores as a posterior-like compatibility distribution.
9. Simulate future paths from high-scoring candidates.
10. Save CSV and SVG outputs.

This matches the group workflow:

```text
parameter set A + track B -> false
parameter set B + track B -> true
```

## 4. Command To Run

From the repository root:

```powershell
& "C:\Users\liuyo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" scripts\lorenz_sbi_numpy_demo.py --epochs 500 --out-dir outputs\lorenz_sbi_demo
```

After the project Python environment is repaired, the shorter version should
also work:

```powershell
python scripts\lorenz_sbi_numpy_demo.py --epochs 500 --out-dir outputs\lorenz_sbi_demo
```

## 5. Demo Outputs

The demo writes files to:

```text
outputs/lorenz_sbi_demo/
```

Generated files:

- `summary.json`
  - summary of dataset size, observed sample, final accuracy, and top candidate
- `history.csv`
  - train/validation loss and accuracy during classifier training
- `top_candidates.csv`
  - highest-scoring parameter candidates for the observed trajectory
- `training_loss.svg`
  - training and validation loss visualization
- `posterior_scores.svg`
  - candidate parameter scores in `(rho, epsilon)` space
- `future_paths.svg`
  - observed path plus future paths simulated from likely parameters

## 6. Current Demo Result

The latest run used:

```text
epochs = 500
dataset samples = 1000
train pairs = 1600
validation pairs = 200
```

Final classifier metrics:

```text
train loss     = 0.4594
train accuracy = 0.7919
val loss       = 0.4972
val accuracy   = 0.7400
```

This means the classifier learned a real compatibility signal between
parameters and trajectories. It is not a final neural SBI model yet, but it is
enough for a project demo.

Observed validation sample:

```text
index   = 8
sigma   = 10.7267
rho     = 34.4614
beta    = 3.2570
epsilon = 1.2982
```

Highest-scoring candidate:

```text
index   = 85
sigma   = 19.6578
rho     = 49.0131
beta    = 4.2075
epsilon = 1.3760
```

The highest-scoring candidate is not identical to the true parameter. This is
expected in a stochastic inverse problem. The point is to recover a plausible
parameter region, not one exact parameter vector.

## 7. How The Demo Classifier Works

For each trajectory, the script computes trajectory summaries such as:

- coordinate mean and standard deviation
- coordinate min and max
- total path length
- average step size
- max step size
- displacement
- covariance eigenvalues

For each parameter vector, the script uses:

```text
(sigma, log1p(rho), beta, log1p(epsilon))
```

The classifier input is built from:

```text
parameter features
trajectory features
parameter-feature x trajectory-feature interactions
```

The interaction terms are important. A simple concatenation has the same
positive and negative marginal distributions, so a linear classifier cannot
learn much. The interaction terms give the model a way to learn compatibility.

The PyTorch version should replace this with:

```text
trajectory encoder + parameter encoder + binary classifier head
```

## 8. Relationship To The PyTorch Pipeline

The earlier PyTorch files remain the intended full training path:

- `src/data/lorenz_dataset.py`
- `src/models/lorenz_models.py`
- `scripts/train_lorenz.py`

They support:

```text
--task regime
--task params
--task ratio
```

The most important task for the final project is:

```text
--task ratio
```

because it implements the matched/mismatched classifier used for
simulation-based inference.

The NumPy demo exists because the current `.venv` is broken and the bundled
runtime does not include PyTorch. Once the environment is fixed, the PyTorch
ratio estimator should become the main model.

## 9. Environment Issue

The current `.venv` points to a missing Python installation:

```text
C:\Users\liuyo\AppData\Local\Programs\Python\Python313
```

Recommended fix:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 or 3.12 is safer than 3.13 for PyTorch compatibility.

## 10. Next Step: Football CRW Model

After the Lorenz demo, switch the simulator to correlated random walk (CRW).

Suggested football SDE state:

```text
state = (x, y, v, heading)
```

Suggested parameters:

```text
theta = (kappa, v0, sigma_heading, sigma_v)
```

Interpretation:

- `kappa`: how quickly the player corrects heading
- `v0`: preferred running speed
- `sigma_heading`: directional randomness
- `sigma_v`: speed randomness

The same SBI workflow applies:

```text
simulate CRW tracks
-> build matched/mismatched pairs
-> train classifier
-> score candidate parameters for observed track
-> simulate future football paths
```

## 11. Presentation Structure

Use this order:

1. Problem: football movement is structured but stochastic.
2. SDE idea: drift models intent, diffusion models randomness.
3. Challenge: one parameter set can produce many possible tracks.
4. Bayesian target: infer `p(theta | observed track)`.
5. Method: simulation-based inference with binary classification.
6. Demo: Lorenz SDE validates the pipeline.
7. Results: classifier reaches about 74 percent validation accuracy.
8. Visualization: show `posterior_scores.svg` and `future_paths.svg`.
9. Next step: replace Lorenz with football CRW.
10. Final target: probability distribution over future player locations.

## 12. Current Status

Completed:

- runnable Lorenz SBI demo
- matched/mismatched classifier
- posterior-like parameter scoring
- future-path sampling from likely parameters
- CSV/SVG outputs
- project workflow documentation

Still to do:

- repair Python/PyTorch environment
- train the neural ratio estimator
- add CRW dataset generation
- apply the same posterior-scoring workflow to football trajectories
- optionally add model selection across Brownian, OU, CRW, and social-force SDEs

