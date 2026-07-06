# Football Tracks To SBI Model Workflow

This document explains how real football tracking data should be fitted into
the current encoder/contrastive/SBI model.

The short answer:

```text
raw tracking CSV
-> fixed real windows: track, y0, target
-> synthetic OU training data conditioned on matching y0/target distribution
-> train C_phi(track, theta, y0, target)
-> score candidate theta for one real window
-> simulate posterior predictive paths
```

## 1. Why The Raw Trajectory Cannot Go Directly Into The Current Model

The previous Lorenz pipeline assumed:

```text
track:  (steps, 3)
theta:  (sigma, rho, beta, noise_scale)
```

Football Phase A OU needs:

```text
track:     (steps, 2)
theta:     (k, noise_scale)
y0:        (x_start, y_start)
target:    (x_end, y_end)
condition: concat(y0_norm, target_norm)
```

The same `(k, noise_scale)` can generate very different absolute paths from
different starting points and targets. Therefore the classifier must receive
`y0` and `target`. Without this conditioning, it cannot judge whether a
parameter-track pair is physically compatible.

## 2. New Files

Core simulator:

```text
src/sde/football_ou.py
```

Dataset adapter:

```text
src/data/football_dataset.py
```

Real CSV window extractor:

```text
scripts/extract_football_windows.py
```

Reusable tracking-data package:

```text
src/data/football_tracking.py
```

Reusable football visualization helpers:

```text
src/utils/football_viz.py
```

Visualization CLI wrapper:

```text
scripts/football_tracking_viz.py
```

Synthetic OU dataset generator:

```text
scripts/generate_football_ou_data.py
```

Conditioned ratio classifier training:

```text
scripts/train_football_ou_ratio.py
```

Real-window posterior scoring and output rendering:

```text
scripts/score_real_football_window.py
```

This file now covers the football analogue of both Lorenz:

```text
recover_posterior.py
evaluate.py
```

It can run random-walk Metropolis-Hastings and it renders both predictive
tracks and parameter distribution histograms.

## 3. Step-By-Step Workflow

### Step 1: Extract Real Windows

Use the reusable tracking parser through:

```powershell
python scripts\extract_football_windows.py `
  --home data\Sample_Game_1\Sample_Game_1_RawTrackingData_Home_Team.csv `
  --away data\Sample_Game_1\Sample_Game_1_RawTrackingData_Away_Team.csv `
  --team home `
  --entity Ball `
  --T 5.0 `
  --dt 0.04 `
  --stride 25 `
  --out data\real_football_windows.npz
```

Here `--T` means duration of each extracted window. If you want to choose one
specific window instead of scanning with `--stride`, provide either
`--start-time` or `--start-frame`:

```powershell
python scripts\extract_football_windows.py `
  --home data\Sample_Game_1\Sample_Game_1_RawTrackingData_Home_Team.csv `
  --away data\Sample_Game_1\Sample_Game_1_RawTrackingData_Away_Team.csv `
  --team home `
  --entity Ball `
  --period 1 `
  --start-time 37.2 `
  --T 5.0 `
  --dt 0.04 `
  --out data\real_football_windows.npz
```

This extracts the 5-second ball trajectory starting at the row closest to
37.2 seconds in period 1. To select by exact frame instead:

```powershell
python scripts\extract_football_windows.py `
  --home data\Sample_Game_1\Sample_Game_1_RawTrackingData_Home_Team.csv `
  --away data\Sample_Game_1\Sample_Game_1_RawTrackingData_Away_Team.csv `
  --team home `
  --entity Ball `
  --start-frame 12400 `
  --T 5.0 `
  --dt 0.04 `
  --out data\real_football_windows.npz
```

Output:

```text
tracks: (N, steps, 2)
y0:     (N, 2)
target: (N, 2)
meta:   window metadata
```

Start with `Ball`. Later use `Player7`, `Player10`, etc.

`scripts/football_tracking_viz.py` is not the training data adapter anymore.
Its role is visual inspection:

```powershell
python scripts\football_tracking_viz.py `
  --home data\Sample_Game_1\Sample_Game_1_RawTrackingData_Home_Team.csv `
  --away data\Sample_Game_1\Sample_Game_1_RawTrackingData_Away_Team.csv `
  --frame 1 `
  --t 25 `
  --out outputs\football_visualisation.png
```

The reusable parsing functions live in `src/data/football_tracking.py`, so both
the extractor and the visualizer use the same CSV interpretation.

For a moving clip of one selected time window, use:

```powershell
python scripts\football_window_clip.py `
  --game data\Sample_Game_1 `
  --period 1 `
  --start-time 37.2 `
  --duration 5.0 `
  --frame-step 2 `
  --fps 12 `
  --out outputs\football_window_clip.gif
```

Use `.gif` for the most portable output. Use `.mp4` only if ffmpeg is
available in your Python environment.

### Step 2: Generate Synthetic OU Training Data

Train on synthetic tracks, not directly on real tracks, because the real data
has no ground-truth SDE parameters.

```powershell
python scripts\generate_football_ou_data.py `
  --real-windows data\real_football_windows.npz `
  --n-samples 200 `
  --n-tracks 20 `
  --T 5.0 `
  --dt 0.04 `
  --out-dir data\football_ou_dataset
```

The generator bootstraps real `(y0, target)` pairs if
`data\real_football_windows.npz` exists. This makes synthetic conditions match
real football windows.

Output:

```text
data/football_ou_dataset/dataset.npz
```

with:

```text
tracks:     (N, steps, 2)
parameters: (N, 2)  # k, noise_scale
y0:         (N, 2)
target:     (N, 2)
group_ids:  (N,)
```

### Step 3: Train The Conditioned Ratio Classifier

```powershell
python scripts\train_football_ou_ratio.py `
  --data-dir data\football_ou_dataset `
  --epochs 80 `
  --batch-size 128 `
  --out-dir checkpoints
```

The model is:

```text
track encoder:      TrajectoryEncoder(in_channels=2)
theta encoder:      MLP(k, noise_scale)
condition encoder:  MLP(y0, target)
classifier head:    binary matched/mismatched logit
```

Positive pair:

```text
(track_i, theta_i, y0_i, target_i) -> 1
```

Negative pair:

```text
(track_i, theta_j, y0_i, target_i) -> 0
```

Note that `y0_i` and `target_i` stay with the track. Only `theta` is shuffled.

Output:

```text
checkpoints/football_ou_ratio_best.pt
checkpoints/football_ou_ratio_history.csv
```

### Step 4: Run Real-Window Inference

```powershell
python scripts\score_real_football_window.py `
  --real-windows data\real_football_windows.npz `
  --checkpoint checkpoints\football_ou_ratio_best.pt `
  --window-index 0 `
  --n-candidates 5000 `
  --sampler mcmc `
  --mcmc-steps 6000 `
  --burn-in 1500 `
  --out-dir outputs\football_ou_real
```

This first samples candidate `(k, noise_scale)` values for initialization and
diagnostics, then runs random-walk Metropolis-Hastings in physical parameter
space.

The MCMC target is:

```text
log p(theta | track)
  ~= log prior(theta)
   + classifier_logit(track, theta, y0, target)
   + likelihood_weight * OU_transition_loglik(track | theta, y0, target)
```

The explicit OU transition-likelihood term is a calibration term. It helps
avoid the failure mode where the neural classifier assigns high scores to very
large `noise_scale` values that are broadly plausible but render poor
predictive paths.

Outputs:

```text
outputs/football_ou_real/top_candidates.csv
outputs/football_ou_real/posterior_predictive.svg
outputs/football_ou_real/parameter_distributions.svg
outputs/football_ou_real/posterior_predictive.npz
outputs/football_ou_real/summary.json
```

Quick candidate-grid mode, without MCMC:

```powershell
python scripts\score_real_football_window.py `
  --sampler candidates `
  --n-candidates 5000 `
  --out-dir outputs\football_ou_real_candidates
```

## 4. Desired Output

For the project demo, the desired output is not one predicted path. It should
be:

1. The observed real trajectory.
2. A ranked posterior-like list of candidate parameters.
3. A fan of possible future/simulated trajectories.
4. A visualization showing whether the posterior predictive paths cover the
   observed trajectory.
5. Parameter distribution histograms for:
   - `k`
   - `noise_scale`

The key figure is:

```text
outputs/football_ou_real/posterior_predictive.svg
outputs/football_ou_real/parameter_distributions.svg
```

Blue line:

```text
real observed window
```

Green lines:

```text
posterior predictive OU paths
```

The histogram figure follows the Lorenz `evaluate.py` idea:

```text
grey = prior distribution
color = posterior samples from MCMC
```

For real football data there is no ground-truth parameter line, because the
real track does not come with true `(k, noise_scale)`.

## 5. How This Fits The Existing Modules

Existing module:

```text
src/models/encoder.py
```

Already supports `in_channels`, so football uses:

```python
TrajectoryEncoder(in_channels=2)
```

Existing concept:

```text
contrastive / ratio learning
```

Now implemented for football in:

```text
scripts/train_football_ou_ratio.py
```

Dataset contract:

```python
batch["track"]      # (B, 2, steps)
batch["params"]     # (B, 2)
batch["condition"]  # (B, 4), concat(y0_norm, target_norm)
```

Forward call:

```python
logit = model(track, params, condition)
```

Tracking-data package role:

```text
src/data/football_tracking.py
```

owns:

```text
load_tracking()
denormalize()
entity_xy()
extract_fixed_windows()
```

Visualization package role:

```text
src/utils/football_viz.py
```

owns:

```text
pitch_background()
visualize_tracking_frame()
```

## 6. Current Limitations

- This is Phase A: position-only OU.
- The target is currently the end point of the window.
- Real tracks do not provide true SDE parameters, so evaluation is visual and
  posterior-predictive, not parameter-MAE.
- Full player behavior needs Phase B or social force later.

## 7. Next Improvements

1. Add velocity/momentum OU:

```text
state = (x, y, vx, vy)
theta = (k, damping, noise_scale)
condition = (y0, target, v0)
```

2. Use ball position as target for player movement instead of window end point.
3. Add model selection across Brownian, OU, CRW, and social force.
4. Replace candidate-grid scoring with MCMC after the conditioned classifier is
   trained.
