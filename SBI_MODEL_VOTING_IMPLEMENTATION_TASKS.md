# SBI Model Voting Implementation Tasks

## 1. Final Project Direction

The agreed final scope is the movement and probabilistic prediction of the
**football ball only**. Player trajectories, player roles, teammates, and
opponents are not part of this implementation. The project should not rely on
one fixed motion model such as OU. Observed ball trajectories are often
piecewise and event-driven:

```text
straight segment -> sharp turn -> straight segment -> stop/deflection
```

The final inference target should therefore be:

```text
p(model, theta | observed_track)
```

and the final prediction should be:

```text
p(future_position | observed_track)
```

not one deterministic future trajectory.

The required method remains:

```text
simulation-based inference
contrastive / ratio learning
MCMC posterior sampling
posterior predictive distribution
```

Current implemented run order:

```powershell
python scripts\model_voting_pipeline\extract_football_windows.py `
  --home data\Sample_Game_1\Sample_Game_1_RawTrackingData_Home_Team.csv `
  --away data\Sample_Game_1\Sample_Game_1_RawTrackingData_Away_Team.csv `
  --team home `
  --entity Ball `
  --period 1 `
  --start-time 37.2 `
  --T 5.0 `
  --prefix-T 2.0 `
  --dt 0.04 `
  --out data\real_football_windows.npz

python scripts\model_voting_pipeline\generate_model_voting_data.py `
  --real-windows data\real_football_windows.npz `
  --n-per-model 1000 `
  --T 5.0 `
  --dt 0.04 `
  --out-dir data\model_voting_dataset

python scripts\tools\plot_model_voting_dataset.py `
  --dataset data\model_voting_dataset\dataset.npz `
  --out-dir outputs\model_voting_dataset_viz

python scripts\model_voting_pipeline\train_model_voting_ratio.py `
  --data-dir data\model_voting_dataset `
  --epochs 100 `
  --batch-size 128 `
  --out-dir checkpoints

python scripts\model_voting_pipeline\recover_model_voting_posterior.py `
  --real-windows data\real_football_windows.npz `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --window-index 0 `
  --mcmc-steps 3000 `
  --burn-in 800 `
  --n-evidence-samples 4096 `
  --out-dir outputs\model_voting_posterior

python scripts\model_voting_pipeline\evaluate_model_voting.py `
  --posterior outputs\model_voting_posterior\posterior_chains.npz `
  --n-paths 300 `
  --out-dir outputs\model_voting_evaluation

python scripts\model_voting_pipeline\evaluate_synthetic_model_recovery.py `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --dataset data\model_voting_dataset\dataset.npz `
  --n-cases 80 `
  --n-evidence-samples 512 `
  --out-dir outputs\synthetic_model_recovery

python scripts\tools\football_model_voting_clip.py `
  --game data\Sample_Game_1 `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --period 1 `
  --start-time 37.2 `
  --duration 5.0 `
  --trail-seconds 2.0 `
  --score-window-seconds 2.0 `
  --out outputs\football_model_voting_clip.gif
```

In the extraction command, `--T` is the window duration and `--start-time`
chooses where that observed window begins. The example above extracts a
5-second window starting closest to 37.2 seconds in period 1. `--prefix-T 2.0`
saves the first 2 seconds as `prefix_tracks` and the remaining 3 seconds as
`suffix_tracks`. Posterior recovery uses the prefix only, and evaluation scores
posterior predictive samples against the held-out suffix. Use `--start-frame`
instead for an exact tracking frame. If neither option is provided, the
extractor falls back to scan mode and creates many windows using `--stride`.

When `data\real_football_windows.npz` contains `prefix_tracks`, the synthetic
model-voting generator automatically uses that prefix length for training. With
the command above, the classifier therefore trains on 2-second observed
prefixes, while the 3-second suffix is kept only for posterior predictive
evaluation.

Optional segment diagnostic plot:

```powershell
python scripts\tools\plot_real_window_segments.py `
  --real-windows data\real_football_windows.npz `
  --window-index 0 `
  --out outputs\real_window_segments.png
```

Optional real-data visual inspection:

```powershell
python scripts\tools\football_window_clip.py `
  --game data\Sample_Game_1 `
  --period 1 `
  --start-time 37.2 `
  --duration 5.0 `
  --frame-step 2 `
  --fps 12 `
  --trail-seconds 2.0 `
  --out outputs\football_window_clip.gif
```

This clip tool is not part of ratio training or MCMC inference. Its role is to
inspect whether the chosen observed time window contains the kind of ball motion
we want the model-voting SBI demo to explain. `--trail-seconds 2.0` keeps only
the most recent two seconds of ball trajectory visible as a sliding trail.

The model-voting clip uses the trained ratio classifier and a sliding recent
trajectory window. For each scored frame, it samples candidate parameters for
each SDE family, scores them with the learned classifier, and displays a live
soft vote over Brownian, constant-velocity, OU-target, and piecewise-velocity
models beside the pitch.

## 2. Candidate Models For Model Voting

Implement several candidate simulators. Each model must expose:

```text
simulate(theta, condition, rng) -> track
sample_prior(rng, n) -> theta samples
normalize_theta(theta) -> normalized theta
```

Recommended model set:

1. Brownian motion
   - baseline random walk
   - parameters: `noise_scale`

2. Constant velocity SDE
   - single straight-line motion with noise
   - parameters: `vx, vy, noise_scale`

3. OU-to-target
   - single-model OU-to-target baseline
   - parameters: `k, noise_scale`

4. Piecewise constant velocity SDE
   - best fit for observed ball tracks with sharp turns
   - parameters depend on segment count
   - for a fixed 3-segment demo:

```text
theta = (vx1, vy1, vx2, vy2, vx3, vy3, noise_scale)
```

If change points are inferred:

```text
theta = (tau1, tau2, vx1, vy1, vx2, vy2, vx3, vy3, noise_scale)
```

For the first implementation, use detected change points from preprocessing
instead of sampling `tau`.

## 3. Track Preprocessing

From raw position-only data:

```text
p_t = (x_t, y_t)
```

derive:

```text
velocity      v_t = (p_t - p_{t-1}) / dt
speed         ||v_t||
heading       atan2(v_y, v_x)
acceleration  (v_t - v_{t-1}) / dt
turn angle    angle(v_t, v_{t-1})
```

Tasks:

- [x] Add `src/data/trajectory_features.py`.
- [x] Implement finite-difference velocity.
- [x] Implement speed and acceleration.
- [x] Implement heading and turn-angle features.
- [x] Smooth positions lightly before differencing if tracking noise is large.
- [x] Add diagnostics for extreme jumps and missing values.

## 4. Piecewise Segmentation

The purpose is to detect transient structure before simulation/inference.

Simple first algorithm:

```text
1. compute velocity vectors
2. compute angle change between consecutive velocities
3. mark change point if angle change > threshold
4. also mark if speed jumps/drops strongly
5. merge tiny segments
6. keep top K-1 change points for K segments
```

Tasks:

- [x] Add `src/data/segmentation.py`.
- [x] Implement angle-threshold segmentation.
- [x] Implement minimum segment length.
- [x] Implement fixed-K segmentation fallback.
- [x] Plot observed track with detected change points.
- [x] Save segment metadata into real-window `.npz`.

## 5. Synthetic Data Generation

Current OU generator:

```text
scripts/OU_workflow/generate_football_ou_data.py
```

New model-voting generator should create a mixed dataset:

```text
scripts/model_voting_pipeline/generate_model_voting_data.py
```

Dataset keys:

```text
tracks       (N, steps, 2)
model_id     (N,)
parameters   object or padded array
condition    model-specific condition
y0           (N, 2)
target       (N, 2), if applicable
segments     optional change-point metadata
```

Tasks:

- [x] Implement Brownian simulator.
- [x] Implement constant velocity simulator.
- [x] Reuse OU simulator.
- [x] Implement piecewise velocity simulator.
- [x] Add integer `model_id` for model embedding.
- [x] Store parameter masks for models with different parameter dimensions.
- [x] Generate balanced samples per model.
- [x] Plot dataset diversity per model.

Implemented files:

```text
src/sde/model_voting.py
scripts/model_voting_pipeline/generate_model_voting_data.py
scripts/tools/plot_model_voting_dataset.py
```

## 6. Contrastive Ratio Classifier

The classifier should estimate:

```text
log r_phi(track, model, theta, condition)
```

Training labels:

```text
matched model/theta/track      -> 1
mismatched model/theta/track   -> 0
```

Architecture:

```text
track encoder
model embedding / one-hot
parameter encoder
condition encoder
binary classifier head
```

Tasks:

- [x] Add `src/models/model_voting_ratio.py`.
- [x] Support variable parameter dimensions via padding + mask.
- [x] Add model one-hot or model embedding.
- [x] Keep y0/target/segment conditions attached to the track.
- [x] Train with balanced matched/mismatched pairs.
- [x] Track validation accuracy and log-ratio gap per model.

Implemented files:

```text
src/data/model_voting_dataset.py
src/models/model_voting_ratio.py
scripts/model_voting_pipeline/train_model_voting_ratio.py
```

## 7. MCMC Posterior Inference

For a real observed window, run posterior inference:

```text
p(model, theta | track)
```

Practical implementation:

```text
for each model:
    sample theta from its prior
    estimate marginal evidence ratio with log-mean-exp of classifier logits
    run MCMC over theta for the parameter posterior
normalize evidence ratios into model probabilities using equal model priors
```

Tasks:

- [x] Add `scripts/model_voting_pipeline/recover_model_voting_posterior.py`.
- [x] Implement model-specific priors.
- [x] Implement random-walk Metropolis-Hastings per model.
- [x] Store chains per model.
- [x] Compute acceptance rate per model.
- [x] Compute model vote weights using prior Monte Carlo evidence integration.
- [x] Save posterior samples and model scores.

Implemented output:

```text
outputs/model_voting_posterior/summary.json
outputs/model_voting_posterior/posterior_chains.npz
```

## 8. Posterior Predictive Distribution

The output must be distributional.

For each posterior sample:

```text
sample model/theta
simulate future path
collect endpoint and full path
```

Render:

- posterior predictive path cloud
- density heatmap of future positions
- model vote bar chart
- parameter histograms for the winning model
- optional per-time-step density

Tasks:

- [x] Add `scripts/model_voting_pipeline/evaluate_model_voting.py`.
- [x] Plot observed prefix and true future suffix separately.
- [x] Plot sampled future paths.
- [x] Plot future endpoint density.
- [x] Plot model posterior/vote bar chart.
- [x] Plot parameter histograms for the winning model.
- [x] Report single-window radial predictive-region coverage if a suffix is held out.
- [ ] Aggregate coverage over many independent windows to assess calibration.

Implemented output:

```text
outputs/model_voting_evaluation/posterior_predictive_paths.png
outputs/model_voting_evaluation/endpoint_density.png
outputs/model_voting_evaluation/model_vote_weights.png
outputs/model_voting_evaluation/winning_model_parameter_histograms.png
outputs/model_voting_evaluation/summary.json
```

## 9. Correct Prediction Protocol

Avoid using the full observed window as both condition and target.

Use:

```text
observed prefix -> infer posterior -> predict future suffix
```

Example:

```text
window length = 5 seconds
observed prefix = first 2 seconds
future target = next 3 seconds
```

Tasks:

- [x] Update real-window extraction to store prefix/suffix split.
- [x] Condition inference on prefix only.
- [x] Evaluate predictive distribution against suffix.
- [x] Do not leak endpoint unless the task is explicitly reconstruction.
The recommended implementation is integrated into:

```text
scripts/model_voting_pipeline/extract_football_windows.py
scripts/model_voting_pipeline/generate_model_voting_data.py
scripts/model_voting_pipeline/recover_model_voting_posterior.py
scripts/model_voting_pipeline/evaluate_model_voting.py
```

The earlier standalone baseline remains available for comparison:

```text
scripts/extract_prefix_suffix_windows.py
scripts/recover_model_voting_prefix_posterior.py
scripts/evaluate_prefix_suffix_prediction.py
```

Implemented data contract:

```text
data/real_football_prefix_suffix_windows.npz

prefix_tracks          # first 2 seconds
suffix_tracks          # held-out next 3 seconds
full_tracks
y0
prefix_end
suffix_end
target_for_evaluation
dt
prefix_steps
suffix_steps
```

Historical baseline note:

```text
The current trained ratio model expects a target-like condition. For strict
future prediction, the real suffix endpoint must not be used. The prefix
posterior script therefore supports prefix-only target heuristics:

average
recent
prefix_end
```

Historical result from the full-window-trained checkpoint:

```text
The protocol runs end-to-end, but the window-0 baseline is not yet a strong
future predictor:

suffix endpoint median error: 22.69 m
path RMSE median: 14.94 m
coverage rate: 0.04
```

Next validation target:

```text
Regenerate the synthetic dataset from prefix-enabled real windows, retrain the
ratio classifier, and evaluate on an independent held-out dataset. The old
22.69 m result does not measure the new integrated workflow.
```

- [x] Keep the OU target condition consistent between inference and prediction.
- [x] Do not reuse historical change-point timestamps as future events.
- [ ] Learn or sample future change-point times instead of assuming no future turn.

Current conservative future assumptions:

```text
OU target               = last observed ball position
constant velocity       = continue inferred velocity
piecewise velocity      = continue latest inferred segment
future direction change = none within the short forecast horizon
```

## 10. Validation Required Before Final Claims

The end-to-end demo is implemented, but a scientifically defensible result
requires tests on data where the generating truth is known and tests across
more than one real window.

Tasks:

- [x] Generate fresh synthetic test cases not copied from classifier training rows.
- [x] Report a true-model versus selected-model confusion matrix.
- [x] Report top-1 model recovery accuracy and mean model log score.
- [ ] Run MCMC on synthetic examples with known theta.
- [ ] Report parameter bias, interval width, and 50/80/90% posterior coverage.
- [ ] Extract evaluation windows from both Sample_Game_1 and Sample_Game_2.
- [ ] Report aggregate ADE, FDE, and predictive-region coverage.
- [ ] Compare against stationary, last-velocity, and empirical-noise baselines.
- [ ] Repeat evidence estimation with multiple random seeds to measure Monte
      Carlo variability.

## 11. Optional Demonstration Checklist

Use the current OU failure as motivation:

```text
The single-model OU baseline validated the SBI pipeline but failed on real ball tracks
because ball motion is piecewise and event-driven.
We therefore extend to model voting: several SDE hypotheses compete under a
contrastive ratio estimator, and MCMC samples the posterior over model and
parameters.
The final output is a distribution over future positions, not one track.
```

Current model-voting demonstration:

- [x] Show raw observed ball window.
- [x] Show detected piecewise segments.
- [ ] Regenerate model vote distribution with corrected prior-integrated evidence.
- [ ] Regenerate winning-model parameter posterior with corrected inference.
- [ ] Regenerate posterior predictive future density with corrected forecasting assumptions.

Presentation artifact commands:

```powershell
# Raw observed ball movement clip.
python scripts\tools\football_window_clip.py `
  --game data\Sample_Game_1 `
  --period 1 `
  --start-time 37.2 `
  --duration 5.0 `
  --trail-seconds 2.0 `
  --out outputs\football_window_clip.gif

# Detected piecewise segment figure.
python scripts\tools\plot_real_window_segments.py `
  --real-windows data\real_football_windows.npz `
  --window-index 0 `
  --out outputs\real_window_segments.png

# Model-voting posterior figures:
# model_vote_weights.png, winning_model_parameter_histograms.png,
# endpoint_density.png, posterior_predictive_paths.png.
python scripts\model_voting_pipeline\evaluate_model_voting.py `
  --posterior outputs\model_voting_posterior\posterior_chains.npz `
  --n-paths 300 `
  --out-dir outputs\model_voting_evaluation
```
