# Football Ball Model-Voting SBI Project Guide

## 1. Purpose And Scope

This project predicts the future movement of the football **ball**, not the
players. It does not try to output one certain future line. Instead, it asks:

```text
Given the last two seconds of observed ball positions:

1. Which candidate stochastic motion model could explain the movement?
2. Which parameter values are plausible for that model?
3. What distribution of ball paths could occur during the next three seconds?
```

The inference target is:

```text
p(model, theta | observed ball track)
```

where:

- `model` is one of four candidate SDE families;
- `theta` is the parameter vector belonging to that model;
- the final output is a distribution of future paths and positions.

Player roles, teammates, opponents, and tactical context are outside the
current project scope.

## 2. The Whole Picture Without Code

The real tracking data contains ball positions but no correct SDE label and no
correct SDE parameters. Direct supervised training is therefore impossible.

Simulation-Based Inference (SBI) solves this by creating its own labelled
training examples:

```text
choose model and theta
        |
        v
simulate a synthetic ball trajectory
        |
        v
train a classifier to recognize matching and mismatching
(trajectory, model, theta) combinations
        |
        v
apply the learned ratio to a real observed ball trajectory
        |
        v
use MCMC to sample plausible theta for each model
        |
        v
integrate scores over the prior to obtain approximate model weights
        |
        v
simulate many future paths from the model/theta posterior
```

The classifier is not directly asked to predict the future. It learns a score
that behaves like a likelihood ratio. MCMC and posterior simulation turn that
score into parameter distributions and future-path distributions.

## 3. Candidate Motion Models

All models represent the two-dimensional pitch position:

```text
X_t = (x_t, y_t)
```

### 3.1 Brownian Motion

```text
dX_t = sigma dW_t
theta = (sigma)
```

Interpretation: no preferred direction; only stochastic movement. It is a
random-motion baseline.

### 3.2 Constant Velocity

```text
dX_t = v dt + sigma dW_t
theta = (vx, vy, sigma)
```

Interpretation: continue moving in one direction, with stochastic deviation.

### 3.3 OU To Target

```text
dX_t = k(target - X_t) dt + sigma dW_t
theta = (k, sigma)
```

Interpretation: the ball is pulled toward an equilibrium location. During
future prediction, the last observed position is used as the no-leak target.
This makes OU a stop/settle hypothesis.

### 3.4 Piecewise Constant Velocity

```text
dX_t = v_j dt + sigma dW_t
theta = (vx1, vy1, vx2, vy2, vx3, vy3, sigma)
```

Interpretation: the observed trajectory can contain three approximately
straight segments separated by two direction changes. For future prediction,
the latest inferred segment is continued. Historical change-point times are
not reused as future events.

## 4. Repository Layout

```text
deep-learning-lab/
|-- data/
|   |-- Sample_Game_1/              raw Metrica match data
|   |-- Sample_Game_2/              raw Metrica match data
|   |-- real_football_windows.npz   extracted real prefix/suffix window
|   `-- model_voting_dataset/       synthetic SBI training data
|
|-- checkpoints/
|   |-- model_voting_ratio_best.pt
|   `-- model_voting_ratio_history.csv
|
|-- scripts/
|   |-- model_voting_pipeline/       required end-to-end pipeline stages
|   |-- tools/                       optional visualization/inspection tools
|   |-- OU_workflow/                 historical single-model baseline
|   `-- Lorenz_workflow/             historical Lorenz demonstration
|
|-- src/
|   |-- data/                        active tracking and dataset modules
|   |-- models/                      active encoder and ratio classifier
|   |-- sde/                         active candidate simulators
|   |-- utils/                       active football visualization helpers
|   `-- legacy/                      modules excluded from active pipeline
|
|-- outputs/                          generated figures, metrics, and clips
|-- README.md                         short run order
|-- SBI_MODEL_VOTING_IMPLEMENTATION_TASKS.md
|-- MODEL_VOTING_PROJECT_GUIDE.md     this detailed guide
`-- OU_BASELINE_WORKFLOW.md           historical OU documentation
```

The active model-voting workflow uses `src/sde/football_ou.py` for the OU
candidate simulator. Therefore that source module remains active even though
the old standalone OU scripts are archived under `scripts/OU_workflow`.

### Script Groups

Required model-voting pipeline:

| Order | Script | Role |
|---:|---|---|
| 1 | `model_voting_pipeline/extract_football_windows.py` | Create real prefix/suffix windows. |
| 2 | `model_voting_pipeline/generate_model_voting_data.py` | Simulate labelled training data. |
| 3 | `model_voting_pipeline/train_model_voting_ratio.py` | Train the ratio classifier. |
| 4 | `model_voting_pipeline/evaluate_synthetic_model_recovery.py` | Validate model selection on fresh simulations. |
| 5 | `model_voting_pipeline/recover_model_voting_posterior.py` | Run evidence estimation and MCMC on a real prefix. |
| 6 | `model_voting_pipeline/evaluate_model_voting.py` | Evaluate future paths, errors, and coverage. |

Complementary tools, not required for training or MCMC:

| Tool | Role |
|---|---|
| `tools/plot_model_voting_dataset.py` | Check synthetic diversity and parameter priors. |
| `tools/plot_real_window_segments.py` | Inspect detected historical change points. |
| `tools/football_tracking_viz.py` | Render one raw tracking frame. |
| `tools/football_window_clip.py` | Render a raw ball/player time-window clip. |
| `tools/football_model_voting_clip.py` | Render a sliding classifier-score animation. |

Historical groups:

- `OU_workflow`: archived single-model OU baseline;
- `Lorenz_workflow`: archived educational Lorenz demonstration.

## 5. Important Data Shapes

With `dt = 0.04`, tracking contains 25 samples per second.

```text
full real window:    5 seconds = 125 positions
observed prefix:     2 seconds = 50 positions
held-out suffix:     3 seconds = 75 positions
one position:                    (x, y)
```

Important arrays in `real_football_windows.npz`:

```text
tracks          (N, 125, 2) complete extracted windows
prefix_tracks   (N,  50, 2) input visible to inference
suffix_tracks   (N,  75, 2) hidden future used only for evaluation
y0              (N, 2)      start positions
target          (N, 2)      window endpoints
change_points   (N, 2)      detected historical segment boundaries
diagnostics     (N,)        missing-data and jump information
meta            (N,)        frame, period, and time metadata
```

Important arrays in `model_voting_dataset/dataset.npz`:

```text
tracks              (N, 50, 2) synthetic observed-prefix trajectories
model_id            (N,)       integer model labels
parameters          (N, 7)     padded physical theta vectors
parameters_norm     (N, 7)     theta normalized for the network
parameter_mask      (N, 7)     active dimensions for each model
conditions          (N, 8)     start, target, and change-point information
y0, target                      simulation conditions
change_points                   simulator segment boundaries
```

## 6. Installation And Starting Point

Open PowerShell in the repository root:

```powershell
cd "C:\Users\liuyo\Desktop\Studium\SS26\DL Lab - VScode\deep-learning-lab"
```

Create and activate an environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Confirm that Python can import the major packages:

```powershell
python -c "import torch, numpy, pandas, matplotlib; print('environment ready')"
```

All workflow commands below assume the current directory is the repository
root. Do not run them from inside `scripts`.

## 7. Complete Step-By-Step Run

### Step 1: Extract A Real Five-Second Window

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
```

This creates a 2-second inference input and a separate 3-second evaluation
target. The suffix is never passed to MCMC.

Inspect the segmentation:

```powershell
python scripts\tools\plot_real_window_segments.py `
  --real-windows data\real_football_windows.npz `
  --window-index 0 `
  --out outputs\real_window_segments.png
```

### Step 2: Generate Synthetic SBI Training Data

```powershell
python scripts\model_voting_pipeline\generate_model_voting_data.py `
  --real-windows data\real_football_windows.npz `
  --n-per-model 1000 `
  --T 5.0 `
  --dt 0.04 `
  --out-dir data\model_voting_dataset
```

Because the real file contains `prefix_tracks`, the generator automatically
uses 50-step synthetic tracks, matching the 2-second observed prefix.

Inspect the synthetic model prior, parameter priors, condition priors, and
prior-predictive trajectory diversity before training:

```powershell
python scripts\tools\plot_model_voting_dataset.py `
  --dataset data\model_voting_dataset\dataset.npz `
  --out-dir outputs\model_voting_dataset_viz
```

### Step 3: Train The Neural Ratio Classifier

```powershell
python scripts\model_voting_pipeline\train_model_voting_ratio.py `
  --data-dir data\model_voting_dataset `
  --epochs 100 `
  --batch-size 128 `
  --out-dir checkpoints
```

Outputs:

```text
checkpoints/model_voting_ratio_best.pt
checkpoints/model_voting_ratio_history.csv
```

Validation accuracy answers whether matched and mismatched tuples can be
separated. It does not by itself prove correct model selection or calibrated
future prediction.

### Step 4: Validate Model Selection On Fresh Synthetic Tracks

```powershell
python scripts\model_voting_pipeline\evaluate_synthetic_model_recovery.py `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --dataset data\model_voting_dataset\dataset.npz `
  --n-cases 80 `
  --n-evidence-samples 512 `
  --out-dir outputs\synthetic_model_recovery
```

This creates new trajectories with known model labels and reports a confusion
matrix. The current 80-case run achieved 87.5% top-1 recovery. This supports
synthetic model selection but does not prove real-data calibration.

### Step 5: Recover Model And Parameter Posteriors For The Real Prefix

```powershell
python scripts\model_voting_pipeline\recover_model_voting_posterior.py `
  --real-windows data\real_football_windows.npz `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --window-index 0 `
  --n-init-candidates 2048 `
  --n-evidence-samples 4096 `
  --mcmc-steps 3000 `
  --burn-in 800 `
  --out-dir outputs\model_voting_posterior
```

For every model family, this command:

1. samples theta from the prior;
2. scores prior samples with the classifier;
3. estimates a marginal evidence ratio with `logmeanexp`;
4. initializes and runs random-walk Metropolis-Hastings;
5. saves posterior theta chains and acceptance rates;
6. normalizes evidence ratios into approximate model weights.

Outputs:

```text
outputs/model_voting_posterior/summary.json
outputs/model_voting_posterior/posterior_chains.npz
```

### Step 6: Evaluate The Held-Out Three-Second Future

```powershell
python scripts\model_voting_pipeline\evaluate_model_voting.py `
  --posterior outputs\model_voting_posterior\posterior_chains.npz `
  --n-paths 300 `
  --out-dir outputs\model_voting_evaluation
```

Outputs:

```text
posterior_predictive_paths.png
endpoint_density.png
model_vote_weights.png
winning_model_parameter_histograms.png
posterior_predictive_samples.npz
summary.json
```

The summary contains:

- ADE: average displacement error over future timesteps;
- FDE: final displacement error at the last future timestep;
- best sampled ADE/FDE;
- per-sample path and endpoint error distributions;
- 50%, 80%, and 90% radial predictive-region coverage.

Coverage from one suffix is only a diagnostic. Calibration requires many
independent windows.

### Step 7: Optional Animation

Raw tracking clip:

```powershell
python scripts\tools\football_window_clip.py `
  --game data\Sample_Game_1 `
  --period 1 `
  --start-time 37.2 `
  --duration 5.0 `
  --trail-seconds 2.0 `
  --out outputs\football_window_clip.gif
```

Classifier-score animation:

```powershell
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

The live animation displays local classifier scores over a sliding trajectory
window. These are visualization votes, not full MCMC evidence estimates.

## 8. Script Reference

The first six entries below are the required main pipeline. Visualization
scripts are complementary and may be skipped during a training/inference run.

### `extract_football_windows.py`

Purpose: convert raw Metrica CSV tracking data into fixed real ball windows.

Main logic is in `main()`:

- `load_tracking()` parses the three-row Metrica header;
- `entity_xy()` converts normalized coordinates into pitch metres;
- `find_start_index()` selects a requested time or frame;
- `extract_single_window()` extracts one demonstration window;
- `extract_fixed_windows()` supports scanning many windows;
- `detect_change_points()` detects transient direction/speed changes;
- `trajectory_diagnostics()` records missing values and extreme jumps;
- prefix and suffix arrays are created when `--prefix-T` is provided.

Important options:

- `--T`: complete window duration;
- `--prefix-T`: observed duration inside the complete window;
- `--start-time` or `--start-frame`: choose one exact demonstration;
- `--stride`: spacing between windows in scan mode;
- `--max-gap-fraction`: allowed missing-data fraction.

### `generate_model_voting_data.py`

Purpose: generate balanced synthetic tracks for all candidate models.

Functions:

- `load_real_condition_pool()`: obtains real start/end/change-point conditions;
- `synthetic_condition_pool()`: fallback when no real window file exists;
- `sample_conditions()`: selects condition rows and aligns prefix length;
- `main()`: samples theta, simulates tracks, pads parameters, and saves data.

The generator creates the ground-truth labels needed for contrastive training:

```text
track, model_id, theta, parameter_mask, condition
```

### `plot_model_voting_dataset.py`

Purpose: visually verify that synthetic model families are diverse.

Functions:

- `load_dataset()`: loads the NPZ archive;
- `choose_indices()`: limits the number of displayed tracks;
- `plot_tracks_by_model()`: overlays tracks for each model family;
- `plot_model_prior()`: shows the empirical discrete probability of every SDE
  family and verifies that the dataset is balanced;
- `theoretical_prior_density()`: evaluates the configured uniform or
  log-uniform density for comparison with sampled theta;
- `plot_model_parameter_histograms()`: gives every theta dimension its own
  axis, scale, bounds, and theoretical prior curve;
- `plot_condition_priors()`: shows start-position, target-position, and
  change-point distributions;
- `plot_displacement_summary()`: compares prior-predictive speed,
  displacement, and path length;
- `main()`: writes all diagnostic figures.

Generated figures:

```text
tracks_by_model.png
model_prior.png
parameter_priors.png
condition_priors.png
prior_predictive_trajectory_statistics.png
```

The condition plot also reports the number of unique start, target, and
change-point values. The current dataset was bootstrapped from one real window,
so these condition priors are point masses rather than diverse distributions.
The tool prints a warning when this occurs. Parameter and stochastic trajectory
diversity can still be present, but final training should use multiple real
windows or the synthetic condition pool.

This script should be run before training. If every model produces nearly the
same trajectories, the classifier has no meaningful model-selection problem.

### `train_model_voting_ratio.py`

Purpose: train the neural classifier used as the likelihood-ratio surrogate.

Functions:

- `roll_negative()`: mismatches model/theta values while preserving the track
  condition;
- `batch_loss_and_metrics()`: computes positive/negative logits, binary
  cross-entropy, accuracy, and the log-ratio gap;
- `run_epoch()`: performs one training or validation pass;
- `validation_metrics_by_model()`: reports accuracy and score gap separately
  for each true model family;
- `main()`: creates data loaders, trains, logs CSV history, and saves the best
  checkpoint.

Positive example:

```text
(track_i, model_i, theta_i, condition_i) -> 1
```

Negative example:

```text
(track_i, model_j, theta_j, condition_i) -> 0
```

### `evaluate_synthetic_model_recovery.py`

Purpose: test model-family selection on fresh simulated tracks.

Functions:

- `plot_confusion_matrix()`: renders true model versus selected model;
- `main()`: simulates fresh cases, estimates prior-integrated evidence for all
  candidate models, and reports recovery accuracy and model log score.

This test is stronger than classifier validation accuracy because it evaluates
the actual model-voting decision rule.

### `recover_model_voting_posterior.py`

Purpose: infer approximate model weights and theta posterior chains from one
real observed prefix.

Functions:

- `load_checkpoint()`: reconstructs the trained classifier;
- `checkpoint_array()`: safely reads normalization arrays;
- `load_observed_window()`: loads only the prefix and retains the suffix for
  later evaluation;
- `normalize_track()`: applies training-set normalization;
- `score_params()`: evaluates classifier logits for candidate theta batches;
- `log_prior()`: implements model-specific uniform/log-uniform priors;
- `proposal_scale_for_model()`: determines random-walk proposal widths;
- `run_model_mcmc()`: performs Metropolis-Hastings for one model;
- `logmeanexp()`: integrates likelihood-ratio scores over prior samples;
- `softmax()`: combines model evidence ratios under equal model priors;
- `main()`: orchestrates all model chains and saves results.

The model weight calculation is:

```text
log evidence ratio(model)
  approximately log mean exp(classifier_logit(x, model, theta))
  for theta sampled from prior(theta | model)
```

### `evaluate_model_voting.py`

Purpose: turn posterior samples into future paths and quantitative metrics.

Functions:

- `load_npz()`: loads posterior data;
- `sample_posterior_paths()`: samples model/theta pairs and simulates futures;
- `plot_posterior_predictive_paths()`: draws the observed prefix, true suffix,
  and sampled paths;
- `plot_endpoint_density()`: plots the predictive final-position density;
- `plot_model_votes()`: displays approximate model weights;
- `predictive_region_coverage()`: computes radial 50/80/90% coverage;
- `plot_winning_parameter_histograms()`: shows theta uncertainty for the
  highest-weight model;
- `main()`: calculates metrics and writes all outputs.

### Visualization Scripts

`football_tracking_viz.py` renders one static frame and short trail.

`football_window_clip.py` creates a raw GIF/MP4 clip. Its helper functions find
game files, extract player/ball coordinates, select a time interval, and save
the animation.

`football_model_voting_clip.py` creates a similar clip with a live model-score
gauge. Its helper functions normalize recent tracks, score one sliding window,
precompute votes, and render the animation.

`plot_real_window_segments.py` plots one extracted track and its detected
change points.

## 9. Reusable Source Modules

### Active Source Status

| Module | Status | Used by |
|---|---|---|
| `src/data/football_tracking.py` | Active | Extraction and visualization tools |
| `src/data/trajectory_features.py` | Active | Diagnostics and segmentation |
| `src/data/segmentation.py` | Active | Extraction, generation, posterior recovery |
| `src/data/model_voting_dataset.py` | Active | Ratio-classifier training |
| `src/models/encoder.py` | Active shared | Model-voting classifier and archived demos |
| `src/models/model_voting_ratio.py` | Active | Training, recovery, synthetic evaluation |
| `src/sde/model_voting.py` | Active | Generation, recovery, future simulation |
| `src/sde/football_ou.py` | Active shared | OU candidate plus archived OU baseline |
| `src/utils/football_viz.py` | Active | Evaluation and visualization tools |

Legacy source status:

| Module group | Status | Reason retained |
|---|---|---|
| `src/legacy/lorenz/` | Legacy | Supports the archived Lorenz demonstration and earlier experiments. |
| `src/legacy/ou/football_dataset.py` | Legacy | Supports the archived standalone OU workflow. |
| `src/legacy/inference/mcmc.py` | Legacy reference only | Old unused inference prototype; not imported by the active pipeline. |

Nothing under `src/legacy` is required for the current model-voting run.

### `src/data/football_tracking.py`

Owns CSV parsing, coordinate conversion, entity lookup, fixed-window
extraction, single-window extraction, and start-time/frame selection.

### `src/data/trajectory_features.py`

Computes smoothed positions, finite-difference velocity, speed, acceleration,
heading, turn angle, and trajectory diagnostics.

### `src/data/segmentation.py`

Detects change points from turn-angle and speed changes, enforces minimum
segment lengths, and provides an evenly spaced fallback.

### `src/data/model_voting_dataset.py`

Loads the synthetic NPZ file, normalizes trajectories, converts arrays to
PyTorch tensors, and provides `track`, `params`, `param_mask`, `model_id`, and
`condition` for each training row.

### `src/sde/model_voting.py`

Defines candidate names, parameter names, prior bounds, parameter padding and
normalization, and Brownian/constant/piecewise simulation. It dispatches OU
simulation to `src/sde/football_ou.py`.

### `src/models/encoder.py`

`TrajectoryEncoder` converts a `(batch, 2, steps)` trajectory into a fixed-size
feature vector using one-dimensional convolutions and pooling.

### `src/models/model_voting_ratio.py`

`ModelVotingRatioClassifier` combines:

```text
trajectory encoder
+ model embedding
+ theta and parameter-mask encoder
+ condition encoder
-> binary classifier logit
```

### `src/utils/football_viz.py`

Contains reusable pitch drawing, player drawing, and static tracking-frame
visualization functions.

## 10. How To Read The Results

### Training History

Good signs:

- training and validation losses decrease together;
- validation accuracy is above chance but not based only on one model;
- every per-model score gap is positive;
- validation loss does not rise while training loss keeps falling.

### MCMC Summary

Check:

- acceptance rate for every model;
- whether chains explore rather than remain constant;
- whether posterior histograms concentrate only at prior boundaries;
- stability when changing seed and proposal length.

An acceptance rate near zero means proposals are too large or the target is
pathological. A rate near one can mean proposals are too small and mixing is
slow.

### Model Weights

Model weights are approximate. They depend on:

- simulator realism;
- classifier ratio quality;
- prior ranges;
- number of evidence samples;
- equal model-prior assumption.

A high model weight does not imply a good real forecast. Always inspect ADE,
FDE, coverage, and the path plots.

### Predictive Coverage

If a nominal 90% predictive region almost never contains the true trajectory,
the posterior is overconfident or the simulator family is misspecified.

Coverage from one window cannot establish calibration. Aggregate it over many
windows and compare the observed coverage frequency with the nominal level.

## 11. Project Advantages

- The result is probabilistic rather than a falsely certain single trajectory.
- SBI permits training without real ground-truth SDE parameters.
- Contrastive ratio learning connects neural networks with Bayesian inference.
- MCMC produces interpretable parameter distributions.
- Model voting avoids forcing every ball trajectory into one equation family.
- Prefix/suffix separation prevents direct future leakage.
- Candidate equations and theta parameters remain interpretable.
- Fresh synthetic model recovery evaluates the actual model-selection rule.
- The code is modular: tracking, simulation, training, inference, and plotting
  are separated.
- The complete pipeline can be demonstrated with real Metrica tracking data.

## 12. Project Disadvantages And Risks

- The candidate simulator family is still naive relative to real football.
- Passes, shots, collisions, possession changes, and player contacts are not
  explicitly represented.
- Future direction-change times are not inferred; the current forecast assumes
  continuation of the latest regime.
- The OU equilibrium is a conservative stop/settle assumption.
- Synthetic model recovery does not guarantee real-data validity.
- Current training conditions are bootstrapped from very limited real windows.
- Approximate model evidence can be sensitive to prior ranges and classifier
  calibration.
- Separate MCMC chains do not directly jump between model dimensions.
- Random-walk MCMC may mix slowly, especially for seven piecewise parameters.
- One real held-out suffix cannot establish uncertainty calibration.
- Pitch clipping changes the transition distribution near boundaries.
- The encoder sees positions but no explicit event, possession, or contact
  information.
- A visually dense path cloud can hide poor quantitative accuracy.

## 13. Outlook And Prioritized Improvements

### Priority 1: Evaluation Over Many Real Windows

Extract many non-overlapping windows from both Sample Games. Report aggregate:

- ADE and FDE distributions;
- 50/80/90% predictive coverage;
- performance by speed and turn intensity;
- results by match and period;
- bootstrap confidence intervals.

This is the most important next step because current real evaluation uses one
demonstration window.

### Priority 2: Simple Forecast Baselines

Implement and compare against:

- stationary ball;
- last observed velocity;
- damped velocity;
- empirical Gaussian velocity noise.

The SBI system is useful only if it improves predictive accuracy or calibrated
uncertainty over these simpler approaches.

### Priority 3: Synthetic Parameter Recovery

For fresh tracks with known theta:

1. run MCMC;
2. calculate posterior bias;
3. measure interval width;
4. check 50/80/90% theta coverage;
5. inspect chain convergence and effective sample size.

This directly tests the PDF requirement of learning SDE parameters.

### Priority 4: Future Change-Point Uncertainty

Extend the piecewise model with latent future turn variables:

```text
theta = (future_tau, velocity_before, velocity_after, sigma)
```

Possible alternatives include a switching SDE, hidden Markov model, or
continuous-time jump process. The output should remain a distribution over
turn time and direction.

### Priority 5: Better Ball Physics

Add velocity as part of the state:

```text
state = (x, y, vx, vy)
```

Candidate dynamics can then include damping, rolling friction, acceleration,
and impact-like velocity changes. This is more physically natural than directly
diffusing position.

### Priority 6: Better Evidence And Sampling

- repeat prior Monte Carlo evidence with several seeds;
- report Monte Carlo standard error;
- increase prior samples adaptively;
- tune proposals per parameter;
- run multiple MCMC chains;
- calculate R-hat and effective sample size;
- consider sequential neural ratio estimation or normalizing-flow posteriors.

### Priority 7: More Diverse Training Conditions

Use many start positions, targets, speeds, and segment patterns from both games.
Separate synthetic train, validation, and test generation seeds. Avoid
evaluating only on conditions copied from one real prefix.

## 14. Common Problems

### `ModuleNotFoundError: No module named 'src'`

Run commands from the repository root and use the scripts shown in this guide.
Do not run a copied script from an unrelated folder.

### PyTorch Checkpoint `weights_only` Error

The project checkpoints contain metadata as well as tensors. Project scripts
load trusted local checkpoints with `weights_only=False`. Do not apply this to
untrusted checkpoints.

### Missing Posterior Output

The output cleanup intentionally removed stale posterior/evaluation results.
Run Steps 5 and 6 again with the corrected evidence and future-condition code.

### Very Confident Weight But Bad Forecast

This indicates simulator misspecification or ratio miscalibration. Model weight
only compares the available candidates. If every candidate is poor, the best
candidate can still receive a high relative weight.

### Zero Predictive Coverage

The posterior path cloud is too narrow or centred on the wrong motion. Check
the candidate equations, future assumptions, prior noise bounds, and whether
the observed real track lies inside the synthetic training distribution.

## 15. Current Honest Conclusion

The project implements a complete model-voting SBI prototype for football-ball
trajectories:

```text
real prefix
-> neural ratio estimation
-> prior-integrated approximate model weights
-> per-model MCMC theta posterior
-> posterior predictive future distribution
```

Fresh synthetic testing shows that the classifier/evidence mechanism can
distinguish the four implemented simulators. The current real-window forecast
is not yet well calibrated, so the project should be described as a working
probabilistic inference pipeline with demonstrated simulator recovery and an
open simulator-to-reality gap.
