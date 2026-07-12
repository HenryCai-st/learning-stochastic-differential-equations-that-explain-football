# Deep Learning Project

## Current football SBI workflow

The agreed project scope is model-voting simulation-based inference for the
football **ball trajectory only**. Player movement, player roles, teammate
effects, and opponent effects are outside the current scope. The goal is to
infer a distribution
`p(model, theta | observed_track)` and then render a posterior predictive
distribution over future ball positions.

This is a probabilistic forecasting project. It does not claim that one SDE
produces the deterministic true future path.

Main implementation task list:

```text
SBI_MODEL_VOTING_IMPLEMENTATION_TASKS.md
```

The two-part project structure and next-stage roadmap are documented in:

```text
PROJECT_RESTRUCTURE_PLAN.md
```

For a beginner-oriented explanation of the full project, every active script,
all important functions, result interpretation, advantages, limitations, and
the improvement roadmap, read:

```text
MODEL_VOTING_PROJECT_GUIDE.md
```

The archived single-model OU baseline is documented separately in
`OU_BASELINE_WORKFLOW.md`.

Script organization:

```text
scripts/method_validation/      controlled synthetic method evaluation
scripts/football_case_study/    football extraction, inference, and forecasting
scripts/tools/                  optional plots, raw-data inspection, and clips
scripts/OU_workflow/            archived standalone OU baseline
scripts/Lorenz_workflow/        archived Lorenz demonstration
```

Source organization:

```text
src/sbi          shared ratio estimation, evidence, MCMC, and metadata
src/simulators   candidate stochastic motion models and priors
src/synthetic    controlled synthetic data adapters
src/football     football tracking, features, segmentation, and visualization
src/legacy       historical modules only
```

See `src/README.md` for a file-by-file active/legacy table.

Controlled method-validation run order:

```powershell
python scripts\method_validation\generate_synthetic_benchmark.py `
  --out-dir data\method_validation `
  --n-train-per-model 1000 `
  --n-validation-per-model 100 `
  --n-test-per-model 100

python scripts\method_validation\train_ratio_estimator.py `
  --train-data data\method_validation\train.npz `
  --validation-data data\method_validation\validation.npz `
  --epochs 100 `
  --out-dir checkpoints\method_validation

python scripts\method_validation\evaluate_synthetic_model_recovery.py `
  --checkpoint checkpoints\method_validation\ratio_estimator_best.pt `
  --test-data data\method_validation\test.npz `
  --n-evidence-samples 512 `
  --out-dir outputs\method_validation\model_recovery

python scripts\method_validation\evaluate_synthetic_parameter_recovery.py `
  --checkpoint checkpoints\method_validation\ratio_estimator_best.pt `
  --test-data data\method_validation\test.npz `
  --cases-per-model 25 `
  --chains 4 `
  --mcmc-steps 2400 `
  --burn-in 800

python scripts\method_validation\generate_synthetic_forecast_benchmark.py `
  --test-data data\method_validation\test.npz `
  --future-T 1.0 `
  --out data\method_validation\forecast_test.npz

python scripts\method_validation\evaluate_synthetic_forecasts.py `
  --checkpoint checkpoints\method_validation\ratio_estimator_best.pt `
  --forecast-data data\method_validation\forecast_test.npz `
  --cases-per-model 25 `
  --n-evidence-samples 1024 `
  --n-paths 256

python scripts\tools\synthetic_forecast_validation_animation.py
```

Model recovery, parameter diagnostics, and controlled forecast results are
reported in `METHOD_VALIDATION_RESULTS.md`. The animation shows one observed
prefix followed by synchronized conditional predictions from all four models;
the highest-probability model and held-out truth are emphasized.

Football case-study run order:

```powershell
python scripts\football_case_study\extract_football_windows.py `
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

python scripts\football_case_study\generate_model_voting_data.py `
  --real-windows data\real_football_windows.npz `
  --n-per-model 1000 `
  --T 5.0 `
  --dt 0.04 `
  --out-dir data\model_voting_dataset

python scripts\tools\plot_model_voting_dataset.py `
  --dataset data\model_voting_dataset\dataset.npz `
  --out-dir outputs\model_voting_dataset_viz

python scripts\football_case_study\train_model_voting_ratio.py `
  --data-dir data\model_voting_dataset `
  --epochs 100 `
  --batch-size 128 `
  --out-dir checkpoints

python scripts\football_case_study\recover_model_voting_posterior.py `
  --real-windows data\real_football_windows.npz `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --window-index 0 `
  --mcmc-steps 3000 `
  --burn-in 800 `
  --n-evidence-samples 4096 `
  --out-dir outputs\model_voting_posterior

python scripts\football_case_study\evaluate_model_voting.py `
  --posterior outputs\model_voting_posterior\posterior_chains.npz `
  --n-paths 300 `
  --out-dir outputs\model_voting_evaluation

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

`--T` is the window duration. To extract one specific observed interval instead
of scanning all possible windows, add either `--start-time` or `--start-frame`:

```powershell
python scripts\football_case_study\extract_football_windows.py `
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

With `--prefix-T 2.0`, the first 2 seconds are saved as the observed input and
the remaining 3 seconds are saved as the future suffix for evaluation. The
model-voting data generator automatically trains on `prefix_tracks` when they
exist, so MCMC inference is conditioned on the 2-second prefix rather than the
full 5-second window.

To visually inspect the same kind of time window as a short clip:

```powershell
python scripts\tools\football_window_clip.py `
  --game data\Sample_Game_1 `
  --period 1 `
  --start-time 37.2 `
  --duration 5.0 `
  --frame-step 2 `
  --fps 12 `
  --out outputs\football_window_clip.gif
```

The raw clip above only shows tracking data. The model-voting clip in the run
order uses the trained ratio classifier and adds a live gauge showing which SDE
candidate currently best matches the recent ball trajectory.

The active workflow stores the observed prefix and held-out suffix directly in
`real_football_windows.npz`; posterior recovery and evaluation consume that
single data contract.

Generated NPZ files, checkpoints, and result directories now include schema,
Git commit, run arguments, simulator priors, and data-contract metadata.
Posterior recovery and synthetic evaluation fail early when trajectory length,
`dt`, model schema, or simulator priors do not match the checkpoint.

## Statistical interpretation

The ratio classifier is trained on matched and mismatched
`(track, model, theta)` pairs. For each candidate model, MCMC samples theta
conditional on the observed prefix. Model-family weights are estimated by
integrating the learned likelihood ratio over prior theta samples:

```text
log evidence ratio(model)
    = log mean exp(log_ratio(track, model, theta)), theta ~ prior(theta | model)
```

The resulting weights assume equal prior probability for all model families.
They are approximate SBI model probabilities and must be validated on fresh
synthetic data before being interpreted as calibrated probabilities.

For future prediction, OU uses the last observed position as its equilibrium.
The piecewise model continues the latest inferred velocity segment and assumes
no unobserved future turn. Predicting future turn times remains open work.

## Current validation status

Implemented:

- 2-second observed prefix and 3-second held-out suffix protocol
- contrastive neural ratio estimation
- per-model random-walk Metropolis-Hastings
- prior-integrated model evidence approximation
- posterior predictive paths, ADE/FDE, and 50/80/90% predictive-region coverage
- independent football-free train/validation/test benchmark
- 400-case model-recovery confusion matrix, 89.75% top-1 accuracy, and model log score

Still required for robust conclusions:

- improve piecewise parameter convergence and identifiability
- repeat training and parameter recovery across network seeds
- evaluation over many windows from both available Sample Games
- improve aggregate forecast error beyond the last-velocity baseline

## Setup Instructions

1. **Create and Activate a Virtual Environment:**
   Using `venv`:

   ```bash
   python -m venv venv
   ```

   * Windows: `venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`
2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Install PyTorch with CUDA 12.8:**
   If you have the CPU version installed, uninstall it first:

   ```bash
   pip uninstall torch torchvision
   ```

   Then install the CUDA-enabled version:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
