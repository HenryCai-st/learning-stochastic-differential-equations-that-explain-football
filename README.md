# Deep Learning Project

## Current football SBI workflow

The project scope is model-voting simulation-based inference for the football **ball trajectory**. The goal is to infer a distribution  `p(model, theta | observed_track)` and then render a posterior predictive distribution over future ball positions. 

This is a probabilistic forecasting project. It does not claim that one SDE produces the deterministic true future path.

Main implementation task list:

```text
SBI_MODEL_VOTING_IMPLEMENTATION_TASKS.md
```

The explanary doc of the full project, every active script, all important functions, result interpretation, advantages, limitations, and the improvement roadmap, please read:

```text
MODEL_VOTING_PROJECT_GUIDE.md
```

Script organization:

```text
scripts/model_voting_pipeline/  required six-stage training/inference workflow
scripts/tools/                  optional plots, raw-data inspection, and clips
scripts/OU_workflow/            archived standalone OU(Ornstein-Uhlenbeck) workflow
scripts/Lorenz_workflow/        archived Lorenz demonstration
```

Source organization:

```text
src/data, src/models, src/sde, src/utils   active model-voting modules
src/legacy                                historical modules only
```

See `src/README.md` for a file-by-file active/legacy table.

Current run order:

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

python scripts\model_voting_pipeline\evaluate_synthetic_model_recovery.py `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --dataset data\model_voting_dataset\dataset.npz `
  --n-cases 80 `
  --n-evidence-samples 512 `
  --out-dir outputs\synthetic_model_recovery

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

With `--prefix-T 2.0`, the first 2 seconds are saved as the observed input and the remaining 3 seconds are saved as the future suffix for evaluation. The model-voting data generator automatically trains on `prefix_tracks` when they
exist, so MCMC inference is conditioned on the 2-second prefix rather than the full 5-second window.

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

The raw clip above only shows tracking data. The model-voting clip in the run order uses the trained ratio classifier and adds a live gauge showing which SDE candidate currently best matches the recent ball trajectory.

Notes of clip rendering:

* `.gif` should work if your environment has `matplotlib` and Pillow.
* `.mp4` is supported too, but only if `ffmpeg` is installed.

## Statistical interpretation

The ratio classifier is trained on matched and mismatched  `(track, model, theta)` pairs. For each candidate model, MCMC samples theta conditional on the observed prefix. Model-family weights are estimated by integrating the learned likelihood ratio over prior theta samples:

```text
log evidence ratio(model)
    = log mean exp(log_ratio(track, model, theta)), theta ~ prior(theta | model)
```

$$
\log \text{ER}(m) = \log \mathbb{E}_{\theta \sim p(\theta \mid m)} \left[ \exp \left( \log r(t, m, \theta) \right) \right]
$$

The resulting weights assume equal prior probability for all model families. They are approximate SBI model probabilities and must be validated on fresh synthetic data before being interpreted as calibrated probabilities.

For future prediction, 

* OU uses the last observed position as its equilibrium.
* The piecewise model continues the latest inferred velocity segment and assumes no unobserved future turn.
* Predicting future turn times remains open work.

## Current validation status

Implemented:

- 2-second observed prefix and 3-second held-out suffix protocol
- contrastive neural ratio estimation
- per-model random-walk Metropolis-Hastings
- prior-integrated model evidence approximation
- posterior predictive paths, ADE/FDE, and 50/80/90% predictive-region coverage
- fresh synthetic model-recovery confusion matrix and model log score

Still required for robust conclusions:

- synthetic parameter posterior coverage against known theta
- evaluation over many windows from both available Sample Games
- aggregate calibration and comparison with simple motion baselines

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
