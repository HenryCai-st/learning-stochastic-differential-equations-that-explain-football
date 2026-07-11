# Deep Learning Project

## Current football SBI workflow

The active project direction is model-voting simulation-based inference for
football ball trajectories. The goal is to infer a distribution
`p(model, theta | observed_track)` and then render a posterior predictive
distribution over future ball positions.

Main implementation task list:

```text
SBI_MODEL_VOTING_IMPLEMENTATION_TASKS.md
```

Current run order:

```powershell
python scripts\extract_football_windows.py `
  --home data\Sample_Game_1\Sample_Game_1_RawTrackingData_Home_Team.csv `
  --away data\Sample_Game_1\Sample_Game_1_RawTrackingData_Away_Team.csv `
  --team home `
  --entity Ball `
  --T 5.0 `
  --dt 0.04 `
  --out data\real_football_windows.npz

python scripts\generate_model_voting_data.py `
  --real-windows data\real_football_windows.npz `
  --n-per-model 1000 `
  --T 5.0 `
  --dt 0.04 `
  --out-dir data\model_voting_dataset

python scripts\plot_model_voting_dataset.py `
  --dataset data\model_voting_dataset\dataset.npz `
  --out-dir outputs\model_voting_dataset_viz

python scripts\train_model_voting_ratio.py `
  --data-dir data\model_voting_dataset `
  --epochs 100 `
  --batch-size 128 `
  --out-dir checkpoints

python scripts\recover_model_voting_posterior.py `
  --real-windows data\real_football_windows.npz `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --window-index 0 `
  --mcmc-steps 3000 `
  --burn-in 800 `
  --out-dir outputs\model_voting_posterior

python scripts\evaluate_model_voting.py `
  --posterior outputs\model_voting_posterior\posterior_chains.npz `
  --n-paths 300 `
  --out-dir outputs\model_voting_evaluation
```

`--T` is the window duration. To extract one specific observed interval instead
of scanning all possible windows, add either `--start-time` or `--start-frame`:

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

To visually inspect the same kind of time window as a short clip:

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

The stricter prediction protocol has a first baseline implementation: split
each real window into observed prefix and held-out future suffix, infer only
from the prefix, then score the predictive distribution against the held-out
future.

```powershell
python scripts\extract_prefix_suffix_windows.py `
  --input data\real_football_windows.npz `
  --prefix-seconds 2.0 `
  --out data\real_football_prefix_suffix_windows.npz

python scripts\recover_model_voting_prefix_posterior.py `
  --windows data\real_football_prefix_suffix_windows.npz `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --window-index 0 `
  --target-strategy prefix_end `
  --mcmc-steps 3000 `
  --burn-in 800 `
  --out-dir outputs\prefix_suffix_posterior

python scripts\evaluate_prefix_suffix_prediction.py `
  --posterior outputs\prefix_suffix_posterior\posterior_chains.npz `
  --n-paths 300 `
  --out-dir outputs\prefix_suffix_prediction
```

Current caveat: the prefix/suffix protocol runs, but the first baseline is not
yet a good future predictor. The model was trained with full-window target
conditioning, so prediction without the future endpoint needs more work.

## Branch description

This branch uses [lorenz system](https://en.wikipedia.org/wiki/Lorenz_system) for simulating the SDE. A template script can be found in subfolder `scripts` to generate and plot.
Planned steps described in `documentation.md`. `lorenz_sde_design__notes.md` gives thoughts about it.

## Project description

We are to achieve "trajectories to parameters", the inverse design of simulating differential equation using parameters based on "Learning Stochastic Differential Equations that Explain Football".

For the current, following steps are planned:

1. **Reproduce Simulation**

Know how we generate trajectories based on parameters. This is assumed to be provided.

2. **Modelling**

Model the whole piepline of project. e.g.:

- Design parameters for predicting the trajectories
- Decide shape of dataset, 2d with trajectory length, type image or sequence of coordinates
- Decide architecture used for prediction
- Parameter sensitivity analysis: find out how parameters influence distribution of SDEs

3. **Prototype**

- Start with simple CNN, we can set up binary classifier first to let it separate one distribution of trajectories to another
- Note: Need to find parameters with maximal difference in distribution regarding parameter sensitivity analysis
- Then if result looks good, use transfer learning to finetune to do regression on parameters

4. **Iteration**

- After prototype, one evaluates using real generated simulation and simulation based on predicted parameters
- Then try use larger dataset by ust generating more simulations, more advanced architecture, additionals techniques

5. **Report**

- This should begin as soon as the pipeline proceeds
- Take notes on every iteration, modelling
- We can separate different iterations by branching
- Also note down expermient and model design every iteration

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

## SDE Generation Instructions

1. **Running batch generation**

   ```
   python scripts/run_lorenz.py
   ```
   3 modes to choose via `--mode`:

   - grid: runs over all permutations of parameter sets
   - sensitivity: given base params, only change one parameter at a time
   - both

   The parameters are configured inside run_lorenz.py
