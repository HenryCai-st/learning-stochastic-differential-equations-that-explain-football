# Football SDE Track Prediction — Implementation Plan (Updated)

## Problem Statement (from groupmates' notes)

The core difficulty is that an SDE ends differently every run — it is **only probabilistic, never deterministic**. Predicting a single parameter vector only gives us the mean, which is inaccurate. The correct target is:

$$p(\theta \mid \text{observed track}) \quad \longrightarrow \quad \text{distribution over future tracks}$$

---

## Full Pipeline (5 steps from groupmates' notes)

```
1. Simulate SDE using known parameters         [src/sde/crw_sde.py]
         ↓
2. Generate dataset using prior distribution   [scripts/generate_crw_dataset.py]
   - Different starting points (as in real data)
   - Multiple SDE families → 1-hot encoding
         ↓
3. Build binary classifier (contrastive learning)   [src/models/crw_models.py]
   - Input: concatenate(1-hot + params, trajectory encoding) or map via MLP
   - Architecture: trajectory encoder CNN + parameter encoder MLP + binary head
   - Label param_set_A + track_B → 0 (false)
   - Label param_set_B + track_B → 1 (true)
         ↓
4. Use feature extraction to output prob. distribution over parameters
   [scripts/evaluate_crw.py — scoring step]
         ↓
5. Use distribution to output prob. distribution of future paths
   [scripts/evaluate_crw.py — sampling + visualization step]
```

---

## File Audit

### ✅ Active Pipeline Files (do not touch)

| File | Role |
|---|---|
| `lorenz_dataset.npz` | Generated Lorenz dataset (1000 samples) |
| `src/sde/lorenz_sde.py` | Production Lorenz SDE simulator (torchsde) |
| `src/data/lorenz_dataset.py` | PyTorch `LorenzTrajectoryDataset` + `LorenzPairDataset` |
| `src/models/lorenz_models.py` | `TrajectoryEncoder1D`, `LorenzRatioEstimator`, etc. |
| `scripts/train_lorenz.py` | Training entry point for `regime`, `params`, `ratio` tasks |
| `scripts/lorenz_sbi_numpy_demo.py` | Working NumPy-only end-to-end SBI demo |
| `scripts/run_lorenz.py` | Grid + sensitivity runner for Lorenz SDE |
| `scripts/movements` | Four football SDE simulators (Brownian, OU, CRW, Social Force) |
| `lorenz_pipeline.md` | Authoritative pipeline documentation |
| `lorenz_sde_design_notes.md` | Parameter ranges, normalization design notes |
| `PROJECT_DEMO_REPORT.md` | Demo results and workflow description |
| `TODO.md` | Current task checklist |
| `README.md` | Setup and usage instructions |
| `requirements.txt` | Python dependencies |

### 🗄️ Archived Files (prefixed `_archive_`)

These files were early experiments / exploration steps that are **superseded** by the current pipeline. They are kept for reference but not part of the active codebase.

| Renamed File | Reason Archived |
|---|---|
| `scripts/_archive_1_d_sde_demo_copilot.py` | Toy 1D GBM demo — not Lorenz, not football, not used anywhere |
| `scripts/_archive_3_dim_lorenz_demo.py` | Early 2D Lorenz experiment with wrong commented-out z-axis; superseded by `lorenz_sde.py` |
| `scripts/_archive_generate-sde-plot.py` | One-off Lorenz visualisation, not connected to dataloader or training |
| `scripts/_archive_multi_lines_gen.py` | Early multi-trajectory batch demo, superseded by `lorenz_sde.py` |
| `scripts/_archive_dataset_gen.py` | Early dataset generator that produced `lorenz_dataset.npz` — generation is done; file no longer needed to re-run |
| `_archive_lorenz_sde_improved.py` | Duplicate of `src/sde/lorenz_sde.py` living at root; improvements merged into the src version |
| `_archive_documentation.md` | Early raw Q&A notes replaced by `lorenz_sde_design_notes.md` |

---

## What We Build (New Files)

### Component 1 — CRW Football SDE Simulator

#### [NEW] `src/sde/crw_sde.py`
Clean Euler-Maruyama CRW simulator, reusing the structure from `scripts/movements`.

- **State**: `(x, y, v, θ)` — position, speed, heading angle
- **Parameters** `θ = (kappa, v0, sigma_theta, sigma_v)`:
  - `kappa` — turning rate (how quickly heading reverts to 0)
  - `v0` — preferred cruising speed
  - `sigma_theta` — directional noise (heading randomness)
  - `sigma_v` — speed noise
- Output: `(time, 2)` — only `(x, y)`, matching Lorenz format
- Public API:
  - `simulate(kappa, v0, sigma_theta, sigma_v, x0, y0, theta0, v0_init)` → `np.ndarray (steps, 2)`
  - `generate_dataset(n_samples, prior_ranges, ...)` → saves `.npz`

**Prior ranges** for sampling:
```
kappa       ~ Uniform(0.1, 3.0)   # turning rate
v0          ~ Uniform(1.0, 8.0)   # preferred speed (m/s)
sigma_theta ~ Uniform(0.1, 1.5)   # directional noise
sigma_v     ~ Uniform(0.1, 2.0)   # speed noise
x0, y0      ~ Uniform pitch boundaries (100×68m)
```

---

### Component 2 — Football Dataloader

#### [NEW] `src/data/crw_dataset.py`
Mirrors `lorenz_dataset.py` exactly, adapted for CRW.

- **`CRWTrajectoryDataset`**
  - Loads `.npz` with `trajectories (N, T, 2)`, `params (N, 4)`, optional `model_id (N,)`
  - Normalizes trajectory to `[-1, 1]` per coordinate
  - Normalizes params: `log1p(sigma_theta)`, `log1p(sigma_v)` then min-max
  - Returns `(channels=2, time)` tensor for 1D CNN
  - **One-hot model encoding** (TODO §3): if `model_id` is present, concatenates a 4-dim one-hot `[0,0,1,0]` (=CRW) with the param vector → `param_dim = 4+4 = 8` or `param_dim = 4` for single-model
  - Supports `split=train|val|test`

- **`CRWPairDataset`**
  - Wraps `CRWTrajectoryDataset`
  - 50/50 matched/mismatched pairs at each `__getitem__` call
  - Positive: `(theta_A, track_A)` → label 1
  - Negative: `(theta_A, track_B≠A)` → label 0

---

### Component 3 — Football Model

#### [NEW] `src/models/crw_models.py`
Reuses `TrajectoryEncoder1D` from `lorenz_models.py`.

- **`CRWRatioEstimator`** (main model for SBI)
  - `trajectory_encoder`: `TrajectoryEncoder1D(in_channels=2, hidden_dim=128)`
  - `param_encoder`: MLP `param_dim → 128 → 128` (param_dim = 4 or 8 with one-hot)
  - `head`: `256 → 128 → 1` logit
  - Input: `(batch, 2, T)` trajectory + `(batch, 4|8)` params
  - Output: one logit (BCEWithLogitsLoss)

- **`CRWParameterRegressor`** (deterministic baseline)
  - `trajectory_encoder`: same CNN
  - `head`: `128 → 4` with sigmoid (predicts normalized params directly)
  - Loss: L1

---

### Component 4 — CRW Dataset Generation Script

#### [NEW] `scripts/generate_crw_dataset.py`
Generates synthetic football CRW trajectories and saves `.npz`.

```
python scripts/generate_crw_dataset.py --n-samples 1000 --out crw_dataset.npz --seed 42
```

Arguments: `--n-samples`, `--out`, `--T`, `--dt`, `--seed`, `--model-id` (for multi-model one-hot)

Output `.npz` contains:
- `trajectories`: `(N, T, 2)` — x, y positions
- `params`: `(N, 4)` — raw `(kappa, v0, sigma_theta, sigma_v)`
- `model_id`: `(N,)` — integer 0=Brownian, 1=OU, 2=CRW, 3=SocialForce

---

### Component 5 — Training Script

#### [NEW] `scripts/train_crw.py`
Same structure as `train_lorenz.py`.

```
python scripts/train_crw.py --task ratio --epochs 50 --dataset crw_dataset.npz
python scripts/train_crw.py --task params --epochs 30 --dataset crw_dataset.npz
```

Arguments: `--task ratio|params`, `--epochs`, `--batch-size`, `--lr`, `--dataset`, `--out-dir`, `--device`, `--seed`, `--use-onehot`

Saves:
- `outputs/crw_training/<task>/history.csv`
- `outputs/crw_training/<task>/best.pt`

---

### Component 6 — Evaluation Script

#### [NEW] `scripts/evaluate_crw.py`
The scoring → posterior → future-path pipeline. Implements **steps 4 and 5** from the groupmates' notes.

```
python scripts/evaluate_crw.py --checkpoint outputs/crw_training/ratio/best.pt --dataset crw_dataset.npz
```

Steps:
1. Load trained ratio estimator from `best.pt`
2. Pick one observed trajectory (from val/test split)
3. For every candidate parameter vector in the dataset: compute `score = sigmoid(model(track_obs, theta_i))`
4. Normalize scores to get approximate posterior weights
5. Sample top-K parameter sets from the posterior
6. Simulate `M` future CRW trajectories from top-K parameters
7. Save visualizations:
   - `posterior_scores.png` — scatter `(kappa, sigma_theta)` bubble-sized by score
   - `future_paths.png` — observed track + M future paths on a football pitch
   - `parameter_histograms.png` — histograms of posterior `(kappa, v0, sigma_theta, sigma_v)`
   - `top_candidates.csv` + `summary.json`

Arguments: `--checkpoint`, `--dataset`, `--out-dir`, `--top-k`, `--future-samples`, `--observed-index`, `--device`

---

### Component 7 — Config File

#### [NEW] `configs/crw_default.json`
Default hyperparameters so that training and evaluation are reproducible without memorizing CLI arguments.

---

## Updated TODO Checklist (reconciled with TODO.md)

### Lorenz (still in progress)
- [ ] Repair PyTorch `.venv` environment
- [ ] Train `regime` classifier (sanity check)
- [ ] Train `params` regressor (deterministic baseline)
- [ ] Train `ratio` estimator (main SBI model)
- [ ] Save `best.pt` for each run
- [ ] One-hot encoding for multiple model families (TODO §3.5)

### Football CRW (new work, all unchecked)
- [ ] `src/sde/crw_sde.py` — CRW simulator
- [ ] `scripts/generate_crw_dataset.py` — dataset generation
- [ ] `src/data/crw_dataset.py` — dataloader with one-hot support
- [ ] `src/models/crw_models.py` — ratio estimator + regressor
- [ ] `scripts/train_crw.py` — training script
- [ ] `scripts/evaluate_crw.py` — posterior scoring + future paths
- [ ] Visualize football future positions as heatmap (TODO §6)

---

## Final File Map

```
deep-learning-lab/
│
├── src/
│   ├── sde/
│   │   ├── lorenz_sde.py              ✅ existing
│   │   └── crw_sde.py                 🆕 new
│   ├── data/
│   │   ├── lorenz_dataset.py          ✅ existing
│   │   └── crw_dataset.py             🆕 new
│   └── models/
│       ├── lorenz_models.py           ✅ existing
│       └── crw_models.py              🆕 new
│
├── scripts/
│   ├── lorenz_sbi_numpy_demo.py       ✅ existing (working demo)
│   ├── train_lorenz.py                ✅ existing
│   ├── run_lorenz.py                  ✅ existing
│   ├── movements                      ✅ existing (football simulators)
│   ├── generate_crw_dataset.py        🆕 new
│   ├── train_crw.py                   🆕 new
│   ├── evaluate_crw.py                🆕 new
│   ├── _archive_1_d_sde_demo_copilot.py   🗄️ archived
│   ├── _archive_3_dim_lorenz_demo.py      🗄️ archived
│   ├── _archive_generate-sde-plot.py      🗄️ archived
│   ├── _archive_multi_lines_gen.py        🗄️ archived
│   └── _archive_dataset_gen.py            🗄️ archived
│
├── configs/
│   └── crw_default.json               🆕 new
│
├── _archive_lorenz_sde_improved.py    🗄️ archived
├── _archive_documentation.md         🗄️ archived
├── lorenz_dataset.npz                 ✅ existing dataset
├── lorenz_pipeline.md                 ✅ authoritative docs
├── lorenz_sde_design_notes.md         ✅ design reference
├── PROJECT_DEMO_REPORT.md             ✅ results
├── TODO.md                            ✅ task list
├── README.md                          ✅ setup guide
└── requirements.txt                   ✅ dependencies
```

---

## Verification Plan

| Step | Command | Expected |
|---|---|---|
| Generate CRW dataset | `python scripts/generate_crw_dataset.py --n-samples 200 --out crw_dataset.npz` | `crw_dataset.npz` exists, arrays have correct shapes |
| Smoke-test training | `python scripts/train_crw.py --task ratio --epochs 2 --limit-train 128 --limit-val 32 --device cpu` | Loss decreases, `best.pt` saved |
| Evaluate + visualize | `python scripts/evaluate_crw.py --checkpoint outputs/.../best.pt --dataset crw_dataset.npz` | PNG outputs + CSV generated |
| Visual check | Open `future_paths.png` | Multiple plausible football paths shown on pitch |
