# Model-Voting SBI Minimal Experiment Results

Date: 2026-07-06

Branch: `yuyang`

Python used:

```text
D:\miniconda3\envs\dl\python.exe
```

The default `python` at `D:\miniconda3\python.exe` did not have `numpy`, so all
experiments were run with the existing `dl` conda environment.

## Summary

The three requested minimal experiments were implemented and run.

Result:

```text
Experiment 1: pass
Experiment 2: pass after minimal debug retraining
Experiment 3: pass
```

The only code added was:

```text
scripts/evaluate_model_voting_synthetic.py
```

No new SDE model, prefix/suffix prediction, social-force model, Transformer, or
hyperparameter sweep was added.

## Experiment 1: Pipeline Smoke Test

### 1.1 Real Window Extraction

Command:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\extract_football_windows.py `
  --home data\Sample_Game_1\Sample_Game_1_RawTrackingData_Home_Team.csv `
  --away data\Sample_Game_1\Sample_Game_1_RawTrackingData_Away_Team.csv `
  --team home `
  --entity Ball `
  --T 5.0 `
  --dt 0.04 `
  --out data\real_football_windows.npz
```

Output:

```json
{
  "out": "data\\real_football_windows.npz",
  "windows": 3067,
  "steps": 125
}
```

Status: pass.

### 1.2 Small Mixed-Model Dataset

Command:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\generate_model_voting_data.py `
  --real-windows data\real_football_windows.npz `
  --n-per-model 100 `
  --T 5.0 `
  --dt 0.04 `
  --out-dir data\model_voting_dataset_test
```

Output:

```json
{
  "out": "data\\model_voting_dataset_test\\dataset.npz",
  "models": [
    "brownian",
    "constant_velocity",
    "ou_target",
    "piecewise_velocity"
  ],
  "tracks": 400,
  "steps": 125,
  "condition_sources": {
    "brownian": "real_window_bootstrap",
    "constant_velocity": "real_window_bootstrap",
    "ou_target": "real_window_bootstrap",
    "piecewise_velocity": "real_window_bootstrap"
  }
}
```

Status: pass.

### 1.3 Dataset Visualization

Command:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\plot_model_voting_dataset.py `
  --dataset data\model_voting_dataset_test\dataset.npz `
  --out-dir outputs\model_voting_dataset_viz_test
```

Output:

```text
outputs/model_voting_dataset_viz_test
```

Status: pass.

### 1.4 Small Ratio Classifier Training

Initial command from plan:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\train_model_voting_ratio.py `
  --data-dir data\model_voting_dataset_test `
  --epochs 20 `
  --batch-size 128 `
  --out-dir checkpoints_test
```

Initial 20-epoch result:

```text
best val acc: 68.75%
best val gap: 0.758
checkpoint saved: checkpoints_test/model_voting_ratio_best.pt
history saved: checkpoints_test/model_voting_ratio_history.csv
```

This passed Experiment 1.

## Experiment 2: Synthetic Known-Model Recovery

Implemented:

```text
scripts/evaluate_model_voting_synthetic.py
```

The script evaluates each synthetic track against all implemented candidate
models using candidate-grid scoring:

```text
model_score = logmeanexp(classifier logits)
model_vote = softmax(model_scores)
```

### First Required Run

Command:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\evaluate_model_voting_synthetic.py `
  --data-dir data\model_voting_dataset_test `
  --checkpoint checkpoints_test\model_voting_ratio_best.pt `
  --n-eval 100 `
  --n-candidates 512 `
  --out-dir outputs\model_voting_synthetic_eval
```

Initial result after 20-epoch checkpoint:

```json
{
  "top1_model_accuracy": 0.49,
  "top2_model_accuracy": 0.93,
  "mean_vote_weight_for_true_model": 0.4143638467499424,
  "mean_entropy_of_model_votes": 1.0729628668185476
}
```

This was below the requested `top1_model_accuracy > 50%` threshold.

Diagnostic all-400 evaluation with the same checkpoint:

```json
{
  "top1_model_accuracy": 0.505,
  "top2_model_accuracy": 0.8575
}
```

Interpretation: the pipeline had signal, but the 20-epoch small checkpoint was
not stable enough for the specified 100-sample acceptance test.

### Minimal Debug Retraining

The same small dataset was retrained for 50 epochs, without changing the model
or dataset:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\train_model_voting_ratio.py `
  --data-dir data\model_voting_dataset_test `
  --epochs 50 `
  --batch-size 128 `
  --out-dir checkpoints_test
```

Best small-checkpoint training point:

```text
epoch 36
val loss: 0.5247
val acc: 75.00%
val gap: 2.098
```

### Passing Run

Re-running the required synthetic evaluation:

```json
{
  "top1_model_accuracy": 0.57,
  "top2_model_accuracy": 0.96,
  "mean_vote_weight_for_true_model": 0.4617484094293714,
  "mean_entropy_of_model_votes": 0.8900859827625801,
  "per_model_accuracy": {
    "brownian": 0.6923076923076923,
    "constant_velocity": 0.22727272727272727,
    "ou_target": 0.6538461538461539,
    "piecewise_velocity": 0.6538461538461539
  }
}
```

Output files:

```text
outputs/model_voting_synthetic_eval/summary.json
outputs/model_voting_synthetic_eval/per_sample_votes.csv
outputs/model_voting_synthetic_eval/confusion_matrix.csv
```

Status: pass.

Note: `constant_velocity` remains the weakest model class in this small
synthetic recovery test.

## Experiment 3: Real-Window Posterior Predictive Demo

Because Experiment 2 passed after minimal debug retraining, the larger run was
executed.

### 3.1 Larger Dataset And Training

Dataset command:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\generate_model_voting_data.py `
  --real-windows data\real_football_windows.npz `
  --n-per-model 1000 `
  --T 5.0 `
  --dt 0.04 `
  --out-dir data\model_voting_dataset
```

Output:

```json
{
  "out": "data\\model_voting_dataset\\dataset.npz",
  "tracks": 4000,
  "steps": 125
}
```

Training command:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\train_model_voting_ratio.py `
  --data-dir data\model_voting_dataset `
  --epochs 100 `
  --batch-size 128 `
  --out-dir checkpoints
```

Final checkpoint result:

```text
epoch 100
train loss: 0.1555
train acc: 94.61%
train gap: 19.077
val loss: 0.2386
val acc: 91.41%
val gap: 19.071
```

Output files:

```text
checkpoints/model_voting_ratio_best.pt
checkpoints/model_voting_ratio_history.csv
```

Status: pass.

### 3.2 Real-Window Posterior Recovery

Command:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\recover_model_voting_posterior.py `
  --real-windows data\real_football_windows.npz `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --window-index 0 `
  --mcmc-steps 3000 `
  --burn-in 800 `
  --out-dir outputs\model_voting_posterior
```

Model vote weights:

```json
{
  "brownian": 0.004839778067566553,
  "constant_velocity": 0.0015068518253233902,
  "ou_target": 0.8259515688400416,
  "piecewise_velocity": 0.16770180126706846
}
```

MCMC acceptance rates:

```text
brownian: 85.47%
constant_velocity: 44.60%
ou_target: 88.53%
piecewise_velocity: 49.17%
```

Winning model:

```text
ou_target
```

Status: pass.

### 3.3 Posterior Predictive Evaluation

Command:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\evaluate_model_voting.py `
  --posterior outputs\model_voting_posterior\posterior_chains.npz `
  --n-paths 300 `
  --out-dir outputs\model_voting_evaluation
```

Summary:

```json
{
  "winning_model": "ou_target",
  "sampled_model_counts": {
    "brownian": 1,
    "constant_velocity": 1,
    "ou_target": 246,
    "piecewise_velocity": 52
  },
  "endpoint_error_m": {
    "median": 2.6735711097717285,
    "p10": 0.8913376331329346,
    "p90": 39.3418083190918
  }
}
```

Output files:

```text
outputs/model_voting_evaluation/posterior_predictive_paths.png
outputs/model_voting_evaluation/endpoint_density.png
outputs/model_voting_evaluation/model_vote_weights.png
outputs/model_voting_evaluation/winning_model_parameter_histograms.png
outputs/model_voting_evaluation/summary.json
```

PNG integrity check:

```text
posterior_predictive_paths.png: 285110 bytes, 1408 x 999
endpoint_density.png: 77592 bytes, 1562 x 999
model_vote_weights.png: 48781 bytes, 1428 x 747
winning_model_parameter_histograms.png: 35696 bytes, 1411 x 542
```

Status: pass.

## Acceptance Decision

The final decision rule was:

```text
Experiment 1 passes
Experiment 2 top1_model_accuracy > 50%
Experiment 3 produces plausible real-window posterior predictive plots
```

Observed:

```text
Experiment 1 passed.
Experiment 2 top1_model_accuracy = 57% after minimal debug retraining.
Experiment 3 generated all expected posterior and evaluation outputs.
```

Decision:

```text
The current model-voting SBI pipeline is minimally validated.
It is reasonable to proceed to the next project step, but constant_velocity
confusion should be kept in mind.
```

## Follow-Up Experiments: Large Checkpoint And Prefix/Suffix Prediction

These follow-up runs validate the larger checkpoint, check whether the
`constant_velocity` confusion remains, test several real windows, and implement
the prefix/suffix prediction protocol.

### Large Synthetic Known-Model Recovery

Command:

```powershell
D:\miniconda3\envs\dl\python.exe scripts\evaluate_model_voting_synthetic.py `
  --data-dir data\model_voting_dataset `
  --checkpoint checkpoints\model_voting_ratio_best.pt `
  --n-eval 400 `
  --n-candidates 1024 `
  --out-dir outputs\model_voting_synthetic_eval_large
```

Result:

```json
{
  "top1_model_accuracy": 0.8875,
  "top2_model_accuracy": 0.9825,
  "mean_vote_weight_for_true_model": 0.8135455801907032,
  "mean_entropy_of_model_votes": 0.31999856035022733,
  "per_model_accuracy": {
    "brownian": 0.9245283018867925,
    "constant_velocity": 0.9444444444444444,
    "ou_target": 0.8452380952380952,
    "piecewise_velocity": 0.8235294117647058
  }
}
```

Status: pass. The large checkpoint is well above the 70% "good" threshold.

### constant_velocity Confusion

Confusion matrix from `outputs/model_voting_synthetic_eval_large`:

```text
true_by_pred,brownian,constant_velocity,ou_target,piecewise_velocity
brownian,98,1,5,2
constant_velocity,1,102,1,4
ou_target,12,0,71,1
piecewise_velocity,2,13,3,84
```

`constant_velocity` is no longer weak:

```text
total constant_velocity samples: 108
wrong: 6
accuracy: 94.44%
wrong predictions:
  piecewise_velocity: 4
  ou_target: 1
  brownian: 1
```

Conclusion: the earlier `constant_velocity` failure was mainly a small
checkpoint issue. The remaining confusion is mostly with `piecewise_velocity`,
which is expected because constant velocity is a special case of piecewise
velocity.

### Multiple Real Windows

Windows evaluated:

```text
0, 50, 100, 200, 500
```

| window | winning model | main vote weights | endpoint median | endpoint p90 |
|---:|---|---|---:|---:|
| 0 | ou_target | ou_target 0.826, piecewise_velocity 0.168 | 2.67 m | 39.34 m |
| 50 | constant_velocity | constant_velocity 0.841, ou_target 0.157 | 23.35 m | 41.20 m |
| 100 | ou_target | ou_target 0.417, brownian 0.267, piecewise_velocity 0.214 | 20.59 m | 45.04 m |
| 200 | piecewise_velocity | piecewise_velocity 0.935 | 35.26 m | 72.63 m |
| 500 | piecewise_velocity | piecewise_velocity 0.638, ou_target 0.184, constant_velocity 0.172 | 26.46 m | 69.45 m |

Status: partial. Different windows choose different models, so the posterior is
not locked to one model. However, endpoint errors are often large, especially
for windows 200 and 500.

### Prefix/Suffix Prediction

Implemented scripts:

```text
scripts/extract_prefix_suffix_windows.py
scripts/recover_model_voting_prefix_posterior.py
scripts/evaluate_prefix_suffix_prediction.py
```

Window split:

```text
full window: 5 seconds
prefix: first 2 seconds = 50 steps
suffix: next 3 seconds = 75 steps
dt: 0.04
```

The prefix posterior script does not use the held-out suffix endpoint. Because
the current trained model expects a target-like condition, it supports
prefix-only target heuristics:

```text
average
recent
prefix_end
```

For the main baseline output, `prefix_end` was used because it was the best of
the three no-leakage heuristics on window 0.

Main output:

```text
outputs/prefix_suffix_posterior
outputs/prefix_suffix_prediction
```

Result on window 0:

```json
{
  "winning_model": "ou_target",
  "model_vote_weights": {
    "brownian": 0.0000023018585611747305,
    "constant_velocity": 0.00000023740362597594198,
    "ou_target": 0.9237816346572517,
    "piecewise_velocity": 0.07621582608056114
  },
  "suffix_endpoint_error_m": {
    "median": 22.69134521484375,
    "p10": 20.62403106689453,
    "p90": 35.05808639526367
  },
  "path_rmse_m": {
    "median": 14.944526672363281,
    "p10": 13.907381057739258,
    "p90": 23.685897827148438
  },
  "coverage_rate": 0.04
}
```

Target-strategy comparison on window 0:

| target strategy | endpoint median | path RMSE median | coverage |
|---|---:|---:|---:|
| recent | 27.81 m | 24.39 m | 0.00 |
| average | 39.17 m | 29.79 m | 0.00 |
| prefix_end | 22.69 m | 14.94 m | 0.04 |

Status: implemented, but not yet a good future predictor.

### Follow-Up Decision

Observed:

```text
large synthetic recovery: pass, top1 = 88.75%
constant_velocity confusion: mostly resolved
multi-window real posterior: model votes vary, endpoint errors often high
prefix/suffix prediction: implemented, but endpoint error and coverage are poor
```

Decision:

```text
Do not add new SDE models yet.
Pause on project-feature expansion and debug future-prediction conditioning.
Recommended next debugging target: retrain model-voting with prefix-compatible
conditioning, or remove/replace the full-window target condition for prediction.
```
