# Script Groups

Run every command from the repository root.

## Controlled Method Validation

| Script | Purpose |
|---|---|
| `method_validation/generate_synthetic_benchmark.py` | Generate independent football-free train, validation, and test splits. |
| `method_validation/train_ratio_estimator.py` | Train with an explicit independent validation artifact. |
| `method_validation/evaluate_synthetic_model_recovery.py` | Evaluate the held-out test split with prior-integrated evidence. |
| `method_validation/evaluate_synthetic_parameter_recovery.py` | Run known-model multi-chain parameter recovery and diagnostics. |
| `method_validation/generate_synthetic_forecast_benchmark.py` | Extend test prefixes with controlled held-out futures. |
| `method_validation/evaluate_synthetic_forecasts.py` | Compare posterior predictions with three simple baselines. |

The controlled pipeline through parameter recovery and forecasting is
implemented. Current results identify piecewise parameter inference as the
main Part I limitation.

## Football Case Study

| Order | Script | Required output |
|---:|---|---|
| 1 | `football_case_study/extract_football_windows.py` | `data/real_football_windows.npz` |
| 2 | `football_case_study/generate_model_voting_data.py` | `data/model_voting_dataset/dataset.npz` |
| 3 | `football_case_study/train_model_voting_ratio.py` | `checkpoints/model_voting_ratio_best.pt` |
| 4 | `football_case_study/recover_model_voting_posterior.py` | MCMC chains and approximate model weights |
| 5 | `football_case_study/evaluate_model_voting.py` | Future-path figures and forecast metrics |

## Complementary Tools

| Script | Purpose |
|---|---|
| `tools/plot_model_voting_dataset.py` | Inspect synthetic track and prior diversity. |
| `tools/plot_real_window_segments.py` | Inspect detected change points. |
| `tools/football_tracking_viz.py` | Render one raw tracking frame. |
| `tools/football_window_clip.py` | Create a raw tracking-window animation. |
| `tools/football_model_voting_clip.py` | Create a sliding classifier-score animation. |
| `tools/synthetic_forecast_validation_animation.py` | Compare four conditional model forecasts against one synchronized held-out trajectory. |

## Historical Workflows

- `OU_workflow/`: archived standalone OU baseline;
- `Lorenz_workflow/`: archived Lorenz demonstration.
