# Script Groups

Run every command from the repository root.

## Required Model-Voting Pipeline

These scripts form the active end-to-end workflow and should normally be run
in this order:

| Order | Script | Required output |
|---:|---|---|
| 1 | `model_voting_pipeline/extract_football_windows.py` | `data/real_football_windows.npz` |
| 2 | `model_voting_pipeline/generate_model_voting_data.py` | `data/model_voting_dataset/dataset.npz` |
| 3 | `model_voting_pipeline/train_model_voting_ratio.py` | `checkpoints/model_voting_ratio_best.pt` |
| 4 | `model_voting_pipeline/evaluate_synthetic_model_recovery.py` | Synthetic model-recovery metrics |
| 5 | `model_voting_pipeline/recover_model_voting_posterior.py` | MCMC chains and approximate model weights |
| 6 | `model_voting_pipeline/evaluate_model_voting.py` | Future-path figures and forecast metrics |

Step 4 validates the learned model-selection rule and should be completed
before interpreting real-data model weights.

## Complementary Tools

These scripts improve understanding and visualization but do not train the
classifier or recover the posterior:

| Script | Purpose |
|---|---|
| `tools/plot_model_voting_dataset.py` | Plot model, parameter, condition, and prior-predictive diversity. |
| `tools/plot_real_window_segments.py` | Inspect detected change points. |
| `tools/football_tracking_viz.py` | Render one raw tracking frame. |
| `tools/football_window_clip.py` | Create a raw tracking-window animation. |
| `tools/football_model_voting_clip.py` | Create a sliding classifier-score animation. |

## Historical Workflows

- `OU_workflow/`: archived standalone OU baseline;
- `Lorenz_workflow/`: archived Lorenz demonstration.

Historical workflows are not part of the active football-ball model-voting run.
