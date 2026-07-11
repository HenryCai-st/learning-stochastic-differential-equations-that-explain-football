# Source Module Status

The active football-ball model-voting pipeline uses only the top-level
`data`, `models`, `sde`, and `utils` packages. Historical code is isolated in
`legacy` and is not imported by the active workflow.

## Active

| Path | Purpose |
|---|---|
| `data/football_tracking.py` | Parse Metrica tracking data and extract windows. |
| `data/trajectory_features.py` | Velocity, speed, heading, acceleration, diagnostics. |
| `data/segmentation.py` | Detect piecewise trajectory change points. |
| `data/model_voting_dataset.py` | PyTorch dataset for mixed-model SBI training. |
| `models/encoder.py` | Shared 1D trajectory encoder. |
| `models/model_voting_ratio.py` | Active neural ratio classifier. |
| `sde/model_voting.py` | Candidate definitions, priors, and simulators. |
| `sde/football_ou.py` | Active OU candidate simulator. |
| `utils/football_viz.py` | Pitch and tracking visualization helpers. |

## Legacy

| Path | Purpose |
|---|---|
| `legacy/lorenz/` | Lorenz demonstration loaders, models, losses, and SDE. |
| `legacy/ou/football_dataset.py` | Dataset adapters for the standalone OU baseline. |
| `legacy/inference/mcmc.py` | Unused historical inference prototype. |

Legacy modules are retained for reference and archived workflows. They should
not be used when extending the active model-voting pipeline.
