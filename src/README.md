# Source Module Status

## Active Shared SBI

| Path | Purpose |
|---|---|
| `sbi/encoder.py` | Shared trajectory encoder. |
| `sbi/ratio_model.py` | Neural ratio classifier. |
| `sbi/scoring.py` | Checkpoint loading, normalization, and candidate scoring. |
| `sbi/evidence.py` | Prior-integrated evidence helpers. |
| `sbi/mcmc.py` | Model priors and random-walk MCMC. |
| `sbi/artifacts.py` | Dataset, checkpoint, and run metadata contracts. |
| `sbi/training.py` | Shared contrastive training and validation loops. |
| `sbi/diagnostics.py` | Multi-chain R-hat, ESS, and interval diagnostics. |
| `sbi/forecasting.py` | Controlled future simulation, baselines, and metrics. |

## Active Domain Modules

| Path | Purpose |
|---|---|
| `simulators/model_voting.py` | Candidate definitions, priors, and simulators. |
| `simulators/ou.py` | Shared OU candidate implementation. |
| `synthetic/dataset.py` | Mixed-model synthetic training dataset. |
| `synthetic/conditions.py` | Football-independent controlled conditions. |
| `football/tracking.py` | Parse Metrica tracking and extract windows. |
| `football/features.py` | Position-derived features and diagnostics. |
| `football/segmentation.py` | Piecewise trajectory segmentation. |
| `football/visualization.py` | Pitch and tracking visualization helpers. |

## Legacy

Historical Lorenz, OU, and inference prototypes live under `legacy/`. They are
retained for reference and are not dependencies of the active SBI workflow.
