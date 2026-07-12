# Project Restructure And Roadmap

## Purpose

The project is split into two parts with different claims:

1. **Controlled method validation** is the main contribution. It validates
   model selection, parameter inference, and probabilistic forecasting when
   the simulator and true parameters are known.
2. **Football case study** applies the same tools to Metrica ball tracking and
   documents data quality, simulator mismatch, unobserved events, and other
   practical obstacles.

Synthetic success must not be presented as proof of real-football validity.
Football limitations must not obscure whether the SBI implementation works in
a controlled setting.

## Current Structure

```text
src/
  sbi/                 shared encoder, ratio model, scoring, evidence, MCMC,
                       and artifact contracts
  simulators/          candidate motion models and parameter priors
  synthetic/           synthetic dataset adapters
  football/            tracking, features, segmentation, and visualization
  legacy/              archived Lorenz, OU, and inference prototypes

scripts/
  method_validation/   controlled synthetic evaluation commands
  football_case_study/ football extraction, training, posterior, and forecast
  tools/               optional visualizations and clips
  OU_workflow/          archived OU baseline
  Lorenz_workflow/     archived Lorenz demonstration
```

The active scripts remain thin orchestration layers. Shared numerical logic
must live under `src/`, not be imported from another CLI script.

## Completed Structural Changes

- Moved ratio estimation and the trajectory encoder to `src/sbi/`.
- Moved candidate motion models and priors to `src/simulators/`.
- Moved synthetic dataset loading to `src/synthetic/`.
- Moved football-specific tracking and visualization to `src/football/`.
- Extracted checkpoint scoring, evidence integration, and MCMC from the
  football posterior CLI into shared SBI modules.
- Split active scripts into `method_validation/` and `football_case_study/`.
- Kept archived workflows separate under `legacy`, `OU_workflow`, and
  `Lorenz_workflow`.

These changes preserve the existing simulation, training, inference, and
evaluation calculations. They change ownership and import paths only.

## Artifact And Run Metadata

Every newly generated active artifact records schema version 1 metadata.

Window and synthetic dataset NPZ files include `artifact_metadata_json` with:

- generating Git commit and UTC timestamp;
- command arguments and input files;
- `steps`, `dt`, duration, channels, and prefix/suffix contract;
- model names, parameter dimensions, prior bounds, and log-scale flags;
- condition source and random seed where applicable.

Ratio-classifier checkpoints include `artifact_metadata` with the same data
contract, simulator contract, training arguments, and runtime versions.

Each stage also writes a `run_metadata.json` file, or an adjacent `.run.json`
for a single NPZ output. Posterior recovery and synthetic evaluation reject a
checkpoint when `steps`, `dt`, or model names differ from the input artifact.

## Part I: Controlled Method Validation

This is the primary experimental result. It must not read football tracking
windows when generating conditions.

Planned flow:

```text
independent synthetic train/validation/test conditions
-> ratio-estimator training
-> fresh model-recovery evaluation
-> known-theta MCMC recovery
-> synthetic prefix/suffix forecast
-> calibration and simple-baseline comparison
```

Required outputs:

- true-model versus selected-model confusion matrix;
- model log score and evidence stability across seeds;
- parameter bias, interval width, ESS, and 50/80/90% coverage;
- ADE, FDE, predictive coverage, and proper scoring rules;
- stationary, last-velocity, and damped-velocity baselines.

## Part II: Football Case Study

The football case study reuses the validated SBI core but makes weaker claims.

Planned flow:

```text
raw Metrica tracking
-> continuity and quality filtering
-> observed prefix / held-out suffix windows
-> football-adapted synthetic conditions
-> posterior recovery and forecasting
-> multi-window baseline comparison and failure analysis
```

Required analysis:

- results from multiple games and periods;
- ADE, FDE, and coverage by speed and turn intensity;
- tracking gaps and frame discontinuities;
- prior sensitivity and simulator-to-reality mismatch;
- unobserved passes, contacts, possession changes, and future turns.

## Next Implementation Order

1. Fix posterior-path/model-ID alignment in forecast outputs.
2. Enforce period, frame, and time continuity during football window extraction.
3. Record missing-data diagnostics before interpolation.
4. Add a football-independent condition generator for Part I.
5. Split conditions into independent train, validation, and test artifacts.
6. Add known-theta posterior recovery and convergence summaries.
7. Add synthetic forecast baselines and aggregate calibration.
8. Run the football case study only after the controlled benchmark is fixed.

Automated test coverage is not a project milestone. Small contract and smoke
checks are useful, but scientific validation and artifact consistency have
priority.
