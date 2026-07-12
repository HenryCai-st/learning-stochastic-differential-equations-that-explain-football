# Controlled Method Validation Results

## Scope

This report records model recovery, known-model parameter recovery, and
posterior predictive forecasting. No football tracking data is used.

## Protocol

- Four balanced simulators: Brownian, constant velocity, OU-to-target, and
  three-segment piecewise velocity.
- Independent split seeds: train `20260712`, validation `20260713`, test
  `20260714`.
- Split sizes: 4,000 train, 400 validation, and 400 test trajectories.
- Each trajectory has 50 two-dimensional positions (`T=2.0`, `dt=0.04`).
- Starts are sampled centrally; targets use random directions and requested
  distances of 8-25 metres; change points are fixed at one and two thirds.
- Training: 100 epochs, batch size 128, Adam learning rate `1e-3`.
- Evidence: 512 independent prior samples per candidate and test case.

## Training Result

The best checkpoint, selected only by the independent validation split, was
epoch 80 with validation loss `0.085429`. Final-epoch matched-versus-mismatched
classification accuracy stayed near 95-97%; this is a training diagnostic, not
the model-recovery result.

## Independent Test Result

Top-1 model recovery was **89.75%** on all 400 test trajectories. Mean weight
on the true model was `0.8595`; mean true-model log score was `-0.2816`.

| True model | Accuracy | Brownian | Constant velocity | OU target | Piecewise velocity |
|---|---:|---:|---:|---:|---:|
| Brownian | 98% | 98 | 0 | 1 | 1 |
| Constant velocity | 88% | 3 | 88 | 3 | 6 |
| OU target | 84% | 16 | 0 | 84 | 0 |
| Piecewise velocity | 89% | 0 | 11 | 0 | 89 |

Rows are true models and count 100 cases each. The main ambiguity is weak OU
attraction being confused with Brownian motion, followed by piecewise tracks
that resemble one constant velocity over the short horizon.

Repeating only the 512-sample evidence integration with evaluation seeds
`20260715`, `20260716`, and `20260717` produced accuracies of 89.75%, 90.25%,
and 91.00%. The mean was 90.33% and the range was 1.25 percentage points.
Across those runs, mean true-model weight ranged from 0.8568 to 0.8625 and
mean log score from -0.2855 to -0.2748. The conclusion is stable to this
limited Monte Carlo repetition; training-seed sensitivity remains untested.

## Known-Model Parameter Recovery

Parameter recovery used 25 held-out cases per model, four MH chains per case,
2,400 steps, 800 burn-in steps, and 64 prior candidates for initialization.
The model family was fixed to the known truth so model-selection errors could
not contaminate parameter diagnostics.

Aggregate 50/80/90% interval coverage was `51.1% / 76.9% / 85.5%`. Coverage
alone is not sufficient: only 42.2% of case-parameter combinations achieved
split R-hat below 1.05. Brownian was close to converged (96% R-hat pass rate;
median ESS 529), while piecewise velocity had a median ESS of 37 and no
parameter reached an acceptable aggregate R-hat pass rate.

| Model | Mean absolute error | 50% | 80% | 90% | Median ESS | Max R-hat |
|---|---:|---:|---:|---:|---:|---:|
| Brownian | 0.143 | 56.0% | 88.0% | 88.0% | 529 | 1.066 |
| Constant velocity | 1.975 | 42.7% | 74.7% | 82.7% | 198 | 1.602 |
| OU target | 0.202 | 62.0% | 82.0% | 86.0% | 162 | 1.151 |
| Piecewise velocity | 7.998 | 50.9% | 74.9% | 86.3% | 37 | 1.896 |

The diagnostic status is therefore **convergence not established**. In
particular, the current result does not support a strong claim that all seven
piecewise parameters are reliably recovered.

## Controlled Forecast Result

The forecast benchmark extends the independent two-second prefixes by one
held-out second under the same dynamics. Piecewise futures explicitly continue
the latest observed segment and introduce no unseen future turn. The formal
evaluation used 25 cases per model, 1,024 prior evidence samples per candidate,
and 256 aligned posterior predictive paths per case.

| Method | Mean ADE | Median ADE | Mean FDE | Median FDE |
|---|---:|---:|---:|---:|
| SBI model voting | 2.642 m | 1.209 m | 4.534 m | 1.463 m |
| Stationary | 3.924 m | 2.278 m | 6.990 m | 3.402 m |
| Last velocity | **1.528 m** | **0.672 m** | **2.773 m** | **1.130 m** |
| Damped velocity | 1.951 m | 1.681 m | 3.882 m | 2.708 m |

SBI recovered the correct model in 88% of these 100 forecast cases and beat
last velocity on ADE in 48% of individual cases. It was strongest relative to
the baseline on Brownian and OU cases, but piecewise mean ADE was 7.60 m versus
1.74 m for last velocity. This is consistent with the weak piecewise parameter
diagnostics above. All methods use the same pitch-boundary clipping.

Predictive time-point coverage for nominal 50/80/90% radial regions was
`58.9% / 82.2% / 89.4%`; endpoint coverage was `64% / 84% / 90%`. The
uncertainty regions are reasonably calibrated under this controlled policy,
but their mean error does not beat the strongest simple baseline overall.

## Reproducible Artifacts

Generated data and large results are intentionally Git-ignored:

```text
data/method_validation/{train,validation,test}.npz
checkpoints/method_validation/ratio_estimator_best.pt
checkpoints/method_validation/ratio_estimator_history.csv
outputs/method_validation/model_recovery/summary.json
outputs/method_validation/model_recovery/confusion_matrix.png
outputs/method_validation/model_recovery/cases.npz
outputs/method_validation/model_recovery_seed_*/summary.json
data/method_validation/forecast_test.npz
outputs/method_validation/parameter_recovery/{summary.json,case_parameter_metrics.csv,posterior_samples.npz}
outputs/method_validation/forecast_evaluation/{summary.json,case_metrics.csv,posterior_predictive_samples.npz}
```

Each stage writes metadata with arguments, seeds, Git state, runtime versions,
simulator priors, and trajectory contracts.

## Interpretation And Boundary

The pipeline recovers model families and produces reasonably calibrated
predictive regions under unseen, football-independent conditions. It does not
yet establish reliable high-dimensional parameter convergence, does not beat
last velocity on aggregate forecast error, and says nothing about validity on
football data. Improving piecewise parameter inference is the next Part I
priority.
