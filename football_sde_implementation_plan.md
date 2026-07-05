# Implementation Plan: SBI on Football Tracking Data (2D OU Process)

Goal: reuse the existing ratio-estimation + MCMC pipeline, but swap the Lorenz
SDE for a 2D Ornstein-Uhlenbeck (OU) target-seeking process, and swap synthetic
inputs for real tracking data extracted from `football_tracking_viz.py`'s CSVs.

Recommended order: **start with the position-only OU model** (Phase A) to
validate the whole retrained pipeline end-to-end before adding velocity/
momentum (Phase B).

---

## 0. Model definition

### Phase A — position-only OU (start here)

State: `(x, y)`. Known conditioning inputs: `y0 = (x0, y0)` (observed start),
`target = (tx, ty)` (observed end point of the window, or ball position).

```
dx = k * (tx - x) dt + noise_scale * dWx
dy = k * (ty - y) dt + noise_scale * dWy
```

`theta = (k, noise_scale)` — 2 free parameters.

### Phase B — velocity/momentum OU (upgrade later)

State: `(x, y, vx, vy)`. Same known conditioning inputs.

```
dvx = [k*(tx - x) - c*vx] dt + noise_scale * dWx
dvy = [k*(ty - y) - c*vy] dt + noise_scale * dWy
dx  = vx dt
dy  = vy dt
```

`theta = (k, c, noise_scale)` — 3 free parameters. Requires observed initial
velocity too (finite-difference the first 2-3 real frames).

---

## 1. Data extraction from real tracking CSVs

Reuse `load_tracking()` from `football_tracking_viz.py` as-is — it already
parses the 3-row header and denormalizes to metres via `denormalize()`.

New steps needed:

1. **Choose an entity to track** — a single player column (e.g. `Player7_x/_y`)
   or the ball (`Ball_x/_y`). Start with the ball; it's least noisy and has
   only 2D position, no team-formation confounds.
2. **Slice a window**: pick a `start_frame`, take `T` seconds at `dt=0.04s`
   → `n_steps = int(T/dt)` frames (e.g. T=5s → 125 frames). No interpolation.
3. **Denormalize** to metres using `denormalize()` (already handles the
   [0,1] → pitch-metres conversion).
4. **Extract conditioning inputs**:
   - `y0 = (x[0], y[0])` from the first frame in the window.
   - `target = (x[-1], y[-1])` from the last frame in the window (Phase A),
     or ball position at window end if modeling a player being drawn to the
     ball.
   - (Phase B only) `v0 ≈ (x[1]-x[0], y[1]-y[0]) / dt` — finite difference of
     first two frames.
5. Handle missing/NaN frames (players can drop out of tracking) — either drop
   windows with gaps, or forward-fill short gaps before extraction. Don't
   silently interpolate across large gaps; large gaps are lost information,
   not lost samples.
6. Save extracted real windows to a separate `.npz`
   (`real_tracks.npz`: array of `(track, y0, target, meta)`) — this becomes
   your **evaluation set**, not training data. You still train on synthetic
   data from the OU simulator (see below); real tracks are what you run
   inference on at the end.

---

## 2. `generate_data.py` changes

- **New simulator function** `simulate_ou_batch(k, noise_scale, y0_batch,
  target_batch, t_grid, rng, clip_value)` replacing `simulate_batch`'s Lorenz
  update with the OU update above. Vectorize the same way (batch dimension
  first).
- **`sample_parameters`**: replace 4D Lorenz prior with 2D OU prior, e.g.
  - `k ~ Uniform(low, high)` — pick range so `1/k` (relaxation timescale)
    spans from "snaps to target within a frame or two" to "barely moves
    toward target over the whole window." Calibrate against real average
    speeds you computed while reading the data.
  - `noise_scale` — log-uniform, similarly calibrated against observed
    frame-to-frame jitter in real tracks.
  - Drop the `rho`-style stratified sampling entirely — no chaotic regimes
    here.
- **`y0` and `target` are no longer fixed constants** — for each synthetic
  sample, draw `y0` and `target` from a distribution matching real pitch
  positions/displacements you observed (e.g. uniform over pitch bounds, or
  bootstrap actual observed `(y0, target)` pairs from the real dataset so
  synthetic conditioning matches the real distribution). Store both alongside
  `theta`, `tracks`, `group_ids` in the saved `.npz` — the encoder and MCMC
  routine will need them as inputs, not just the track.
- **`dt`**: set to `0.04` to match the sensor frame rate; `T` matches your
  chosen window (e.g. `5.0`). If Euler-Maruyama is unstable at this coarse a
  step for your chosen `k` range, simulate at a finer internal `dt_sub`
  (e.g. `0.01`) and subsample every 4th step before saving — so the saved
  arrays are always at 0.04s spacing, matching real data exactly.
- Drop `PARAMETER_LOG_SCALE`/`RHO_REGIME_BOUNDS`-style Lorenz-specific
  metadata; add `PITCH_LENGTH`/`PITCH_WIDTH` metadata instead if useful for
  bounds-checking.

---

## 3. Dataset / encoder changes (`src/data/dataset.py`, `src/models/encoder.py`)

These weren't uploaded, but based on how they're used:

- `SDEDataset` currently returns `(query_t, _, _, params_t)` per item and
  exposes `track_mean`/`track_std` and a `normalizer` for `theta`. It will
  need to also return `(y0, target)` per item (or a combined conditioning
  vector) since these are now required inputs to both the encoder and the
  simulator-based MCMC step — plan on extending the tuple/dict returned by
  `__getitem__` rather than hiding this info only in the track itself.
- `TrajectoryEncoder` takes `(B, 3, steps)` for Lorenz (3 spatial dims). For
  2D tracks, this becomes `(B, 2, steps)` — check if the first conv/linear
  layer hardcodes `in_channels=3` anywhere; if so, parametrize it.
- Decide how `y0`/`target` enter the classifier: simplest is to concatenate
  them (after normalizing, e.g. by pitch length/width) to the track-encoder
  output before the classifier head, alongside the existing `theta` embedding
  — i.e. `RatioClassifier.forward` becomes
  `forward(tracks, params, y0, target)`, concatenating
  `[z_track, z_theta, y0_norm, target_norm]` before the final MLP. Without
  this, the classifier has no way to know what "correct" dynamics look like
  for a given start/target pair, since the same `theta` produces very
  different absolute tracks depending on `y0`/`target`.

---

## 4. `train_ratio_classifier.py` changes

- Update `RatioClassifier.__init__` for `param_dim=2` (Phase A: `k`,
  `noise_scale`) and pass `y0`/`target` through `forward` as above.
- `make_negative_params` (roll-shuffle within batch) still works unchanged —
  it's shuffling `theta`, and pairing a track with the wrong `theta` while
  keeping its own true `y0`/`target` is still a valid "mismatched" example
  (arguably a *harder*, more informative negative than Lorenz's case, since
  `y0`/`target` provide extra context the classifier should learn to use).
- Everything else (BCE loss, train/val split, checkpointing) is
  architecture-agnostic and needs no changes.

---

## 5. `recover_posterior.py` changes

- `log_prior_physical`: rewrite for the 2D OU prior (uniform `k`, log-uniform
  `noise_scale`) — much simpler than the Lorenz version, no regime-width
  logic needed.
- `random_walk_metropolis_hastings`: needs `y0` and `target` threaded through
  to `log_ratio_for_theta` → `model(track_t, theta_t, y0_t, target_t)`.
  Everything else (propose → accept/reject loop) is unchanged.
- `proposal_scale` now has 2 components instead of 4 — retune based on the
  new parameter ranges (check acceptance rate lands roughly in the 20-50%
  ballpark as before).
- For real-data inference: load a window from your `real_tracks.npz`
  (Section 1), extract its `track`, `y0`, `target`, and run MH directly
  against it — this is the actual "predict SDE parameters for this observed
  football trajectory" step.

---

## 6. `evaluate.py` changes

- `sample_ground_truth`: for validating on synthetic data, same idea —
  sample a `theta`, and now also sample or fix a `y0`/`target`, simulate.
- For **real-data evaluation** (the actual goal), replace this function
  entirely with "load a real extracted window + its `y0`/`target`" instead
  of simulating a synthetic ground truth — you won't have a true `theta` to
  compare against, so the ground-truth comparison panels
  (`plot_prior_posterior_histograms`'s black GT line, `build_predictive_figure`'s
  bold orange line) become optional/absent for real data. Keep them for
  synthetic validation runs, drop the GT line for real-data runs (or replace
  with "observed track" instead of "ground truth track" in the 3D predictive
  plot — the bold line still makes sense as "the actual observed trajectory,"
  just without an associated true `theta`).
- `simulate_ensemble`: pass the same real `y0`/`target` into every posterior
  sample's simulation, not a fixed constant — this is the key change so that
  predictive tracks are compared against the real trajectory under the same
  starting/target conditions.
- Posterior predictive check remains the main validation tool for real data
  even without ground-truth `theta`: if posterior-sampled tracks visually
  cluster tightly around the real observed trajectory, that's evidence the
  recovered `(k, noise_scale)` are meaningful.

---

## 7. Suggested validation order

1. Implement Phase A OU simulator + 2-param prior in `generate_data.py`.
   Generate synthetic dataset at `dt=0.04, T=5.0`.
2. Update encoder to accept 2D tracks + `y0`/`target` conditioning; retrain
   ratio classifier; confirm val accuracy/`log_ratio_gap` look healthy
   (same diagnostics already printed each epoch).
3. Run `recover_posterior.py` on held-out **synthetic** validation items
   first (you have ground truth there) — confirm MAE is low and acceptance
   rate is reasonable before touching real data at all.
4. Extract a handful of real windows (Section 1) — start with the ball,
   since it's cleanest.
5. Run MCMC on real windows; inspect posterior predictive plots
   (`evaluate.py`, adapted per Section 6) for visual match to the observed
   track, since there's no ground-truth `theta` to MAE against.
6. Once Phase A works end-to-end, add velocity/damping (Phase B) by
   extending `theta` to 3 dims and adding `v0` as a third conditioning input.
