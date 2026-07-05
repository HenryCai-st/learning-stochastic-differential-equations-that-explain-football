# Lorenz SDE — Design Notes for Neural Network Parameter Prediction

## Overview

The goal is to train a neural network that predicts the parameters of a **Lorenz SDE** from observed trajectory images, approximating random trajectories as Lorenz SDEs. These notes cover four key design decisions in the data generation and loading pipeline.

---

## 1. Valid Parameter Ranges and Dynamical Regimes

### The Lorenz System

The standard Lorenz SDE adds diffusion to the deterministic skeleton:

$$
dx = \sigma(y - x)\,dt + \epsilon\,dW_x
$$
$$
dy = (x(\rho - z) - y)\,dt + \epsilon\,dW_y
$$
$$
dz = (xy - \beta z)\,dt + \epsilon\,dW_z
$$

with parameters $\sigma$ (Prandtl number), $\rho$ (Rayleigh number), $\beta$ (geometric factor), and $\epsilon$ (noise scale).

### Dynamical Regimes

The regime is primarily governed by $\rho$, with $\sigma$ and $\beta$ playing secondary roles:

| Regime | Typical $\rho$ | Behaviour |
|--------|---------------|-----------|
| **Fixed point** | $\rho < 1$ | All trajectories decay to origin |
| **Stable fixed points** | $1 < \rho \lesssim 13.9$ | Trajectories settle to one of two symmetric attractors $C^\pm$ |
| **Limit cycle / transient chaos** | $13.9 \lesssim \rho \lesssim 24.1$ | Complex transients; some parameter pockets show periodic orbits |
| **Chaos** | $\rho \gtrsim 24.74$ | Classic butterfly attractor; the well-known strange attractor appears near $\rho \approx 28$ |

> **Note:** The boundary $\rho \approx 24.74$ is the subcritical Hopf bifurcation point for the canonical $\sigma = 10$, $\beta = 8/3$. If you vary $\sigma$ or $\beta$, these boundaries shift.

### Parameter Ratios and Invariant Structure

Two ratios are particularly important:

- $\sigma / \beta$ influences the aspect ratio and "spread" of the attractor lobes. Trajectories with the same $\sigma/\beta$ ratio but different absolute values can look qualitatively similar after rescaling.
- The **Rayleigh–Bénard ratio** $r = \rho / \rho_c$ (where $\rho_c = 1$ is the trivial bifurcation point) is sometimes used to normalise across parameter families.

### Suggested Sampling Ranges

```
σ  ∈ [1,  20]   (canonical: 10)
ρ  ∈ [0.5, 50]  (covers all three regimes)
β  ∈ [0.5,  5]  (canonical: 8/3 ≈ 2.67)
ε  ∈ [0,    2]  (0 = deterministic; >1 = heavily noise-dominated)
```

**Feedback / Recommendations:**

- Avoid sampling $\rho$ uniformly — the chaotic regime occupies most of $[0, 50]$ by length but you likely want balanced regime representation. Use **stratified sampling** (e.g. equal numbers from each regime band).
- Very large noise $\epsilon \gg 1$ destroys all attractor structure. Consider capping at $\epsilon \approx 1.5$ unless noise-dominated trajectories are part of the task.
- Near the bifurcation boundaries (e.g. $\rho \approx 24.74$) trajectories can be ambiguous — consider a small exclusion margin (e.g. $|\rho - 24.74| > 0.5$) to prevent label ambiguity.
- Log-spacing $\rho$ makes sense if you want to resolve the fixed-point regime well (it occupies $\rho \in (0,1)$, which is tiny on a linear scale).

---

## 2. Image Generation: Fixed vs. Scaled Axes

### The Choice

| Approach | Fixed axes | Auto-scaled axes |
|----------|-----------|-----------------|
| **What is preserved** | Absolute trajectory magnitude | Shape / topology |
| **What is lost** | Pattern clarity for small attractors | Size information |
| **Risk** | Fixed-point and small-$\rho$ trajectories look tiny | Network may confuse scale-related amplitude with regime |

Since the stated goal is **approximating patterns, not sizes**, auto-scaling is generally preferable. However, there are nuances:

### Recommendations

- **Use per-trajectory min-max scaling on all three axes independently** before rendering. This fills the image canvas and makes patterns directly comparable.
- Alternatively, **scale all axes by the same factor** (uniform scaling) to preserve the 3D aspect ratio of the attractor — this retains lobe asymmetry information that per-axis scaling destroys.
- For 2D projections (XY, XZ, YZ): render all three projections as separate channels or as a 3-channel image. The XZ projection is most diagnostic for distinguishing regimes.
- Consider rendering **the trajectory as a density map** (2D histogram / kernel density) rather than a line plot. Density maps are more robust to integration step size and trajectory length, and give a stable image even for noisy SDEs.
- Use a fixed **image resolution** (e.g. 64×64 or 128×128) and a **fixed simulation time** (e.g. $T = 50$) with a standard step size (e.g. $\Delta t = 0.01$) so the network sees a consistent input distribution. Discard the initial transient (e.g. first 10 time units).

---

## 3. DataLoader: Normalisation Strategies

### Parameters

| Parameter | Suggested scaling | Rationale |
|-----------|-----------------|-----------|
| $\sigma$ | **Min-max** to $[0,1]$ | Roughly linear effect; no heavy tail |
| $\rho$ | **Log then min-max** or **Z-score on log** | Large dynamic range; regime boundaries are log-spaced |
| $\beta$ | **Min-max** to $[0,1]$ | Narrow range, well-behaved |
| $\epsilon$ | **Log then min-max** | Noise scale spans orders of magnitude; log-transform prevents small $\epsilon$ from being compressed near zero |

**On log-transforming $\epsilon$:**

```python
# Add a small offset to handle ε = 0 (deterministic)
eps_log = np.log1p(epsilon)   # log(1 + ε), maps 0 → 0 smoothly
# Then min-max normalise eps_log to [0, 1]
```

`log1p` is preferable to `log(ε)` because it handles the deterministic case $\epsilon = 0$ without a singularity.

**On Z-score vs. Min-max:**

- **Min-max** is simpler and guarantees output in $[0,1]$, but is sensitive to outliers. If your sampling bounds are well-defined (as above), min-max is fine.
- **Z-score** is better if the network needs to generalise beyond the training range (e.g. during inference on out-of-distribution inputs). It does not clip to a fixed range.
- A practical middle ground: **clip to the 1st–99th percentile** of your training distribution, then min-max normalise. This avoids outlier sensitivity without losing boundary information.

### Images

- Convert to **greyscale** (single channel) for trajectory density maps; or keep **3-channel** if using RGB projections.
- Map pixel intensities from $[0, 255]$ (or $[0, 1]$) to **$[-1, 1]$**:

```python
image_normalised = image / 127.5 - 1.0   # from uint8 [0,255] to [-1,1]
# or equivalently for float [0,1]:
image_normalised = image * 2.0 - 1.0
```

This is the standard convention for GAN-style generators and works well with tanh output activations. It also centres the distribution near zero, which benefits batch normalisation layers.

---

## 4. Labelling and Regime Selection for Training

### Should You Use 2 or 3 Regimes?

This is the most consequential design decision, and there is a real argument for **training on only 2 regimes** (e.g. fixed-point and chaotic), at least initially:

| Strategy | Pros | Cons |
|----------|------|------|
| **3 regimes** | Full coverage; network learns all dynamics | Limit cycle regime is poorly defined; ambiguous labels near boundaries |
| **2 regimes (fixed + chaos)** | Cleaner decision boundary; easier to validate | Ignores the transient/limit-cycle regime entirely |
| **2 regimes (fixed + non-fixed)** | Simple binary task; maximum label confidence | Chaotic and limit-cycle lumped together |

**Recommendation:** Start with **2 regimes** — fixed-point vs. chaotic — with a strict exclusion zone around the limit-cycle region ($13.9 < \rho < 24.74$ for canonical parameters). This gives clean labels and avoids the hardest boundary cases. The network can always be extended to 3 classes once the 2-class task is validated.

### Labelling from Parameters Alone

Since you know the parameters at generation time, you can assign labels **deterministically** without simulating:

```python
def label_regime(rho, sigma=10.0, beta=8/3):
    """
    Label Lorenz regime from parameters only.
    Uses canonical bifurcation points for σ=10, β=8/3.
    Scale thresholds if σ, β differ.
    """
    rho_1  = 1.0       # trivial bifurcation: fixed point above this
    rho_lc = 13.926    # onset of limit cycle / transient chaos
    rho_c  = 24.737    # onset of strange attractor

    if rho < rho_1:
        return "fixed_origin"
    elif rho < rho_lc:
        return "fixed_point"   # two stable fixed points C±
    elif rho < rho_c:
        return "limit_cycle"   # ambiguous / transient
    else:
        return "chaos"
```

For variable $\sigma$ and $\beta$, the critical $\rho$ values shift. The onset of the strange attractor can be approximated as:

$$
\rho_c \approx \sigma \cdot \frac{\sigma + \beta + 3}{\sigma - \beta - 1}
$$

(valid for $\sigma > \beta + 1$). Compute this per-sample to get a parameter-adaptive label.

### Practical Label Exclusion Zone

```python
def assign_label_2class(rho, sigma, beta, margin=0.5):
    rho_c = sigma * (sigma + beta + 3) / (sigma - beta - 1)
    rho_1 = 1.0

    if rho < rho_1 + margin:
        return None  # exclude near-origin boundary
    elif rho < 13.926 - margin:
        return 0     # fixed point
    elif rho > rho_c + margin:
        return 1     # chaos
    else:
        return None  # exclude ambiguous middle region
```

Dropping unlabelled samples gives you a high-confidence binary dataset without any simulation-based labelling.

---

## Summary Table

| Design decision | Recommended choice |
|-----------------|-------------------|
| $\rho$ sampling | Stratified + log-spaced within each regime |
| Axis scaling | Per-trajectory uniform scale (preserve 3D aspect ratio) |
| Image type | 3-projection density map, 64×64 or 128×128 |
| $\sigma$, $\beta$ normalisation | Min-max $[0,1]$ |
| $\rho$ normalisation | Log → min-max |
| $\epsilon$ normalisation | `log1p` → min-max |
| Image normalisation | $[-1, 1]$ via `img * 2 - 1` |
| Number of regimes | **2** (fixed-point vs. chaos), exclude limit-cycle band |
| Label source | Parameters only, using analytical bifurcation formula |
