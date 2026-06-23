from matplotlib import pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================

NUM_SAMPLES = 1000 #1000
T = 50
dt = 0.01
CUT = 1000

# ============================================================
# PARAM SAMPLING
# ============================================================

def sample_params():
    sigma = np.random.uniform(1, 20)
    beta  = np.random.uniform(0.5, 5)

    if np.random.rand() < 0.5:
        rho = np.random.uniform(0.5, 1.0)   # fixed
    else:
        rho = np.random.uniform(25, 50)     # chaos

    epsilon = np.random.uniform(0.0, 1.5)

    return sigma, rho, beta, epsilon

# ============================================================
# SIMULATION
# ============================================================

def simulate_lorenz_sde(sigma, rho, beta, epsilon, T=50, dt=0.01):
    N = int(T / dt)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    x[0], y[0], z[0] = np.random.randn(3)

    for i in range(N - 1):
        dx = sigma * (y[i] - x[i]) * dt + epsilon * np.sqrt(dt) * np.random.randn()
        dy = (x[i] * (rho - z[i]) - y[i]) * dt + epsilon * np.sqrt(dt) * np.random.randn()
        dz = (x[i] * y[i] - beta * z[i]) * dt + epsilon * np.sqrt(dt) * np.random.randn()

        x[i+1] = x[i] + dx
        y[i+1] = y[i] + dy
        z[i+1] = z[i] + dz

    return x, y, z

# ============================================================
# PROCESSING
# ============================================================

def remove_transient(x, y, z, cut):
    return x[cut:], y[cut:], z[cut:]

def normalize_traj(traj):
    min_vals = traj.min(axis=0)
    max_vals = traj.max(axis=0)
    return (traj - min_vals) / (max_vals - min_vals + 1e-8)

def label_from_rho(rho):
    if rho < 1.0:
        return 0
    elif rho > 24.7:
        return 1
    else:
        return None

# ============================================================
# DATASET GENERATION
# ============================================================

dataset = []

DEBUG_SHOW = 5   # show first 5 samples only

for i in range(NUM_SAMPLES):

    sigma, rho, beta, epsilon = sample_params()

    label = label_from_rho(rho)
    if label is None:
        continue

    x, y, z = simulate_lorenz_sde(sigma, rho, beta, epsilon, T, dt)
    x, y, z = remove_transient(x, y, z, CUT)

    traj = np.stack([x, y], axis=1)
    traj = normalize_traj(traj)

    # ============================================================
    # 🔍 DEBUG VISUALIZATION
    # ============================================================

    if i < DEBUG_SHOW:
        print("\n==============================")
        print(f"Sample {i}")
        print(f"sigma={sigma:.3f}, rho={rho:.3f}, beta={beta:.3f}, epsilon={epsilon:.3f}")
        print(f"label = {label}")
        print("==============================")

        plt.figure(figsize=(5, 5))
        plt.plot(traj[:, 0], traj[:, 1], linewidth=0.8)
        plt.title(f"Lorenz SDE (label={label})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.show()

    # ============================================================
    # STORE SAMPLE
    # ============================================================

    dataset.append({
        "trajectory": traj.astype(np.float32),
        "params": np.array([sigma, rho, beta, epsilon], dtype=np.float32),
        "label": label
    })
# ============================================================
# SAVE
# ============================================================

np.savez(
    "lorenz_dataset.npz",
    trajectories=np.array([d["trajectory"] for d in dataset], dtype=object),
    params=np.array([d["params"] for d in dataset]),
    labels=np.array([d["label"] for d in dataset])
)

print(f"Saved {len(dataset)} samples.")