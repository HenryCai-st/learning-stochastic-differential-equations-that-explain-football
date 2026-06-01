import numpy as np
import matplotlib.pyplot as plt
# ============================================================
# LORENZ SDE DATASET GENERATION
# ============================================================

# def generate_lorenz_dataset(num_samples=1000):

#     dataset = []
#     for _ in range(num_samples):
#         sigma, rho, beta, epsilon = sample_params()
#         x, y, z = simulate_lorenz_sde(sigma, rho, beta, epsilon)
#         trajectory_2d = (x, y)
#         label = 0 if rho < 1.0 else 1
#         sample = {
#             "trajectory": trajectory_2d,
#             "params": (sigma, rho, beta, epsilon),
#             "label": label
#         }
#         dataset.append(sample)

#     return dataset



sample = {
     "trajectory": [(x1, y1), (x2, y2), ..., (xT, yT)],
     "params": (sigma, rho, beta, epsilon),
    "label": 0 or 1   # fixed / chaos
}

trajectory_2d = (x, y)

def sample_params():
    sigma = np.random.uniform(1, 20)
    beta  = np.random.uniform(0.5, 5)

    # Balanced sampling for rho
    if np.random.rand() < 0.5:
        rho = np.random.uniform(0.5, 1.0)      # fixed
    else:
        rho = np.random.uniform(25, 50)        # chaos

    epsilon = np.random.uniform(0.0, 1.5)

    return sigma, rho, beta, epsilon

def simulate_lorenz_sde(sigma, rho, beta, epsilon, T=50, dt=0.01):
    N = int(T / dt)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    # random initial condition (IMPORTANT)
    x[0], y[0], z[0] = np.random.randn(3)

    for i in range(N - 1):
        dx = sigma * (y[i] - x[i]) * dt + epsilon * np.sqrt(dt) * np.random.randn()
        dy = (x[i] * (rho - z[i]) - y[i]) * dt + epsilon * np.sqrt(dt) * np.random.randn()
        dz = (x[i] * y[i] - beta * z[i]) * dt + epsilon * np.sqrt(dt) * np.random.randn()

        x[i+1] = x[i] + dx
        y[i+1] = y[i] + dy
        z[i+1] = z[i] + dz

    return x, y, z

def remove_transient(x, y, z, cut=1000):
    return x[cut:], y[cut:], z[cut:]

trajectory = np.stack([x, y], axis=1)

def normalize_traj(traj):
    min_vals = traj.min(axis=0)
    max_vals = traj.max(axis=0)
    return (traj - min_vals) / (max_vals - min_vals + 1e-8)

def label_from_rho(rho):
    if rho < 1.0:
        return 0   # fixed
    elif rho > 24.7:
        return 1   # chaos
    else:
        return None  # skip ambiguous
    
    dataset = []

for _ in range(NUM_SAMPLES):
    sigma, rho, beta, epsilon = sample_params()

    label = label_from_rho(rho)
    if label is None:
        continue

    x, y, z = simulate_lorenz_sde(sigma, rho, beta, epsilon)
    x, y, z = remove_transient(x, y, z)

    traj = np.stack([x, y], axis=1)
    traj = normalize_traj(traj)

    dataset.append({
        "trajectory": traj.astype(np.float32),
        "params": np.array([sigma, rho, beta, epsilon], dtype=np.float32),
        "label": label
    })

    np.savez(
    "lorenz_dataset.npz",
    trajectories=[d["trajectory"] for d in dataset],
    params=[d["params"] for d in dataset],
    labels=[d["label"] for d in dataset]
)