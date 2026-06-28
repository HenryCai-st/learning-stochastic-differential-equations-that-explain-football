"""Without the z axis - Multiple trajectories."""

import sys
sys.setrecursionlimit(10000)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchsde
from torchsde import BrownianInterval

# ============================================================
# STOCHASTIC LORENZ SYSTEM (2D)
# ============================================================
class StochasticLorenz(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, lorenz_sigma=10.0, rho=28.0, noise_scale=0.05):
        super().__init__()
        self.lorenz_sigma = float(lorenz_sigma)
        self.rho = float(rho)
        self.noise_scale = float(noise_scale)

    def f(self, t, state):
        x, y = state[:, 0], state[:, 1]
        dx = self.lorenz_sigma * (y - x)
        dy = x * (self.rho - 1) - y
        return torch.stack([dx, dy], dim=1)

    def g(self, t, state):
        x, y = state[:, 0], state[:, 1]
        return torch.stack([self.noise_scale * x, self.noise_scale * y], dim=1)


# ============================================================
# PARAMETERS
# ============================================================
torch.manual_seed(0)
np.random.seed(0)

T = 10.0
dt = 0.005
t = np.arange(0, T + dt, dt)
ts = torch.tensor(t, dtype=torch.float32)

# ------------------------------------------------------------------
# CASE 1: Multiple initial conditions (same drift parameters)
# ------------------------------------------------------------------
num_trajectories = 5
# Create different starting points around (1,1)
init_conditions = [
    torch.tensor([[1.0, 1.0]]),
    torch.tensor([[2.0, 2.0]]),
    torch.tensor([[0.0, 0.0]]),
    torch.tensor([[-1.0, -1.0]]),
    torch.tensor([[1.5, 0.5]]),
]
y0_batch = torch.cat(init_conditions, dim=0)   # shape (num_traj, 2)

sde = StochasticLorenz(lorenz_sigma=1.0, rho=2.80, noise_scale=0.1)

# Brownian motion for a batch of trajectories
bm = BrownianInterval(
    t0=ts[0],
    t1=ts[-1],
    size=(num_trajectories, 2),          # (batch, state_dim)
    device=y0_batch.device,
    levy_area_approximation="none"
)

# Solve all trajectories at once
ys_batch = torchsde.sdeint(sde, y0_batch, ts, bm=bm, method="euler")
ys_batch = ys_batch.detach().numpy()     # shape (time, batch, 2)

# ------------------------------------------------------------------
# Plot all trajectories in the same x-y plane
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(121)
for i in range(num_trajectories):
    xs = ys_batch[:, i, 0]
    ys_coord = ys_batch[:, i, 1]
    ax1.plot(xs, ys_coord, lw=0.8, alpha=0.7, label=f'IC{i+1}')
ax1.set_title("Multiple trajectories (different initial conditions)")
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(122)
for i in range(num_trajectories):
    ax2.plot(t, ys_batch[:, i, 0], lw=0.5, alpha=0.7)
    ax2.plot(t, ys_batch[:, i, 1], lw=0.5, alpha=0.7, linestyle='--')
ax2.set_title("x (solid) and y (dashed) vs time")
ax2.set_xlabel("Time")
ax2.grid(True)

plt.tight_layout()
plt.savefig('multiple_trajectories_ic.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CASE 2 (optional): Different drift parameters, same initial condition
# ============================================================
if False:   # set to True to run second case
    rho_values = [20.0, 28.0, 35.0]
    colors = ['blue', 'green', 'red']
    y0_single = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    fig2, ax = plt.subplots(figsize=(7,5))
    for rho, col in zip(rho_values, colors):
        sde_var = StochasticLorenz(lorenz_sigma=10.0, rho=rho, noise_scale=0.03)
        bm2 = BrownianInterval(t0=ts[0], t1=ts[-1], size=(1,2), device=y0_single.device,
                               levy_area_approximation="none")
        ys_var = torchsde.sdeint(sde_var, y0_single, ts, bm=bm2, method="euler")
        ys_var = ys_var.squeeze().detach().numpy()
        ax.plot(ys_var[:,0], ys_var[:,1], lw=0.7, color=col, label=f'ρ = {rho}')
    ax.set_title("Effect of drift parameter ρ")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend()
    plt.savefig('drift_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()