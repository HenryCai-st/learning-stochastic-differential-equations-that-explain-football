import torch
import torchsde
import matplotlib.pyplot as plt
import numpy as np

# --- Lorenz parameters ---
SIGMA = 10.0
RHO   = 28.0
BETA  = 8/3

# Noise scales (set to 0.0 for deterministic Lorenz)
A1, A2, A3 = 0.5, 0.5, 0.5

batch_size = 32
t_size     = 2000

class LorenzSDE(torch.nn.Module):
    noise_type = 'diagonal'   # each state dim gets its own scalar noise
    sde_type   = 'ito'

    def __init__(self, sigma, rho, beta, a1, a2, a3):
        super().__init__()
        # Register as buffers so they move with .to(device) cleanly
        self.register_buffer('sigma', torch.tensor(sigma))
        self.register_buffer('rho',   torch.tensor(rho))
        self.register_buffer('beta',  torch.tensor(beta))
        self.register_buffer('A',     torch.tensor([a1, a2, a3]))

    def f(self, t, y):
        # y shape: (batch, 3)
        x, yy, z = y[:, 0], y[:, 1], y[:, 2]
        dX = self.sigma * (yy - x)
        dY = x * (self.rho - z) - yy
        dZ = x * yy - self.beta * z
        return torch.stack([dX, dY, dZ], dim=1)

    def g(self, t, y):
        # diagonal noise: output shape (batch, 3)
        return self.A.expand(y.shape[0], 3)


sde = LorenzSDE(SIGMA, RHO, BETA, A1, A2, A3)

# Classic Lorenz starting point (+ tiny perturbation per batch element)
torch.manual_seed(42)
y0_base = torch.tensor([0.1, 0.0, 0.0])
y0 = y0_base.unsqueeze(0).expand(batch_size, 3).clone()
y0 += torch.randn_like(y0) * 0.01   # small spread so paths diverge

ts = torch.linspace(0, 5, t_size)
with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts)   # (t_size, batch, 3)

ys_np = ys.numpy()
ts_np = ts.numpy()
# ── Plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))
colors = plt.cm.plasma(np.linspace(0, 1, batch_size))

# 1) 3-D attractor
ax1 = fig.add_subplot(131, projection='3d')
for i in range(batch_size):
    ax1.plot(ys_np[:, i, 0], ys_np[:, i, 1], ys_np[:, i, 2],
             lw=0.4, alpha=0.6, color=colors[i])
ax1.set_title("3D Lorenz attractor")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

# 2) X vs Z projection (classic butterfly view)
ax2 = fig.add_subplot(132)
for i in range(batch_size):
    ax2.plot(ys_np[:, i, 0], ys_np[:, i, 2],
             lw=0.3, alpha=0.5, color=colors[i])
ax2.set_title("X–Z projection (butterfly)")
ax2.set_xlabel("X"); ax2.set_ylabel("Z")

# 3) X time series
ax3 = fig.add_subplot(133)
for i in range(batch_size):
    ax3.plot(ts_np, ys_np[:, i, 0],
             lw=0.4, alpha=0.4, color=colors[i])
ax3.set_title("X(t) time series")
ax3.set_xlabel("t"); ax3.set_ylabel("X")

plt.suptitle(
    f"Stochastic Lorenz  σ={SIGMA}  ρ={RHO}  β={BETA:.2f}  "
    f"A=({A1},{A2},{A3})", fontsize=11)
plt.tight_layout()
plt.savefig("lorenz_sde.png", dpi=150)
plt.show()