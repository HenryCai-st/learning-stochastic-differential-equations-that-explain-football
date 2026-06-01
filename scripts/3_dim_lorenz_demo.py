"""Without the z axis."""

import sys
sys.setrecursionlimit(10000)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchsde
from torchsde import BrownianInterval


# ============================================================
# STOCHASTIC LORENZ SYSTEM
# ============================================================

class StochasticLorenz(torch.nn.Module):

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, lorenz_sigma=10.0, rho=28.0, noise_scale=0.05): # beta=8/3, noise_scale=0.05):
        super().__init__()
        self.lorenz_sigma = float(lorenz_sigma)
        self.rho           = float(rho)
        #self.beta          = float(beta)
        self.noise_scale   = float(noise_scale)

    def f(self, t, state):
        x, y = state[:, 0], state[:, 1]
        # x, y, z = state[:, 0], state[:, 1], state[:, 2]
        dx = self.lorenz_sigma * (y - x)
        dy = x * (self.rho - 1) - y
        #dz = x * y - self.beta * z
        return torch.stack([dx, dy ], dim=1)  #, dz], dim=1)

    def g(self, t, state):
        x, y =  state[:, 0], state[:, 1]
        #x, y, z = state[:, 0], state[:, 1], state[:, 2]
        return torch.stack(
            [self.noise_scale * x,
             self.noise_scale * y],
             #self.noise_scale * z],
            dim=1
        )


# ============================================================
# PARAMETERS
# ============================================================

torch.manual_seed(0)
np.random.seed(0)

T  = 20.0
dt = 0.005
t  = np.arange(0, T + dt, dt)
ts = torch.tensor(t, dtype=torch.float32)

y0  = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
sde = StochasticLorenz(lorenz_sigma=1.0, rho=2.8, noise_scale=0.05)

#y0  = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
#sde = StochasticLorenz(lorenz_sigma=10.0, rho=28.0, beta=8/3, noise_scale=0.03)


# ============================================================
# BROWNIAN MOTION  (fixes the recursion / trampoline bug)
# ============================================================

bm = BrownianInterval(
    t0=ts[0],
    t1=ts[-1],
    size=(1, 2),                        # (batch, state_dim)
    device=y0.device,
    levy_area_approximation="none"      # "none" is correct for Euler–Maruyama
)


# ============================================================
# SOLVE
# ============================================================

ys = torchsde.sdeint(sde, y0, ts, bm=bm, method="euler")
ys = ys.squeeze().detach().numpy()

x_coord = ys[:, 0]
y_coord = ys[:, 1]
#z_coord = ys[:, 2]


# ============================================================
# PLOTS
# ============================================================

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(121) #, projection='3d')
ax1.plot(x_coord, y_coord, lw = 0.5) #z_coord, lw=0.5)
ax1.set_title("Stochastic Lorenz Attractor")
ax1.set_xlabel("x"); ax1.set_ylabel("y") # ; ax1.set_zlabel("z")

ax2 = fig.add_subplot(122)
ax2.plot(t, x_coord, label="x")
ax2.plot(t, y_coord, label="y")
#ax2.plot(t, z_coord, label="z")
ax2.set_title("Lorenz Coordinates")
ax2.set_xlabel("Time")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('2_d_lorenz_demo.png', dpi=150, bbox_inches='tight')
plt.show()