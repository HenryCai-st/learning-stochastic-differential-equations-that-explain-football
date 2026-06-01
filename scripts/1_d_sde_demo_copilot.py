import numpy as np
import matplotlib.pyplot as plt
import torch
import torchsde

class SDE(torch.nn.Module):
    """SDE: dY = mu * Y * dt + sigma * Y * dW"""

    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
    
    def f(self, t, y):
        """Drift coefficient"""
        return self.mu * y
    
    def g(self, t, y):
        """Diffusion coefficient"""
        return self.sigma * y

# Parameters
mu = 0.1  # drift coefficient
sigma = 0.2  # volatility coefficient
Y0 = 1.0  # initial condition
T = 3.0  # final time
dt = 0.01  # time step
N = int(T / dt)  # number of steps

# Time array
t = np.linspace(0, T, N)
ts = torch.tensor(t, dtype=torch.float32)

# Initialize SDE
sde = SDE(mu, sigma)

# Stochastic solution using torchsde
np.random.seed(42)
torch.manual_seed(42)
Y0_torch = torch.tensor([[Y0]], dtype=torch.float32)  # shape: (1, 1)

# Generate 5 sample paths
Y_stochastic_paths = []
for _ in range(5):
    ys = torchsde.sdeint(sde, Y0_torch, ts, method='euler')
    Y_stochastic_paths.append(ys.squeeze().detach().numpy())

Y_stochastic = np.column_stack(Y_stochastic_paths)

# Deterministic solution (drift only, no diffusion)
Y_deterministic = np.zeros((N, 1))
Y_deterministic[0, 0] = Y0
for i in range(1, N):
    Y_deterministic[i, 0] = Y_deterministic[i-1, 0] * np.exp(mu * dt)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Deterministic vs Stochastic paths
ax1 = axes[0]
ax1.plot(t, Y_deterministic, 'b-', linewidth=2, label='Deterministic')
for path in range(5):
    ax1.plot(t, Y_stochastic[:, path], alpha=0.6, label=f'Stochastic Path {path+1}')
ax1.set_xlabel('Time')
ax1.set_ylabel('Y(t)')
ax1.set_title('SDE: dY = μY dt + σY dW')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Multiple stochastic paths
ax2 = axes[1]
for path in range(5):
    ax2.plot(t, Y_stochastic[:, path], alpha=0.7)
ax2.set_xlabel('Time')
ax2.set_ylabel('Y(t)')
ax2.set_title(f'Stochastic Paths (μ={mu}, σ={sigma})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('1_sde_solutions.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Final values:")
print(f"Deterministic: {Y_deterministic[-1, 0]:.4f}")
print(f"Stochastic (mean): {np.mean(Y_stochastic[-1, :]):.4f}")
print(f"Stochastic (std): {np.std(Y_stochastic[-1, :]):.4f}")
