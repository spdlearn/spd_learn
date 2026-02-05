import torch
import numpy as np
import matplotlib.pyplot as plt
from spd_learn.functional import matrix_log, matrix_exp, log_euclidean_mean

def spd_to_ellipse(spd_matrix, n_points=100):
    eigvals, eigvecs = np.linalg.eigh(spd_matrix)
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.array([np.cos(theta), np.sin(theta)])
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    ellipse = transform @ circle
    return ellipse[0], ellipse[1]

# Create two SPD matrices
S1 = torch.tensor([[4.0, 0.0], [0.0, 0.25]], dtype=torch.float64)
S2 = torch.tensor([[0.25, 0.0], [0.0, 4.0]], dtype=torch.float64)

# Euclidean mean
mean_euclidean = (S1 + S2) / 2

# Log-Euclidean mean
S_stack = torch.stack([S1, S2], dim=0)
weights = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
mean_log_euclidean = log_euclidean_mean(weights, S_stack).squeeze(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Convert to numpy
S1_np, S2_np = S1.numpy(), S2.numpy()
mean_euc_np = mean_euclidean.numpy()
mean_le_np = mean_log_euclidean.numpy()

# Euclidean mean
ax1 = axes[0]
for S, color, label in [(S1_np, 'blue', 'S1'), (S2_np, 'green', 'S2')]:
    x, y = spd_to_ellipse(S)
    ax1.fill(x, y, alpha=0.3, color=color, label=label)
    ax1.plot(x, y, color=color, linewidth=2)
x, y = spd_to_ellipse(mean_euc_np)
ax1.fill(x, y, alpha=0.3, color='red', label=f'Euclidean (det={np.linalg.det(mean_euc_np):.2f})')
ax1.plot(x, y, 'r-', linewidth=2)
ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3)
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_title('Euclidean Mean\n(Swelling Effect)', fontsize=12, fontweight='bold')

# Log-Euclidean mean
ax2 = axes[1]
for S, color, label in [(S1_np, 'blue', 'S1'), (S2_np, 'green', 'S2')]:
    x, y = spd_to_ellipse(S)
    ax2.fill(x, y, alpha=0.3, color=color, label=label)
    ax2.plot(x, y, color=color, linewidth=2)
x, y = spd_to_ellipse(mean_le_np)
ax2.fill(x, y, alpha=0.3, color='purple', label=f'Log-Euclidean (det={np.linalg.det(mean_le_np):.2f})')
ax2.plot(x, y, 'm-', linewidth=2)
ax2.set_xlim(-3, 3); ax2.set_ylim(-3, 3)
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)
ax2.set_title('Log-Euclidean Mean\n(Respects Geometry)', fontsize=12, fontweight='bold')

plt.suptitle('Comparison of Mean Computation Methods', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()