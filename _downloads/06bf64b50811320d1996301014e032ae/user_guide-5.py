import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from spd_learn.modules import ReEig

# Threshold
epsilon = 0.3

# Create matrices with varying eigenvalues (some below threshold)
eigval_sets = [
    np.array([0.1, 2.0]),   # First below
    np.array([0.5, 1.5]),   # Both above
    np.array([0.05, 0.8]),  # First below
    np.array([1.0, 0.15]),  # Second below
    np.array([0.2, 0.25]),  # Both below
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot ReEig function
ax1 = axes[0]
x = np.linspace(0, 2.5, 200)
y_reeig = np.maximum(x, epsilon)

ax1.plot(x, x, 'k--', alpha=0.4, label='Identity (y=x)', linewidth=2)
ax1.plot(x, y_reeig, 'b-', linewidth=3, label=f'ReEig ($\\epsilon$={epsilon})')
ax1.fill_between([0, epsilon], [epsilon, epsilon], [0, 0],
                 color='red', alpha=0.15, label='Clamped region')
ax1.axhline(y=epsilon, color='red', linestyle='--', alpha=0.5)
ax1.axvline(x=epsilon, color='red', linestyle='--', alpha=0.5)

# Mark example eigenvalues
colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(eigval_sets)))
for i, eigvals in enumerate(eigval_sets):
    for ev in eigvals:
        out_ev = max(ev, epsilon)
        ax1.scatter([ev], [out_ev], s=100, c=[colors[i]], edgecolors='black', linewidth=1.5, zorder=5)
        if ev < epsilon:
            ax1.plot([ev, ev], [ev, epsilon], color=colors[i], linestyle=':', linewidth=1.5)

ax1.set_xlim(-0.1, 2.5)
ax1.set_ylim(-0.1, 2.5)
ax1.set_xlabel('Input eigenvalue $\\lambda$', fontsize=12)
ax1.set_ylabel('Output eigenvalue max($\\lambda$, $\\epsilon$)', fontsize=12)
ax1.set_title('ReEig: Eigenvalue Rectification', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Eigenvalue comparison bar chart
ax2 = axes[1]
demo_eigvals = torch.tensor([2.0, 0.5, 0.01, 0.001])
demo_eigvecs = torch.linalg.qr(torch.randn(4, 4))[0]
demo_spd = demo_eigvecs @ torch.diag(demo_eigvals) @ demo_eigvecs.T
demo_spd = demo_spd.unsqueeze(0)

reeig = ReEig(threshold=1e-4)
demo_rectified = reeig(demo_spd)

eigvals_before = torch.linalg.eigvalsh(demo_spd[0]).numpy()
eigvals_after = torch.linalg.eigvalsh(demo_rectified[0]).numpy()

x_pos = np.arange(4)
width = 0.35
bars1 = ax2.bar(x_pos - width/2, eigvals_before, width, label='Before ReEig', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, eigvals_after, width, label='After ReEig', color='#e74c3c', alpha=0.8)
ax2.axhline(y=1e-4, color='green', linestyle='--', linewidth=2, label='Threshold (1e-4)')
ax2.set_xlabel('Eigenvalue index', fontsize=12)
ax2.set_ylabel('Eigenvalue', fontsize=12)
ax2.set_title('Eigenvalue Comparison', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.set_xticks(x_pos)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()