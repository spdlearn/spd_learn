import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from spd_learn.modules import SPDBatchNormMeanVar

def spd_to_ellipse(spd_matrix, center=(0, 0), scale=1.0):
    eigvals, eigvecs = np.linalg.eigh(spd_matrix)
    width = 2 * np.sqrt(eigvals[1]) * scale
    height = 2 * np.sqrt(eigvals[0]) * scale
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    return Ellipse(center, width, height, angle=angle)

# Create batch of 2x2 SPD matrices with different means
torch.manual_seed(42)
np.random.seed(42)
batch_size = 6

# Generate scattered SPD matrices
spd_batch = []
for i in range(batch_size):
    scale = np.random.uniform(0.5, 2.0)
    angle = np.random.uniform(0, np.pi)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    D = np.diag([scale, scale * np.random.uniform(0.3, 1.0)])
    S = R @ D @ D @ R.T
    spd_batch.append(S)

X = torch.tensor(np.array(spd_batch), dtype=torch.float32)

# Apply SPDBatchNormMeanVar
bn = SPDBatchNormMeanVar(num_features=2, momentum=0.1, affine=False)
bn.train()
Y = bn(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Before normalization
ax1 = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, batch_size))
for i, S in enumerate(X.numpy()):
    ellipse = spd_to_ellipse(S, scale=0.5)
    ellipse.set_facecolor(colors[i])
    ellipse.set_alpha(0.6)
    ellipse.set_edgecolor('black')
    ax1.add_patch(ellipse)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_title('Before SPDBatchNormMeanVar\n(Scattered)', fontweight='bold')

# After normalization
ax2 = axes[1]
for i, S in enumerate(Y.detach().numpy()):
    ellipse = spd_to_ellipse(S, scale=0.5)
    ellipse.set_facecolor(colors[i])
    ellipse.set_alpha(0.6)
    ellipse.set_edgecolor('black')
    ax2.add_patch(ellipse)
# Draw identity reference
identity = Ellipse((0, 0), 1, 1, facecolor='none',
                   edgecolor='red', linewidth=2, linestyle='--')
ax2.add_patch(identity)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_title('After SPDBatchNormMeanVar\n(Centered at Identity)', fontweight='bold')

# Eigenvalue comparison
ax3 = axes[2]
eigvals_before = [np.linalg.eigvalsh(s) for s in X.numpy()]
eigvals_after = [np.linalg.eigvalsh(s) for s in Y.detach().numpy()]

x_pos = np.arange(batch_size)
width = 0.35
ax3.bar(x_pos - width/2,
        [np.prod(e) for e in eigvals_before],
        width, label='Before (det)', color='#3498db', alpha=0.8)
ax3.bar(x_pos + width/2,
        [np.prod(e) for e in eigvals_after],
        width, label='After (det)', color='#e74c3c', alpha=0.8)
ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Identity det=1')
ax3.set_xlabel('Sample index')
ax3.set_ylabel('Determinant')
ax3.set_title('Determinant Normalization', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.suptitle('SPDBatchNormMeanVar: Riemannian Batch Normalization', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()