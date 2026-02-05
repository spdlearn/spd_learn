import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from spd_learn.modules import SPDBatchNormMean

def spd_to_ellipse(spd_matrix, center=(0, 0), scale=1.0):
    eigvals, eigvecs = np.linalg.eigh(spd_matrix)
    width = 2 * np.sqrt(eigvals[1]) * scale
    height = 2 * np.sqrt(eigvals[0]) * scale
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    return Ellipse(center, width, height, angle=angle)

# Create batch of 2x2 SPD matrices
torch.manual_seed(42)
np.random.seed(42)
batch_size = 6

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

# Apply SPDBatchNormMean
bn = SPDBatchNormMean(num_features=2, momentum=0.1, rebias=False)
bn.train()
Y = bn(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = plt.cm.tab10(np.linspace(0, 1, batch_size))

# Before normalization
ax1 = axes[0]
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
ax1.set_title('Before SPDBatchNormMean', fontweight='bold')

# After normalization
ax2 = axes[1]
for i, S in enumerate(Y.detach().numpy()):
    ellipse = spd_to_ellipse(S, scale=0.5)
    ellipse.set_facecolor(colors[i])
    ellipse.set_alpha(0.6)
    ellipse.set_edgecolor('black')
    ax2.add_patch(ellipse)
identity = Ellipse((0, 0), 1, 1, facecolor='none',
                   edgecolor='red', linewidth=2, linestyle='--')
ax2.add_patch(identity)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_title('After SPDBatchNormMean', fontweight='bold')

plt.suptitle('SPDBatchNormMean: Riemannian Centering', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()