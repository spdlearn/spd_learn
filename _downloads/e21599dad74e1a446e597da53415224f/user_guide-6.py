import torch
import numpy as np
import matplotlib.pyplot as plt
from spd_learn.modules import LogEig

torch.manual_seed(42)

# Create a 4x4 SPD matrix
n = 4
A = torch.randn(n, n)
X = A @ A.T + 0.1 * torch.eye(n)
X = X.unsqueeze(0)

# Apply LogEig (with and without vectorization)
logeig_full = LogEig(upper=False, flatten=False)
logeig_vec = LogEig(upper=True, flatten=True)

log_matrix = logeig_full(X)
log_vector = logeig_vec(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Input SPD matrix
ax1 = axes[0]
im1 = ax1.imshow(X[0].numpy(), cmap='RdBu_r', aspect='auto')
ax1.set_title('Input SPD Matrix X', fontweight='bold')
plt.colorbar(im1, ax=ax1, shrink=0.8)

# Matrix logarithm
ax2 = axes[1]
im2 = ax2.imshow(log_matrix[0].numpy(), cmap='RdBu_r', aspect='auto')
ax2.set_title(r'log(X) (Tangent Space)', fontweight='bold')
plt.colorbar(im2, ax=ax2, shrink=0.8)

# Vectorized output
ax3 = axes[2]
vec = log_vector[0].numpy()
ax3.bar(range(len(vec)), vec, color='#2ecc71', alpha=0.8)
ax3.set_xlabel('Vector index')
ax3.set_ylabel('Value')
ax3.set_title(f'Vectorized (dim={len(vec)})', fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.suptitle('LogEig: SPD to Tangent Space Mapping', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()