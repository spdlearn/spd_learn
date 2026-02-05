import torch
import numpy as np
import matplotlib.pyplot as plt
from spd_learn.modules import BiMap

torch.manual_seed(42)

# Create an 8x8 SPD matrix
n_in, n_out = 8, 4
A = torch.randn(n_in, n_in)
X = A @ A.T + 0.1 * torch.eye(n_in)
X = X.unsqueeze(0)  # Add batch dimension

# Apply BiMap
bimap = BiMap(in_features=n_in, out_features=n_out, parametrized=True)
Y = bimap(X)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Input
ax1 = axes[0]
im1 = ax1.imshow(X[0].numpy(), cmap='RdBu_r', aspect='auto')
ax1.set_title(f'Input X ({n_in}x{n_in})', fontweight='bold')
plt.colorbar(im1, ax=ax1, shrink=0.8)

# Weight matrix W
ax2 = axes[1]
W = bimap.weight[0].detach().numpy()
im2 = ax2.imshow(W, cmap='RdBu_r', aspect='auto')
ax2.set_title(f'W ({n_in}x{n_out}, Stiefel)', fontweight='bold')
ax2.set_xlabel('Output dim')
ax2.set_ylabel('Input dim')
plt.colorbar(im2, ax=ax2, shrink=0.8)

# W^T W (should be identity)
ax3 = axes[2]
WtW = (bimap.weight[0].T @ bimap.weight[0]).detach().numpy()
im3 = ax3.imshow(WtW, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=1.1)
ax3.set_title(r'$W^T W$ (Identity)', fontweight='bold')
plt.colorbar(im3, ax=ax3, shrink=0.8)

# Output
ax4 = axes[3]
im4 = ax4.imshow(Y[0].detach().numpy(), cmap='RdBu_r', aspect='auto')
ax4.set_title(f'Output Y ({n_out}x{n_out})', fontweight='bold')
plt.colorbar(im4, ax=ax4, shrink=0.8)

plt.suptitle(r'BiMap: $Y = W^T X W$ (Bilinear Mapping)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()