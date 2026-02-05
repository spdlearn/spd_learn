import torch
import numpy as np
import matplotlib.pyplot as plt
from spd_learn.modules import SPDDropout

torch.manual_seed(42)

# Create an SPD matrix
n_channels = 8
A = torch.randn(n_channels, n_channels)
X = A @ A.T + 0.1 * torch.eye(n_channels)
X = X.unsqueeze(0)

# Apply SPDDropout with high probability for visualization
dropout = SPDDropout(p=0.5, epsilon=1e-3)
dropout.train()

# Run multiple times to show effect
torch.manual_seed(123)
Y = dropout(X)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Original matrix
ax1 = axes[0]
im1 = ax1.imshow(X[0].numpy(), cmap='RdBu_r', aspect='auto')
ax1.set_title('Original SPD Matrix', fontweight='bold')
ax1.set_xlabel('Channel')
ax1.set_ylabel('Channel')
plt.colorbar(im1, ax=ax1, shrink=0.8)

# After dropout
ax2 = axes[1]
im2 = ax2.imshow(Y[0].detach().numpy(), cmap='RdBu_r', aspect='auto')
ax2.set_title('After SPDDropout (p=0.5)', fontweight='bold')
ax2.set_xlabel('Channel')
ax2.set_ylabel('Channel')
plt.colorbar(im2, ax=ax2, shrink=0.8)

# Eigenvalue comparison
ax3 = axes[2]
eigvals_before = torch.linalg.eigvalsh(X[0]).numpy()
eigvals_after = torch.linalg.eigvalsh(Y[0]).numpy()

x_pos = np.arange(n_channels)
ax3.bar(x_pos - 0.2, sorted(eigvals_before, reverse=True), 0.35,
        label='Before', color='#3498db', alpha=0.8)
ax3.bar(x_pos + 0.2, sorted(eigvals_after, reverse=True), 0.35,
        label='After', color='#e74c3c', alpha=0.8)
ax3.set_xlabel('Eigenvalue index')
ax3.set_ylabel('Eigenvalue')
ax3.set_title('Eigenvalue Spectrum', fontweight='bold')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.suptitle('SPDDropout: Structured Regularization', fontweight='bold')
plt.tight_layout()
plt.show()