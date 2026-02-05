import torch
import numpy as np
import matplotlib.pyplot as plt
from spd_learn.modules import CovLayer

# Generate synthetic multivariate time series
torch.manual_seed(42)
batch_size, n_channels, n_times = 1, 8, 100

# Create correlated signals
raw_signals = torch.randn(batch_size, n_channels, n_times)
mixing_matrix = torch.randn(n_channels, n_channels)
raw_signals = torch.einsum('ij,bjt->bit', mixing_matrix, raw_signals)

# Apply CovLayer
cov_layer = CovLayer()
covariances = cov_layer(raw_signals)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot raw signal (first 3 channels)
ax1 = axes[0]
for i in range(3):
    ax1.plot(raw_signals[0, i, :].numpy(), label=f'Ch {i+1}', alpha=0.8)
ax1.set_xlabel('Time samples')
ax1.set_ylabel('Amplitude')
ax1.set_title('Raw Signal (3 channels)', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot covariance matrix
ax2 = axes[1]
im = ax2.imshow(covariances[0].numpy(), cmap='RdBu_r', aspect='auto')
ax2.set_title('Covariance Matrix', fontweight='bold')
ax2.set_xlabel('Channel')
ax2.set_ylabel('Channel')
plt.colorbar(im, ax=ax2, shrink=0.8)

# Plot eigenvalue spectrum
ax3 = axes[2]
eigvals = torch.linalg.eigvalsh(covariances[0]).numpy()
ax3.bar(range(n_channels), sorted(eigvals, reverse=True), color='#3498db', alpha=0.8)
ax3.set_xlabel('Eigenvalue index')
ax3.set_ylabel('Eigenvalue')
ax3.set_title('Eigenvalue Spectrum', fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

plt.suptitle('CovLayer: Raw Signal to SPD Covariance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()