import torch
import numpy as np
import matplotlib.pyplot as plt
from spd_learn.modules import Shrinkage, CovLayer

torch.manual_seed(42)

# Generate synthetic data and compute covariance
n_channels = 8
raw_signals = torch.randn(1, n_channels, 100)
mixing = torch.randn(n_channels, n_channels)
raw_signals = torch.einsum('ij,bjt->bit', mixing, raw_signals)

cov_layer = CovLayer()
covariances = cov_layer(raw_signals)

# Apply shrinkage with different coefficients
shrinkage_low = Shrinkage(n_chans=n_channels, init_shrinkage=-2.0)  # ~0.12
shrinkage_high = Shrinkage(n_chans=n_channels, init_shrinkage=2.0)  # ~0.88

cov_low = shrinkage_low(covariances)
cov_high = shrinkage_high(covariances)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Original eigenvalues
eigvals_orig = torch.linalg.eigvalsh(covariances[0]).numpy()
eigvals_low = torch.linalg.eigvalsh(cov_low[0].detach()).numpy()
eigvals_high = torch.linalg.eigvalsh(cov_high[0].detach()).numpy()

for ax, eigv, title, color in zip(
    axes,
    [eigvals_orig, eigvals_low, eigvals_high],
    ['Original', r'Shrinkage $\alpha \approx 0.12$', r'Shrinkage $\alpha \approx 0.88$'],
    ['#3498db', '#e74c3c', '#2ecc71']
):
    ax.bar(range(n_channels), sorted(eigv, reverse=True), color=color, alpha=0.8)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(title, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=min(eigv), color='red', linestyle='--', alpha=0.5)
    cond = max(eigv) / min(eigv)
    ax.text(0.95, 0.95, f'Cond: {cond:.1f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

plt.suptitle('Shrinkage: Eigenvalue Regularization', fontweight='bold')
plt.tight_layout()
plt.show()