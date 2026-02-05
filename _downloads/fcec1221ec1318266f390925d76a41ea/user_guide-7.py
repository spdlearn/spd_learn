import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from spd_learn.modules import CovLayer, BiMap, ReEig, LogEig, Shrinkage

class SimpleSPDNet(nn.Module):
    """A simple SPD network for demonstration."""

    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.cov = CovLayer()
        self.shrinkage = Shrinkage(n_chans=n_channels, init_shrinkage=0.1)
        self.bimap1 = BiMap(in_features=n_channels, out_features=n_channels // 2)
        self.reeig1 = ReEig()
        self.bimap2 = BiMap(in_features=n_channels // 2, out_features=n_channels // 4)
        self.reeig2 = ReEig()
        self.logeig = LogEig(upper=True)
        tangent_dim = (n_channels // 4) * (n_channels // 4 + 1) // 2
        self.classifier = nn.Linear(tangent_dim, n_classes)

    def forward(self, x, return_intermediates=False):
        intermediates = {}
        x = self.cov(x); intermediates['cov'] = x.clone()
        x = self.shrinkage(x); intermediates['shrinkage'] = x.clone()
        x = self.bimap1(x); intermediates['bimap1'] = x.clone()
        x = self.reeig1(x); intermediates['reeig1'] = x.clone()
        x = self.bimap2(x); intermediates['bimap2'] = x.clone()
        x = self.reeig2(x); intermediates['reeig2'] = x.clone()
        x = self.logeig(x); intermediates['logeig'] = x.clone()
        x = self.classifier(x); intermediates['output'] = x.clone()
        if return_intermediates:
            return x, intermediates
        return x

# Create and run the network
torch.manual_seed(42)
n_channels, n_classes = 16, 4
model = SimpleSPDNet(n_channels=n_channels, n_classes=n_classes)

raw_input = torch.randn(1, n_channels, 200)
output, intermediates = model(raw_input, return_intermediates=True)

# Visualize the pipeline in computation order
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1: Matrix representations (computation order)
ax1 = axes[0, 0]
ax1.plot(raw_input[0, :3, :].T.numpy(), alpha=0.7)
ax1.set_title('1. Raw Signal', fontweight='bold')
ax1.set_xlabel('Time')

ax2 = axes[0, 1]
im2 = ax2.imshow(intermediates['cov'][0].detach().numpy(), cmap='RdBu_r', aspect='auto')
ax2.set_title('2. Covariance (16x16)', fontweight='bold')

ax3 = axes[0, 2]
im3 = ax3.imshow(intermediates['bimap1'][0].detach().numpy(), cmap='RdBu_r', aspect='auto')
ax3.set_title('3. After BiMap1 (8x8)', fontweight='bold')

ax4 = axes[0, 3]
im4 = ax4.imshow(intermediates['reeig1'][0].detach().numpy(), cmap='RdBu_r', aspect='auto')
ax4.set_title('4. After ReEig1 (8x8)', fontweight='bold')

# Row 2: Show ReEig effect with before/after eigenvalue comparison
eigvals_bimap1 = torch.linalg.eigvalsh(intermediates['bimap1'][0]).detach().numpy()
eigvals_reeig1 = torch.linalg.eigvalsh(intermediates['reeig1'][0]).detach().numpy()
eigvals_bimap2 = torch.linalg.eigvalsh(intermediates['bimap2'][0]).detach().numpy()
eigvals_reeig2 = torch.linalg.eigvalsh(intermediates['reeig2'][0]).detach().numpy()

# Compute shared y-limits across all eigenvalue plots
all_eigvals = np.concatenate([eigvals_bimap1, eigvals_reeig1, eigvals_bimap2, eigvals_reeig2])
y_min, y_max = max(all_eigvals[all_eigvals > 0].min() * 0.3, 1e-6), all_eigvals.max() * 2

ax5 = axes[1, 0]
ax5.bar(range(len(eigvals_bimap1)), sorted(eigvals_bimap1, reverse=True), color='#3498db')
ax5.set_title('3. BiMap1 eigenvalues', fontweight='bold')
ax5.set_yscale('log')
ax5.set_ylim(y_min, y_max)
ax5.axhline(y=1e-4, color='red', linestyle='--', alpha=0.7, label='ReEig threshold')
ax5.legend(loc='upper right', fontsize=8)

ax6 = axes[1, 1]
ax6.bar(range(len(eigvals_reeig1)), sorted(eigvals_reeig1, reverse=True), color='#e74c3c')
ax6.set_title('4. After ReEig1 (rectified)', fontweight='bold')
ax6.set_yscale('log')
ax6.set_ylim(y_min, y_max)
ax6.axhline(y=1e-4, color='red', linestyle='--', alpha=0.7)

ax7 = axes[1, 2]
ax7.bar(range(len(eigvals_bimap2)), sorted(eigvals_bimap2, reverse=True), color='#2ecc71')
ax7.set_title('5. BiMap2 eigenvalues', fontweight='bold')
ax7.set_yscale('log')
ax7.set_ylim(y_min, y_max)
ax7.axhline(y=1e-4, color='red', linestyle='--', alpha=0.7)

ax8 = axes[1, 3]
ax8.bar(range(len(eigvals_reeig2)), sorted(eigvals_reeig2, reverse=True), color='#9b59b6')
ax8.set_title('6. After ReEig2 (rectified)', fontweight='bold')
ax8.set_yscale('log')
ax8.set_ylim(y_min, y_max)
ax8.axhline(y=1e-4, color='red', linestyle='--', alpha=0.7)

plt.suptitle('SPDNet Pipeline: Computation Flow & ReEig Effect', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()