import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def create_spd_ellipse(ax, spd_matrix, center=(0, 0), color='blue', alpha=0.5, label=None):
    """Visualize a 2x2 SPD matrix as an ellipse."""
    eigvals, eigvecs = np.linalg.eigh(spd_matrix)
    width = 2 * np.sqrt(eigvals[1])
    height = 2 * np.sqrt(eigvals[0])
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

    ellipse = Ellipse(center, width, height, angle=angle,
                      alpha=alpha, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(ellipse)

    # Draw eigenvalue axes
    for i in range(2):
        vec = eigvecs[:, i] * np.sqrt(eigvals[i])
        ax.arrow(center[0], center[1], vec[0], vec[1],
                 head_width=0.1, color='red', linewidth=2, alpha=0.8)

    if label:
        ax.text(center[0], center[1] - 2.2, label, ha='center', fontsize=10)

# Create SPD matrices with different properties
S1 = np.array([[3.0, 0.5], [0.5, 1.0]])  # Anisotropic
S2 = np.array([[2.0, 0.0], [0.0, 2.0]])  # Isotropic
S3 = np.array([[4.0, 1.5], [1.5, 1.0]])  # Highly anisotropic

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, S, title in zip(axes, [S1, S2, S3],
    ['Anisotropic', 'Isotropic (Identity-like)', 'Highly Anisotropic']):
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    eigvals = np.linalg.eigvalsh(S)
    create_spd_ellipse(ax, S, color='#3498db')
    ax.set_title(f'{title}\n$\\lambda$ = [{eigvals[0]:.2f}, {eigvals[1]:.2f}]', fontsize=11)

plt.suptitle('SPD Matrices Visualized as Ellipses (Red: Eigenvector Axes)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()