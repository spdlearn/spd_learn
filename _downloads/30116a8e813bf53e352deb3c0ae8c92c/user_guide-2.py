import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def spd_to_ellipse(spd_matrix, n_points=100):
    """Convert 2x2 SPD matrix to ellipse coordinates."""
    eigvals, eigvecs = np.linalg.eigh(spd_matrix)
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.array([np.cos(theta), np.sin(theta)])
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    ellipse = transform @ circle
    return ellipse[0], ellipse[1]

# Create two very different SPD matrices
A = np.array([[4.0, 0.0], [0.0, 0.25]])  # Large in x, small in y
B = np.array([[0.25, 0.0], [0.0, 4.0]])  # Small in x, large in y

# Euclidean mean
mean_euclidean = (A + B) / 2

# Determinants
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
det_mean = np.linalg.det(mean_euclidean)
geo_mean_det = np.sqrt(det_A * det_B)

fig, ax = plt.subplots(figsize=(8, 8))

# Plot ellipses
x_A, y_A = spd_to_ellipse(A)
x_B, y_B = spd_to_ellipse(B)
x_mean, y_mean = spd_to_ellipse(mean_euclidean)

ax.fill(x_A, y_A, alpha=0.3, color='blue', label=f'A (det={det_A:.2f})')
ax.plot(x_A, y_A, 'b-', linewidth=2)
ax.fill(x_B, y_B, alpha=0.3, color='green', label=f'B (det={det_B:.2f})')
ax.plot(x_B, y_B, 'g-', linewidth=2)
ax.fill(x_mean, y_mean, alpha=0.3, color='red', label=f'Euclidean Mean (det={det_mean:.2f})')
ax.plot(x_mean, y_mean, 'r-', linewidth=2)

# Reference: ideal size
ideal_scale = np.sqrt(geo_mean_det / det_mean)
ax.plot(x_mean * ideal_scale, y_mean * ideal_scale, 'k--', linewidth=2,
        label=f'Expected size (det={geo_mean_det:.2f})')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)
ax.set_title(f'The Swelling Effect: Euclidean Mean is Too Large\nSwelling ratio: {det_mean/geo_mean_det:.2f}x',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()