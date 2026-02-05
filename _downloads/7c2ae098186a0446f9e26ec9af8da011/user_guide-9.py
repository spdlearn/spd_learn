import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyriemann.datasets import make_gaussian_blobs
from spd_learn.functional import airm_geodesic, airm_distance, log_euclidean_mean, matrix_log, matrix_exp

# Helper: map 2x2 SPD [[a,b],[b,c]] to 3D coords (a, b, c)
def spd_to_3d_coords(S):
    if isinstance(S, torch.Tensor):
        S = S.numpy()
    return S[0, 0], S[0, 1], S[1, 1]

# Helper: convert SPD to ellipse for 2D visualization
def spd_to_ellipse(spd_matrix, n_points=100):
    if isinstance(spd_matrix, torch.Tensor):
        spd_matrix = spd_matrix.numpy()
    eigvals, eigvecs = np.linalg.eigh(spd_matrix)
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.array([np.cos(theta), np.sin(theta)])
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    ellipse = transform @ circle
    return ellipse[0], ellipse[1]

# Generate synthetic 2x2 SPD matrices using pyriemann
np.random.seed(42)
X_spd, y = make_gaussian_blobs(
    n_matrices=15, n_dim=2, class_sep=1.5, class_disp=0.5,
    random_state=42, n_jobs=1
)
X_tensor = torch.tensor(X_spd, dtype=torch.float64)

# Separate by class
class0_mask = y == 0
class1_mask = y == 1
X_class0 = X_tensor[class0_mask]
X_class1 = X_tensor[class1_mask]

# Compute class means using Log-Euclidean mean
# For unweighted mean: average in log-domain, then exp back
mean0 = matrix_exp.apply(matrix_log.apply(X_class0).mean(dim=0))
mean1 = matrix_exp.apply(matrix_log.apply(X_class1).mean(dim=0))

# Use class means as start/end of geodesic
A, B = mean0, mean1

# Compute geodesic points and verify eigenvalues stay positive
t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
geodesic_points = []
print("Eigenvalues along AIRM geodesic (all positive = stays in SPD cone):")
for t in t_values:
    G_t = airm_geodesic(A, B, torch.tensor(t))
    geodesic_points.append(G_t)
    eigvals = torch.linalg.eigvalsh(G_t).numpy()
    print(f"  t={t:.2f}: eigenvalues = [{eigvals[0]:.4f}, {eigvals[1]:.4f}]")

# Compute AIRM distance
dist = airm_distance(A.unsqueeze(0), B.unsqueeze(0)).item()
print(f"\nAIRM distance between class means: {dist:.4f}")

# Create figure with 3D cone (left) and 2D ellipses (right)
fig = plt.figure(figsize=(16, 7))

# === Left panel: 3D SPD cone visualization ===
ax1 = fig.add_subplot(121, projection='3d')

# Draw cone boundary surface: det = ac - b^2 = 0
a_range = np.linspace(0.1, 4, 30)
c_range = np.linspace(0.1, 4, 30)
A_grid, C_grid = np.meshgrid(a_range, c_range)
B_pos = np.sqrt(A_grid * C_grid)  # b = sqrt(ac) for det=0
B_neg = -np.sqrt(A_grid * C_grid)

ax1.plot_surface(A_grid, B_pos, C_grid, alpha=0.15, color='gray')
ax1.plot_surface(A_grid, B_neg, C_grid, alpha=0.15, color='gray')

# Plot data points by class
for S in X_class0:
    a, b, c = spd_to_3d_coords(S)
    ax1.scatter(a, b, c, c='blue', marker='o', s=50, alpha=0.7)
for S in X_class1:
    a, b, c = spd_to_3d_coords(S)
    ax1.scatter(a, b, c, c='green', marker='^', s=50, alpha=0.7)

# Plot class means as stars
a0, b0, c0 = spd_to_3d_coords(mean0)
a1, b1, c1 = spd_to_3d_coords(mean1)
ax1.scatter(a0, b0, c0, c='blue', marker='*', s=300, edgecolor='black', linewidth=1.5, label='Class 0 mean', zorder=5)
ax1.scatter(a1, b1, c1, c='green', marker='*', s=300, edgecolor='black', linewidth=1.5, label='Class 1 mean', zorder=5)

# Plot geodesic curve with many points for smoothness
t_fine = np.linspace(0, 1, 50)
geo_coords = []
for t in t_fine:
    G_t = airm_geodesic(A, B, torch.tensor(t))
    geo_coords.append(spd_to_3d_coords(G_t))
geo_coords = np.array(geo_coords)
ax1.plot(geo_coords[:, 0], geo_coords[:, 1], geo_coords[:, 2],
         'r-', linewidth=3, label='AIRM geodesic', zorder=10)

# Mark midpoint
mid = airm_geodesic(A, B, torch.tensor(0.5))
am, bm, cm = spd_to_3d_coords(mid)
ax1.scatter(am, bm, cm, c='red', marker='D', s=150, edgecolor='black', linewidth=1.5, label='Midpoint (t=0.5)', zorder=15)

ax1.set_xlabel('a (S[0,0])', fontsize=11)
ax1.set_ylabel('b (S[0,1])', fontsize=11)
ax1.set_zlabel('c (S[1,1])', fontsize=11)
ax1.set_title('3D SPD Cone with AIRM Geodesic', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)

# === Right panel: 2D ellipse interpolation ===
ax2 = fig.add_subplot(122)

# Plot start and end ellipses (filled)
x, y = spd_to_ellipse(A)
ax2.fill(x, y, alpha=0.3, color='blue')
ax2.plot(x, y, 'b-', linewidth=2, label='Start (Class 0 mean)')

x, y = spd_to_ellipse(B)
ax2.fill(x, y, alpha=0.3, color='green')
ax2.plot(x, y, 'g-', linewidth=2, label='End (Class 1 mean)')

# Plot intermediate geodesic ellipses
for t in [0.25, 0.5, 0.75]:
    G_t = airm_geodesic(A, B, torch.tensor(t))
    x, y = spd_to_ellipse(G_t)
    alpha_val = 0.5 if t == 0.5 else 0.25
    lw = 2.5 if t == 0.5 else 1.5
    color = 'red' if t == 0.5 else 'orange'
    label = f't={t} (midpoint)' if t == 0.5 else f't={t}'
    ax2.plot(x, y, color=color, linewidth=lw, linestyle='--', alpha=0.8, label=label)

ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-2.5, 2.5)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)
ax2.set_title('Ellipse Interpolation Along Geodesic', fontsize=12, fontweight='bold')
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y', fontsize=11)

plt.suptitle('AIRM Geodesics: Shortest Paths on the SPD Manifold', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()