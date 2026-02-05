"""
.. _batchnorm-animation:

SPD Batch Normalization Animation
=================================

This animation visualizes how SPD Batch Normalization centers data
around the Fréchet (geometric) mean on the Riemannian manifold.

.. math::

    \\tilde{X}_i = \\mathcal{G}^{-1/2} X_i \\mathcal{G}^{-1/2}

where :math:`\\mathcal{G}` is the Fréchet mean of the batch.

.. contents:: This visualization shows:
   :local:
   :depth: 2

"""

# sphinx_gallery_thumbnail_number = 1

######################################################################
# Understanding SPD Batch Normalization
# -------------------------------------
#
# Unlike Euclidean batch normalization (subtract mean, divide by std),
# SPD batch normalization uses Riemannian geometry:
#
# 1. Compute the **Fréchet mean** (geometric mean on the manifold)
# 2. **Center** by multiplying with inverse square root of mean
# 3. Optionally apply **learnable bias** (re-center to another point)
#
# The Fréchet mean is computed iteratively using the Karcher flow.
#

import sys

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# Handle both direct execution and import
try:
    _current_dir = Path(__file__).parent
except NameError:
    _current_dir = Path.cwd() / "examples" / "visualizations"

if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from spd_visualization_utils import (
    COLORS,
    create_ellipse_patch,
    frechet_mean,
    setup_spd_plot,
)


######################################################################
# Setup and Data Generation
# -------------------------
#

np.random.seed(42)

batch_size = 8

# Create clustered SPD matrices (simulating a batch with structure)
# Cluster center
center_eigvals = np.array([1.5, 2.5])
center_angle = np.pi / 6
U_center = np.array(
    [
        [np.cos(center_angle), -np.sin(center_angle)],
        [np.sin(center_angle), np.cos(center_angle)],
    ]
)
center_matrix = U_center @ np.diag(center_eigvals) @ U_center.T

# Generate batch around this center with variations
batch_matrices = []
for i in range(batch_size):
    # Perturb eigenvalues
    eigvals = center_eigvals + np.random.randn(2) * 0.4
    eigvals = np.maximum(eigvals, 0.3)  # Ensure positive

    # Perturb rotation
    angle = center_angle + np.random.randn() * 0.3
    U = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    X = U @ np.diag(eigvals) @ U.T
    batch_matrices.append(X)

batch_matrices = np.stack(batch_matrices)

# Compute Fréchet mean
mean_matrix = frechet_mean(batch_matrices, n_iter=15)

# Compute normalized matrices: Y = mean^{-1/2} @ X @ mean^{-1/2}
eigvals_mean, eigvecs_mean = np.linalg.eigh(mean_matrix)
mean_inv_sqrt = eigvecs_mean @ np.diag(1.0 / np.sqrt(eigvals_mean)) @ eigvecs_mean.T

normalized_matrices = []
for X in batch_matrices:
    Y = mean_inv_sqrt @ X @ mean_inv_sqrt
    normalized_matrices.append(Y)

normalized_matrices = np.stack(normalized_matrices)

# Compute new Fréchet mean (should be close to identity)
new_mean = frechet_mean(normalized_matrices, n_iter=10)

# Colors and positions
colors = plt.cm.tab10(np.linspace(0, 1, batch_size))

np.random.seed(42)
positions_before = np.random.randn(batch_size, 2) * 0.8
positions_before[:, 0] += 0  # Center around origin
positions_before[:, 1] += 0

# After normalization, positions centered around origin
positions_after = positions_before - positions_before.mean(axis=0)

print("Fréchet mean eigenvalues:", np.linalg.eigvalsh(mean_matrix))
print("Identity:", np.eye(2))
print("New mean eigenvalues (should be ~1):", np.linalg.eigvalsh(new_mean))

######################################################################
# Static Visualization
# --------------------
#
# First, let's see the before and after of SPD batch normalization.
#

fig_static, axes_static = plt.subplots(1, 3, figsize=(16, 5))

# Before
ax1 = axes_static[0]
setup_spd_plot(ax1, xlim=(-4, 4), ylim=(-4, 4), title="Before: Input Batch")
for i, (X, pos) in enumerate(zip(batch_matrices, positions_before)):
    ellipse = create_ellipse_patch(
        X, tuple(pos), alpha=0.6, color=colors[i], edgecolor="black", linewidth=1.5
    )
    ax1.add_patch(ellipse)
ellipse_mean = create_ellipse_patch(
    mean_matrix,
    (0, 0),
    alpha=0.4,
    color=COLORS["mean"],
    edgecolor=COLORS["mean"],
    linewidth=3,
)
ax1.add_patch(ellipse_mean)
ax1.text(0, -3.5, "Fréchet Mean (purple)", ha="center", fontsize=10)

# Transformation diagram
ax2 = axes_static[1]
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.set_aspect("equal")
ax2.axis("off")
ax2.set_title("Normalization Operation", fontsize=12, fontweight="bold")

# Draw transformation
ax2.annotate(
    "",
    xy=(1.5, 0),
    xytext=(-1.5, 0),
    arrowprops=dict(arrowstyle="->", color="gray", lw=3),
)
ax2.text(
    0,
    0.3,
    r"$\tilde{X} = G^{-1/2} X G^{-1/2}$",
    ha="center",
    fontsize=14,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)
ax2.text(-1.5, -0.5, "Mean: G", ha="center", fontsize=11, color=COLORS["mean"])
ax2.text(1.5, -0.5, "Mean: I", ha="center", fontsize=11, color="darkgoldenrod")

# After
ax3 = axes_static[2]
setup_spd_plot(ax3, xlim=(-4, 4), ylim=(-4, 4), title="After: Normalized Batch")
for i, (X_norm, pos) in enumerate(zip(normalized_matrices, positions_after)):
    ellipse = create_ellipse_patch(
        X_norm, tuple(pos), alpha=0.6, color=colors[i], edgecolor="black", linewidth=1.5
    )
    ax3.add_patch(ellipse)
ellipse_id = create_ellipse_patch(
    np.eye(2), (0, 0), alpha=0.4, color="gold", edgecolor="gold", linewidth=3
)
ax3.add_patch(ellipse_id)
ax3.text(0, -3.5, "New Mean = Identity (gold)", ha="center", fontsize=10)

plt.tight_layout()

######################################################################
# Fréchet Mean Convergence
# ------------------------
#
# Let's visualize how the Karcher flow converges to the Fréchet mean.
#

fig_conv, ax_conv = plt.subplots(figsize=(8, 6))

# Show Karcher flow convergence
n_iter = 15
means_history = [batch_matrices.mean(axis=0)]  # Start with arithmetic mean

current_mean = means_history[0]
for _ in range(n_iter):
    eigvals, eigvecs = np.linalg.eigh(current_mean)
    mean_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    mean_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Compute tangent vectors
    tangent_sum = np.zeros((2, 2))
    for X in batch_matrices:
        M = mean_inv_sqrt @ X @ mean_inv_sqrt
        eigvals_M, eigvecs_M = np.linalg.eigh(M)
        log_M = eigvecs_M @ np.diag(np.log(eigvals_M)) @ eigvecs_M.T
        tangent_sum += log_M

    tangent_avg = tangent_sum / batch_size

    # Exponential map
    eigvals_T, eigvecs_T = np.linalg.eigh(tangent_avg)
    exp_T = eigvecs_T @ np.diag(np.exp(eigvals_T)) @ eigvecs_T.T
    current_mean = mean_sqrt @ exp_T @ mean_sqrt

    means_history.append(current_mean)

# Plot convergence
iterations = range(len(means_history))
eigval1 = [np.linalg.eigvalsh(m)[0] for m in means_history]
eigval2 = [np.linalg.eigvalsh(m)[1] for m in means_history]

ax_conv.plot(
    iterations, eigval1, "b-o", label="lambda_1 (smaller)", linewidth=2, markersize=6
)
ax_conv.plot(
    iterations, eigval2, "r-s", label="lambda_2 (larger)", linewidth=2, markersize=6
)
ax_conv.axhline(
    y=np.linalg.eigvalsh(mean_matrix)[0], color="b", linestyle="--", alpha=0.5
)
ax_conv.axhline(
    y=np.linalg.eigvalsh(mean_matrix)[1], color="r", linestyle="--", alpha=0.5
)

ax_conv.set_xlabel("Karcher Flow Iteration", fontsize=12)
ax_conv.set_ylabel("Eigenvalue", fontsize=12)
ax_conv.set_title(
    "Fréchet Mean Convergence (Karcher Flow)", fontsize=13, fontweight="bold"
)
ax_conv.legend(loc="right", fontsize=10)
ax_conv.grid(True, alpha=0.3)

plt.tight_layout()

######################################################################
# Mathematical Explanation
# ------------------------
#
# SPD Batch Normalization performs:
#
# 1. **Fréchet Mean**: :math:`\mathcal{G} = \arg\min_G \sum_i d^2(G, X_i)`
#    where :math:`d` is the geodesic distance
#
# 2. **Centering**: :math:`\tilde{X}_i = \mathcal{G}^{-1/2} X_i \mathcal{G}^{-1/2}`
#
# 3. **Learnable Bias** (optional): :math:`Y_i = B^{1/2} \tilde{X}_i B^{1/2}`
#
# Key differences from Euclidean batch norm:
#
# - Uses geometric (Fréchet) mean instead of arithmetic mean
# - Centering via matrix multiplication, not subtraction
# - Variance scaling via matrix power, not division
#

######################################################################
# Animation
# ---------
#
# The animation shows SPD Batch Normalization centering the data
# around the identity matrix.

# Create animation figure
fig_anim, axes_anim = plt.subplots(1, 3, figsize=(16, 5))
ax_before = axes_anim[0]
ax_mean = axes_anim[1]
ax_after = axes_anim[2]

# Animation parameters
n_frames = 50
pause_frames = 25


def animate(frame):
    """Animation frame update."""
    # Clear all axes
    ax_before.clear()
    ax_mean.clear()
    ax_after.clear()

    # Progress
    if frame < pause_frames:
        t = 0.0
        phase = "Before normalization"
    elif frame < pause_frames + n_frames:
        t = (frame - pause_frames) / n_frames
        t = 0.5 * (1 - np.cos(np.pi * t))  # Smooth easing
        phase = f"Computing Fréchet mean & centering (t={t:.2f})"
    else:
        t = 1.0
        phase = "After normalization"

    # --- Before plot ---
    setup_spd_plot(ax_before, xlim=(-4, 4), ylim=(-4, 4), title="Input Batch (SPD)")

    for i, (X, pos) in enumerate(zip(batch_matrices, positions_before)):
        ellipse = create_ellipse_patch(
            X,
            tuple(pos),
            alpha=0.5,
            color=colors[i],
            edgecolor="black",
            linewidth=1.5,
        )
        ax_before.add_patch(ellipse)

    # Draw Fréchet mean
    ellipse_mean = create_ellipse_patch(
        mean_matrix,
        (0, 0),
        alpha=0.3,
        color=COLORS["mean"],
        edgecolor=COLORS["mean"],
        linewidth=3,
    )
    ax_before.add_patch(ellipse_mean)
    ax_before.text(
        0, -3.5, "Purple: Fréchet Mean", ha="center", fontsize=9, color=COLORS["mean"]
    )

    # --- Mean computation visualization ---
    ax_mean.set_xlim(-3, 3)
    ax_mean.set_ylim(-3, 3)
    ax_mean.set_aspect("equal")
    ax_mean.set_title("Fréchet Mean Computation", fontsize=12, fontweight="bold")
    ax_mean.grid(True, alpha=0.3)

    # Draw geodesics from each point to mean
    for i, (X, pos) in enumerate(zip(batch_matrices, positions_before)):
        # Draw line from point to mean (geodesic approximation)
        ax_mean.plot(
            [pos[0], 0],
            [pos[1], 0],
            color=colors[i],
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

        # Point
        ax_mean.scatter(
            [pos[0]], [pos[1]], s=80, c=[colors[i]], edgecolors="black", linewidth=1
        )

    # Mean at center
    ax_mean.scatter(
        [0],
        [0],
        s=200,
        c=COLORS["mean"],
        marker="*",
        edgecolors="black",
        linewidth=2,
        zorder=10,
    )
    ax_mean.text(0.2, 0.2, "G", fontsize=12, fontweight="bold", color=COLORS["mean"])

    # Formula
    ax_mean.text(
        0,
        -2.5,
        r"$\mathcal{G} = \arg\min_G \sum_i d^2(G, X_i)$",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # --- After plot ---
    setup_spd_plot(
        ax_after,
        xlim=(-4, 4),
        ylim=(-4, 4),
        title=r"Normalized: $\tilde{X}_i = G^{-1/2} X_i G^{-1/2}$",
    )

    # Identity at center
    identity = np.eye(2)
    ellipse_id = create_ellipse_patch(
        identity,
        (0, 0),
        alpha=0.2 + 0.3 * t,
        color="gold",
        edgecolor="gold",
        linewidth=3,
    )
    ax_after.add_patch(ellipse_id)
    if t > 0.5:
        ax_after.text(
            0,
            0,
            "I",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="darkgoldenrod",
        )

    for i, (X_orig, X_norm, pos_orig, pos_after) in enumerate(
        zip(batch_matrices, normalized_matrices, positions_before, positions_after)
    ):
        # Interpolate matrix
        interp_mat = (1 - t) * X_orig + t * X_norm

        # Interpolate position (move toward centered)
        interp_pos = (1 - t) * pos_orig + t * pos_after

        ellipse = create_ellipse_patch(
            interp_mat,
            tuple(interp_pos),
            alpha=0.5,
            color=colors[i],
            edgecolor="black",
            linewidth=1.5,
        )
        ax_after.add_patch(ellipse)

    if t > 0.5:
        ax_after.text(
            0,
            -3.5,
            "New mean = Identity (gold circle)",
            ha="center",
            fontsize=9,
            color="darkgoldenrod",
        )

    fig_anim.suptitle(
        f"SPD Batch Normalization — {phase}", fontsize=14, fontweight="bold"
    )

    return []


def init():
    """Initialize animation."""
    return []


# Create the animation - must be assigned to a variable that persists
total_frames = 2 * pause_frames + n_frames
anim = animation.FuncAnimation(
    fig_anim, animate, init_func=init, frames=total_frames, interval=60, blit=False
)

plt.tight_layout()
plt.show()
