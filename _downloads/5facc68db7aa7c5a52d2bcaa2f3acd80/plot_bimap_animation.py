"""
.. _bimap-animation:

BiMap Layer Animation
=====================

This animation visualizes how the BiMap layer transforms SPD matrices
on the Riemannian manifold. BiMap performs a bilinear mapping:

.. math::

    Y = W^T X W

where :math:`W` is constrained to be orthogonal (on the Stiefel manifold).

.. contents:: This visualization shows:
   :local:
   :depth: 2

"""

# sphinx_gallery_thumbnail_number = 1

######################################################################
# Understanding BiMap
# -------------------
#
# The BiMap layer performs dimension reduction (or expansion) while
# preserving the SPD property. If :math:`X` is a :math:`n \times n`
# SPD matrix and :math:`W` is a :math:`n \times m` orthogonal matrix,
# then :math:`Y = W^T X W` is an :math:`m \times m` SPD matrix.
#
# Key properties:
#
# 1. **Preserves positive definiteness**: If X is SPD, so is Y
# 2. **Preserves symmetry**: Y is symmetric if X is symmetric
# 3. **Orthogonal constraint**: W lies on the Stiefel manifold
#

# Add parent path for imports
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
    draw_eigenvalue_axes,
    setup_spd_plot,
)


######################################################################
# Setup and Data Generation
# -------------------------
#

np.random.seed(42)

# Input SPD matrices (3x3)
n_matrices = 6
input_matrices = []
for i in range(n_matrices):
    # Create 3x3 SPD matrices
    A = np.random.randn(3, 3)
    spd = A @ A.T + 0.5 * np.eye(3)
    input_matrices.append(spd)

# Orthogonal projection matrix W (3x2) - reduces dimension from 3 to 2
# This is on the Stiefel manifold V(2, 3)
W_init = np.random.randn(3, 2)
W, _ = np.linalg.qr(W_init)

# For visualization, we'll use 2x2 matrices (project to first 2 dims)
input_2d = [mat[:2, :2] for mat in input_matrices]

# Compute BiMap outputs: Y = W^T X W
output_matrices = [W.T @ mat @ W for mat in input_matrices]

# Colors for each matrix
colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_matrices))

# Centers for ellipses (arranged in circle)
angles = np.linspace(0, 2 * np.pi, n_matrices, endpoint=False)
radius = 2.0
centers_input = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
centers_output = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

print(f"Input dimension: {input_matrices[0].shape}")
print(f"W shape (Stiefel): {W.shape}")
print(f"Output dimension: {output_matrices[0].shape}")

######################################################################
# Static Visualization
# --------------------
#
# First, let's create a static visualization showing the transformation.
#

fig_static, axes_static = plt.subplots(1, 3, figsize=(15, 5))

# Before transformation
ax1 = axes_static[0]
setup_spd_plot(ax1, xlim=(-5, 5), ylim=(-5, 5), title="Before: Input SPD")
for i, inp in enumerate(input_2d):
    ellipse = create_ellipse_patch(
        inp,
        centers_input[i],
        alpha=0.7,
        color=colors[i],
        edgecolor="black",
        linewidth=2,
    )
    ax1.add_patch(ellipse)
    draw_eigenvalue_axes(ax1, inp, centers_input[i], color="darkred", scale=0.6)

# W transformation visualization
ax2 = axes_static[1]
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect("equal")
ax2.set_title(r"W: Orthogonal Projection ($W^TW = I$)", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)

# Draw unit circle (Stiefel manifold constraint)
theta = np.linspace(0, 2 * np.pi, 100)
ax2.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit circle")

# Draw W columns as vectors
ax2.arrow(
    0,
    0,
    W[0, 0],
    W[1, 0],
    head_width=0.1,
    head_length=0.05,
    fc=COLORS["input"],
    ec=COLORS["input"],
    linewidth=2,
    label="w₁",
)
ax2.arrow(
    0,
    0,
    W[0, 1],
    W[1, 1],
    head_width=0.1,
    head_length=0.05,
    fc=COLORS["output"],
    ec=COLORS["output"],
    linewidth=2,
    label="w₂",
)
ax2.legend(loc="upper right")
ax2.text(0, -1.3, "W^T W = I (orthogonal)", ha="center", fontsize=10)

# After transformation
ax3 = axes_static[2]
setup_spd_plot(ax3, xlim=(-5, 5), ylim=(-5, 5), title=r"After: $Y = W^T X W$")
for i, out in enumerate(output_matrices):
    ellipse = create_ellipse_patch(
        out,
        centers_output[i],
        alpha=0.7,
        color=colors[i],
        edgecolor="black",
        linewidth=2,
    )
    ax3.add_patch(ellipse)
    draw_eigenvalue_axes(ax3, out, centers_output[i], color="darkblue", scale=0.6)

plt.tight_layout()

######################################################################
# Mathematical Explanation
# ------------------------
#
# The BiMap operation :math:`Y = W^T X W` has important properties:
#
# 1. **Congruence transformation**: This is a congruence, which preserves
#    the positive definiteness of X.
#
# 2. **Eigenvalue relationship**: If :math:`X` has eigenvalues
#    :math:`\lambda_1, ..., \lambda_n`, then :math:`Y` has eigenvalues
#    that are "mixtures" of the original eigenvalues.
#
# 3. **Stiefel constraint**: W lies on the Stiefel manifold
#    :math:`\text{St}(m, n) = \{W \in \mathbb{R}^{n \times m} : W^T W = I_m\}`
#
# In the animation below, you can see how the ellipsoid shapes change but
# remain valid SPD matrices (ellipsoids, not hyperbolas).
#

######################################################################
# Animation
# ---------
#
# The animation shows the BiMap transformation in action.
# Watch how SPD matrices are transformed while preserving positive definiteness.

# Create animation figure
fig_anim, axes_anim = plt.subplots(1, 2, figsize=(14, 6))
fig_anim.suptitle(
    "BiMap: Bilinear Mapping on SPD Manifold", fontsize=14, fontweight="bold"
)

ax_input = axes_anim[0]
ax_output = axes_anim[1]

# Animation parameters
n_frames = 60
pause_frames = 15


def init():
    """Initialize animation."""
    ax_input.clear()
    ax_output.clear()

    setup_spd_plot(
        ax_input,
        xlim=(-5, 5),
        ylim=(-5, 5),
        title="Input SPD Matrices (3x3 → 2x2 view)",
    )
    setup_spd_plot(
        ax_output, xlim=(-5, 5), ylim=(-5, 5), title="Output: Y = W^T X W (2x2)"
    )

    ax_input.text(
        0,
        -4.5,
        "Original ellipsoids on manifold",
        ha="center",
        fontsize=10,
        style="italic",
    )
    ax_output.text(
        0, -4.5, "Transformed by orthogonal W", ha="center", fontsize=10, style="italic"
    )

    return []


def animate(frame):
    """Animation frame update."""
    ax_input.clear()
    ax_output.clear()

    setup_spd_plot(ax_input, xlim=(-5, 5), ylim=(-5, 5), title="Input SPD Matrices")
    setup_spd_plot(
        ax_output, xlim=(-5, 5), ylim=(-5, 5), title=r"Output: $Y = W^T X W$"
    )

    # Progress through animation
    if frame < pause_frames:
        # Show input only
        t = 0.0
        phase_text = "Input SPD matrices"
    elif frame < pause_frames + n_frames:
        # Animate transformation
        t = (frame - pause_frames) / n_frames
        phase_text = f"Applying BiMap (t={t:.2f})"
    else:
        # Show output only
        t = 1.0
        phase_text = "Transformed matrices"

    # Draw ellipses
    for i, (inp, out) in enumerate(zip(input_2d, output_matrices)):
        center_in = centers_input[i]
        center_out = centers_output[i]

        # Input ellipse (always visible)
        ellipse_in = create_ellipse_patch(
            inp,
            center_in,
            alpha=0.6 * (1 - 0.5 * t),
            color=colors[i],
            edgecolor="black",
            linewidth=2,
        )
        ax_input.add_patch(ellipse_in)

        # Draw eigenvalue axes for input
        if t < 0.5:
            draw_eigenvalue_axes(ax_input, inp, center_in, color="darkred", scale=0.8)

        # Interpolate between input and output
        if t > 0:
            # Smooth interpolation
            smooth_t = 0.5 * (1 - np.cos(np.pi * t))

            # Interpolate the SPD matrix
            interp_mat = (1 - smooth_t) * inp + smooth_t * out

            # Make sure it's valid SPD (should be, but enforce symmetry)
            interp_mat = 0.5 * (interp_mat + interp_mat.T)

            # Ensure positive eigenvalues
            eigvals, eigvecs = np.linalg.eigh(interp_mat)
            eigvals = np.maximum(eigvals, 0.01)
            interp_mat = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Output ellipse
            ellipse_out = create_ellipse_patch(
                interp_mat,
                center_out,
                alpha=0.6 * smooth_t + 0.3,
                color=colors[i],
                edgecolor="black",
                linewidth=2,
            )
            ax_output.add_patch(ellipse_out)

            # Draw eigenvalue axes for output
            if t > 0.5:
                draw_eigenvalue_axes(
                    ax_output, interp_mat, center_out, color="darkblue", scale=0.8
                )

    # Add transformation arrow
    if 0 < t < 1:
        ax_input.annotate(
            "",
            xy=(4.5, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=2),
        )

    # Phase indicator
    fig_anim.suptitle(f"BiMap: {phase_text}", fontsize=14, fontweight="bold")

    # Add W matrix visualization
    if t > 0:
        ax_output.text(
            0,
            4.2,
            "W ∈ St(2,3): orthogonal projection",
            ha="center",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    return []


# Create the animation - must be assigned to a variable that persists
total_frames = 2 * pause_frames + n_frames
anim = animation.FuncAnimation(
    fig_anim, animate, init_func=init, frames=total_frames, interval=50, blit=False
)

plt.tight_layout()
plt.show()
