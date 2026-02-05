"""
.. _reeig-animation:

ReEig Layer Animation
=====================

This animation visualizes how the ReEig (Rectified Eigenvalue) layer
introduces non-linearity while preserving the SPD property.

.. math::

    \\text{ReEig}(X) = U \\max(\\Lambda, \\epsilon) U^T

where :math:`X = U \\Lambda U^T` is the eigendecomposition.

.. contents:: This visualization shows:
   :local:
   :depth: 2

"""

# sphinx_gallery_thumbnail_number = 1

######################################################################
# Understanding ReEig
# -------------------
#
# ReEig applies a ReLU-like function to eigenvalues:
#
# - Eigenvalues above threshold :math:`\epsilon` are preserved
# - Eigenvalues below threshold are clamped to :math:`\epsilon`
#
# This ensures the output remains SPD while introducing non-linearity,
# similar to how ReLU introduces non-linearity in standard neural networks.
#

import sys

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D


# Handle both direct execution and import
try:
    _current_dir = Path(__file__).parent
except NameError:
    _current_dir = Path.cwd() / "examples" / "visualizations"

if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from spd_visualization_utils import create_ellipse_patch, setup_spd_plot


######################################################################
# Setup and Data Generation
# -------------------------
#

np.random.seed(42)

# Threshold for ReEig
epsilon = 0.3

# Create matrices where some eigenvalues are below threshold
n_matrices = 5

# Generate eigenvalues - some will be below threshold
eigval_sets = [
    np.array([0.1, 2.0]),  # First below threshold
    np.array([0.5, 1.5]),  # Both above
    np.array([0.05, 0.8]),  # First below
    np.array([1.0, 0.15]),  # Second below
    np.array([0.2, 0.25]),  # Both below threshold
]

# Random rotation matrices for each
rotation_angles = np.linspace(0, np.pi, n_matrices)
rotation_matrices = [
    np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in rotation_angles
]

# Create input matrices: X = U @ diag(eigvals) @ U.T
input_matrices = []
for eigvals, U in zip(eigval_sets, rotation_matrices):
    X = U @ np.diag(eigvals) @ U.T
    input_matrices.append(X)

# Apply ReEig: clamp eigenvalues to epsilon
output_matrices = []
for eigvals, U in zip(eigval_sets, rotation_matrices):
    clamped_eigvals = np.maximum(eigvals, epsilon)
    Y = U @ np.diag(clamped_eigvals) @ U.T
    output_matrices.append(Y)

# Colors
colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, n_matrices))

print(f"Threshold epsilon = {epsilon}")
for i, (inp, out) in enumerate(zip(eigval_sets, eigval_sets)):
    clamped = np.maximum(inp, epsilon)
    print(f"Matrix {i+1}: {inp} -> {clamped}")

######################################################################
# Static Visualization
# --------------------
#
# First, let's visualize the ReEig eigenvalue rectification function.
#

fig_static, axes_static = plt.subplots(1, 2, figsize=(14, 6))

# Eigenvalue rectification function
ax1 = axes_static[0]
x = np.linspace(0, 2.5, 200)
y_reeig = np.maximum(x, epsilon)

ax1.plot(x, x, "k--", alpha=0.4, label="Identity (y=x)", linewidth=2)
ax1.plot(x, y_reeig, "b-", linewidth=3, label=f"ReEig (epsilon={epsilon})")
ax1.fill_between(
    [0, epsilon],
    [epsilon, epsilon],
    [0, 0],
    color="red",
    alpha=0.15,
    label="Clamped region",
)
ax1.axhline(y=epsilon, color="red", linestyle="--", alpha=0.5)
ax1.axvline(x=epsilon, color="red", linestyle="--", alpha=0.5)

# Mark example eigenvalues
for i, eigvals in enumerate(eigval_sets):
    for ev in eigvals:
        out_ev = max(ev, epsilon)
        ax1.scatter(
            [ev],
            [out_ev],
            s=100,
            c=[colors[i]],
            edgecolors="black",
            linewidth=1.5,
            zorder=5,
        )
        if ev < epsilon:
            ax1.plot(
                [ev, ev], [ev, epsilon], color=colors[i], linestyle=":", linewidth=1.5
            )

ax1.set_xlim(-0.1, 2.5)
ax1.set_ylim(-0.1, 2.5)
ax1.set_xlabel("Input eigenvalue lambda", fontsize=12)
ax1.set_ylabel("Output eigenvalue max(lambda, epsilon)", fontsize=12)
ax1.set_title(
    "ReEig: Eigenvalue Rectification Function", fontsize=13, fontweight="bold"
)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect("equal")

# Before/After ellipses
ax2 = axes_static[1]
setup_spd_plot(
    ax2, xlim=(-4, 4), ylim=(-3.5, 3.5), title="ReEig Effect on SPD Matrices"
)

# Positions for ellipses
y_positions = np.linspace(2, -2, n_matrices)
centers = [(0, y) for y in y_positions]

# Draw input and output side by side
x_offset = 1.5
for i, (inp, out, center) in enumerate(zip(input_matrices, output_matrices, centers)):
    # Input (left)
    center_in = (center[0] - x_offset, center[1])
    ellipse_in = create_ellipse_patch(
        inp, center_in, alpha=0.5, color=colors[i], edgecolor="black", linewidth=2
    )
    ax2.add_patch(ellipse_in)

    # Arrow
    ax2.annotate(
        "",
        xy=(center[0] + x_offset - 0.8, center[1]),
        xytext=(center[0] - x_offset + 0.8, center[1]),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )

    # Output (right)
    center_out = (center[0] + x_offset, center[1])
    ellipse_out = create_ellipse_patch(
        out, center_out, alpha=0.5, color=colors[i], edgecolor="black", linewidth=2
    )
    ax2.add_patch(ellipse_out)

ax2.text(-x_offset, 3.2, "Input", ha="center", fontsize=11, fontweight="bold")
ax2.text(x_offset, 3.2, "ReEig(Input)", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()

######################################################################
# Mathematical Explanation
# ------------------------
#
# ReEig introduces non-linearity while preserving SPD structure:
#
# 1. **Eigendecomposition**: :math:`X = U \Lambda U^T`
# 2. **Rectification**: :math:`\Lambda' = \max(\Lambda, \epsilon I)`
# 3. **Reconstruction**: :math:`Y = U \Lambda' U^T`
#
# Key properties:
#
# - **Preserves eigenvectors**: Only eigenvalues change
# - **Non-expansive**: :math:`\|Y\|_F \geq \|X\|_F`
# - **Gradient flow**: Proper backprop through eigendecomposition
#

######################################################################
# Animation
# ---------
#
# The animation shows the ReEig eigenvalue rectification in action.

# Create animation figure
fig_anim = plt.figure(figsize=(16, 6))

# Three subplots: eigenvalue plot, input ellipse, output ellipse
ax_eigen = fig_anim.add_subplot(1, 3, 1)
ax_input = fig_anim.add_subplot(1, 3, 2)
ax_output = fig_anim.add_subplot(1, 3, 3)

# Animation parameters
n_frames = 50
pause_frames = 20


def draw_eigenvalue_plot(ax, t, highlight_idx=None):
    """Draw eigenvalue rectification plot."""
    ax.clear()
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.2, 2.5)
    ax.set_xlabel("Input eigenvalue lambda", fontsize=11)
    ax.set_ylabel("Output eigenvalue", fontsize=11)
    ax.set_title("ReEig: Eigenvalue Rectification", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Draw threshold line
    ax.axhline(y=epsilon, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.axvline(x=epsilon, color="red", linestyle="--", linewidth=2, alpha=0.7)

    # Shade clamped region
    ax.fill_between([0, epsilon], [epsilon, epsilon], [0, 0], color="red", alpha=0.1)

    # Draw identity line (no change)
    x_line = np.linspace(0, 3, 100)
    ax.plot(x_line, x_line, "k--", alpha=0.3, label="Identity")

    # Draw ReEig function
    y_reeig = np.maximum(x_line, epsilon)
    ax.plot(x_line, y_reeig, "b-", linewidth=3, label=f"ReEig(epsilon={epsilon})")

    # Plot eigenvalues for each matrix
    for i, eigvals in enumerate(eigval_sets):
        for j, ev in enumerate(eigvals):
            marker_size = (
                150 if (highlight_idx is not None and i == highlight_idx) else 80
            )
            output_ev = max(ev, epsilon)

            # Interpolate position during animation
            current_y = ev + t * (output_ev - ev)

            ax.scatter(
                [ev],
                [current_y],
                s=marker_size,
                c=[colors[i]],
                edgecolors="black",
                linewidth=1.5,
                zorder=5,
            )

            # Draw vertical arrow showing rectification
            if ev < epsilon and t > 0:
                ax.annotate(
                    "",
                    xy=(ev, current_y),
                    xytext=(ev, ev),
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5),
                )

    # Labels
    ax.text(epsilon + 0.05, 0.05, f"epsilon={epsilon}", fontsize=10, color="red")
    ax.legend(loc="lower right", fontsize=9)


def animate(frame):
    """Animation frame update."""
    # Progress
    if frame < pause_frames:
        t = 0.0
        phase = "Input eigenvalues"
    elif frame < pause_frames + n_frames:
        t = (frame - pause_frames) / n_frames
        t = 0.5 * (1 - np.cos(np.pi * t))  # Smooth easing
        phase = f"Applying ReEig (t={t:.2f})"
    else:
        t = 1.0
        phase = "Rectified eigenvalues"

    # Update eigenvalue plot
    draw_eigenvalue_plot(ax_eigen, t)

    # Update input ellipses
    ax_input.clear()
    setup_spd_plot(ax_input, xlim=(-3, 3), ylim=(-3.5, 3.5), title="Input SPD Matrices")

    for i, (inp, center) in enumerate(zip(input_matrices, centers)):
        ellipse = create_ellipse_patch(
            inp,
            center,
            alpha=0.6,
            color=colors[i],
            edgecolor="black",
            linewidth=2,
        )
        ax_input.add_patch(ellipse)

        # Label eigenvalues
        eigvals = eigval_sets[i]
        label = f"lambda=[{eigvals[0]:.2f}, {eigvals[1]:.2f}]"
        ax_input.text(
            center[0] + 1.8, center[1], label, fontsize=9, va="center", color=colors[i]
        )

        # Mark if eigenvalues are below threshold
        if np.any(eigvals < epsilon):
            ax_input.plot(center[0], center[1], "rx", markersize=12, mew=2)

    # Update output ellipses
    ax_output.clear()
    setup_spd_plot(
        ax_output,
        xlim=(-3, 3),
        ylim=(-3.5, 3.5),
        title=r"Output: $U \max(\Lambda, \epsilon) U^T$",
    )

    for i, (inp, out, center) in enumerate(
        zip(input_matrices, output_matrices, centers)
    ):
        # Interpolate
        interp = (1 - t) * inp + t * out

        ellipse = create_ellipse_patch(
            interp,
            center,
            alpha=0.6,
            color=colors[i],
            edgecolor="black",
            linewidth=2,
        )
        ax_output.add_patch(ellipse)

        # Label eigenvalues
        inp_eigvals = eigval_sets[i]
        out_eigvals = np.maximum(inp_eigvals, epsilon)
        current_eigvals = (1 - t) * inp_eigvals + t * out_eigvals
        label = f"lambda=[{current_eigvals[0]:.2f}, {current_eigvals[1]:.2f}]"
        ax_output.text(
            center[0] + 1.8, center[1], label, fontsize=9, va="center", color=colors[i]
        )

        # Mark if clamping occurred
        if np.any(inp_eigvals < epsilon):
            ax_output.plot(center[0], center[1], "g*", markersize=12, mew=2)

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Eigenvalue < epsilon",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="green",
            markersize=12,
            label="Rectified",
        ),
    ]
    ax_output.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig_anim.suptitle(f"ReEig Layer — {phase}", fontsize=14, fontweight="bold")

    return []


def init():
    """Initialize animation."""
    return []


# Create the animation - must be assigned to a variable that persists
total_frames = 2 * pause_frames + n_frames
anim = animation.FuncAnimation(
    fig_anim, animate, init_func=init, frames=total_frames, interval=50, blit=False
)

plt.tight_layout()
plt.show()
