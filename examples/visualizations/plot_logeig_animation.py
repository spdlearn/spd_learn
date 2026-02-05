"""
.. _logeig-animation:

LogEig: Linearization and the Swelling Effect
=============================================

This animation visualizes the "Swelling Effect"—a phenomenon where the
Euclidean average of SPD matrices results in a matrix with a larger
volume (determinant) than the originals. We show how the ``LogEig``
layer (via Log-Euclidean interpolation) prevents this by linearizing
the manifold.

.. math::

    \\text{Euclidean: } M_{Euc}(t) = (1-t)A + tB

    \\text{Log-Euclidean: } M_{LE}(t) = \\exp((1-t)\\log A + t\\log B)

.. contents:: This visualization shows:
   :local:
   :depth: 2
"""

# sphinx_gallery_thumbnail_number = 1

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.gridspec import GridSpec
from spd_visualization_utils import create_ellipse_patch, setup_spd_plot

from spd_learn.functional import matrix_exp, matrix_log


# Mathematical Utils using spd_learn
def log_matrix(X):
    """Compute matrix logarithm using spd_learn library.

    Small wrapper to handle both single and batch inputs.
    and convert back to numpy, to make our life easier.
    """
    X_torch = torch.from_numpy(X).to(torch.float64)
    if X_torch.ndim == 2:
        X_torch = X_torch.unsqueeze(0)
    # matrix_log is the autograd Function class, we call .apply()
    res = matrix_log.apply(X_torch)
    return res.squeeze(0).detach().numpy()


def exp_matrix(X):
    """Compute matrix exponential using spd_learn library.

    Small wrapper to handle both single and batch inputs.
    and convert back to numpy, to make our life easier.
    """
    X_torch = torch.from_numpy(X).to(torch.float64)
    if X_torch.ndim == 2:
        X_torch = X_torch.unsqueeze(0)
    res = matrix_exp.apply(X_torch)
    return res.squeeze(0).detach().numpy()


######################################################################
# The Swelling Effect
# -------------------
#
# When we average two SPD matrices with different orientations, the
# Euclidean mean "swells". For example, the average of two orthogonal
# narrow ellipses is a large circle. This is undesirable in neural
# networks as it artificially inflates feature volumes.
#
# LogEig projects matrices to the tangent space where they can be
# averaged without swelling.
#

np.random.seed(42)

# Create two highly different SPD matrices (unit determinant)
# Matrix A: Vertical narrow ellipse
A = np.diag([0.2, 5.0])
# Matrix B: Horizontal narrow ellipse
B = np.diag([5.0, 0.2])


def euclidean_interp(A, B, t):
    return (1 - t) * A + t * B


def log_euclidean_interp(A, B, t):
    logA = log_matrix(A)
    logB = log_matrix(B)
    return exp_matrix((1 - t) * logA + t * logB)


# Pre-calculate paths for metrics
n_steps = 100
t_values = np.linspace(0, 1, n_steps)
det_euc = [np.linalg.det(euclidean_interp(A, B, t)) for t in t_values]
det_le = [np.linalg.det(log_euclidean_interp(A, B, t)) for t in t_values]

######################################################################
# Animation Setup
# ---------------
#

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, width_ratios=[1.2, 1], figure=fig)

# Axes for Ellipses
ax_euc_ell = fig.add_subplot(gs[0, 0])
ax_le_ell = fig.add_subplot(gs[1, 0])

# Axes for Determinant Chart
ax_det = fig.add_subplot(gs[:, 1])

# Colors
COLOR_EUC = "#e74c3c"  # Red
COLOR_LE = "#2ecc71"  # Green
COLOR_ORIG = "#3498db"  # Blue


def setup_axes():
    setup_spd_plot(
        ax_euc_ell, xlim=(-3, 3), ylim=(-3, 3), title="Euclidean Path (Swelling!)"
    )
    setup_spd_plot(
        ax_le_ell,
        xlim=(-3, 3),
        ylim=(-3, 3),
        title="Log-Euclidean Path (LogEig - Correct)",
    )

    ax_det.clear()
    ax_det.set_title("Volume (Determinant) over Path", fontweight="bold", fontsize=12)
    ax_det.set_xlabel("Interpolation Parameter (t)", fontsize=10)
    ax_det.set_ylabel("Determinant det(X)", fontsize=10)
    ax_det.plot(
        t_values, det_euc, color=COLOR_EUC, linestyle="--", alpha=0.3, label="Euclidean"
    )
    ax_det.plot(
        t_values,
        det_le,
        color=COLOR_LE,
        linestyle="--",
        alpha=0.3,
        label="Log-Euclidean",
    )
    ax_det.set_ylim(0, max(det_euc) * 1.2)
    ax_det.grid(True, alpha=0.3)
    ax_det.legend(loc="upper left")


# Animation parameters
n_frames = 120
pause_frames = 20


def animate(frame):
    # Determine t with easing and pauses
    if frame < pause_frames:
        t = 0
    elif frame < pause_frames + n_frames:
        t = (frame - pause_frames) / n_frames
        t = 0.5 * (1 - np.cos(np.pi * t))  # Smooth easing
    else:
        t = 1.0

    # Clear ellipse axes
    ax_euc_ell.clear()
    ax_le_ell.clear()
    setup_spd_plot(
        ax_euc_ell, xlim=(-3, 3), ylim=(-3, 3), title="Euclidean Interpolation"
    )
    setup_spd_plot(
        ax_le_ell,
        xlim=(-3, 3),
        ylim=(-3, 3),
        title="Log-Euclidean (LogEig) Interpolation",
    )

    # Matrices at t
    Mt_euc = euclidean_interp(A, B, t)
    Mt_le = log_euclidean_interp(A, B, t)

    # Draw Euclidean Ellipse
    e_euc = create_ellipse_patch(Mt_euc, color=COLOR_EUC, alpha=0.7, linewidth=2)
    ax_euc_ell.add_patch(e_euc)
    ax_euc_ell.text(
        0,
        -2.5,
        f"det={np.linalg.det(Mt_euc):.2f}",
        ha="center",
        fontweight="bold",
        color=COLOR_EUC,
    )

    # Draw Reference Outlines
    for mat in [A, B]:
        outline = create_ellipse_patch(mat, color="gray", alpha=0.2, linewidth=1)
        ax_euc_ell.add_patch(outline)

    # Draw Log-Euclidean Ellipse
    e_le = create_ellipse_patch(Mt_le, color=COLOR_LE, alpha=0.7, linewidth=2)
    ax_le_ell.add_patch(e_le)
    ax_le_ell.text(
        0,
        -2.5,
        f"det={np.linalg.det(Mt_le):.2f}",
        ha="center",
        fontweight="bold",
        color=COLOR_LE,
    )

    for mat in [A, B]:
        outline = create_ellipse_patch(mat, color="gray", alpha=0.2, linewidth=1)
        ax_le_ell.add_patch(outline)

    # Update Determinant lines
    idx = int(t * (n_steps - 1))
    # We clear and redraw the lines to avoid accumulation issues if using blit=True
    # but here we just plot over
    ax_det.plot(t_values[: idx + 1], det_euc[: idx + 1], color=COLOR_EUC, linewidth=3)
    ax_det.plot(t_values[: idx + 1], det_le[: idx + 1], color=COLOR_LE, linewidth=3)

    # Current points
    ax_det.scatter([t], [det_euc[idx]], color=COLOR_EUC, s=50, edgecolors="k", zorder=5)
    ax_det.scatter([t], [det_le[idx]], color=COLOR_LE, s=50, edgecolors="k", zorder=5)

    if t > 0.4 and t < 0.6:
        ax_euc_ell.text(
            0,
            2,
            "SWELLING!",
            color=COLOR_EUC,
            fontsize=14,
            fontweight="black",
            ha="center",
        )

    fig.suptitle(
        f"The Swelling Effect: Why LogEig is Essential\nPath Progress: {t*100:.0f}%",
        fontsize=16,
        fontweight="bold",
    )

    return []


setup_axes()
total_frames = n_frames + 2 * pause_frames
anim = animation.FuncAnimation(
    fig, animate, frames=total_frames, interval=50, blit=False
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
