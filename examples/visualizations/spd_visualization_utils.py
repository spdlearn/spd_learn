"""
Utilities for visualizing SPD matrices and manifold operations.

SPD matrices can be visualized as ellipsoids since a 2x2 SPD matrix defines
an ellipse (its level set). This module provides tools for creating
animations that show how SPD operations transform these ellipsoids.
"""

from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse


def spd_to_ellipse_params(
    spd_matrix: np.ndarray,
) -> Tuple[float, float, float]:
    """Convert a 2x2 SPD matrix to ellipse parameters.

    The ellipse is defined by the level set {x: x^T A^{-1} x = 1}.

    Parameters
    ----------
    spd_matrix : np.ndarray
        A 2x2 SPD matrix.

    Returns
    -------
    tuple
        (width, height, angle_degrees) of the ellipse.
    """
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(spd_matrix)

    # Width and height are sqrt of eigenvalues (for x^T A^{-1} x = 1)
    width = 2 * np.sqrt(eigvals[1])
    height = 2 * np.sqrt(eigvals[0])

    # Angle from the first eigenvector
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

    return width, height, angle


def create_ellipse_patch(
    spd_matrix: np.ndarray,
    center: Tuple[float, float] = (0, 0),
    alpha: float = 0.5,
    color: str = "blue",
    edgecolor: str = "black",
    linewidth: float = 1.5,
) -> Ellipse:
    """Create a matplotlib Ellipse patch from a 2x2 SPD matrix.

    Parameters
    ----------
    spd_matrix : np.ndarray
        A 2x2 SPD matrix.
    center : tuple
        Center of the ellipse.
    alpha : float
        Transparency.
    color : str
        Fill color.
    edgecolor : str
        Edge color.
    linewidth : float
        Edge width.

    Returns
    -------
    Ellipse
        A matplotlib Ellipse patch.
    """
    width, height, angle = spd_to_ellipse_params(spd_matrix)
    return Ellipse(
        center,
        width,
        height,
        angle=angle,
        alpha=alpha,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )


def generate_random_spd(n: int = 2, condition_number: float = 5.0) -> np.ndarray:
    """Generate a random SPD matrix with controlled condition number.

    Parameters
    ----------
    n : int
        Matrix dimension.
    condition_number : float
        Desired condition number.

    Returns
    -------
    np.ndarray
        Random SPD matrix.
    """
    # Random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))

    # Eigenvalues with controlled condition number
    eigvals = np.linspace(1.0, condition_number, n)

    return Q @ np.diag(eigvals) @ Q.T


def generate_spd_batch(
    batch_size: int,
    n: int = 2,
    condition_range: Tuple[float, float] = (1.5, 5.0),
) -> np.ndarray:
    """Generate a batch of random SPD matrices.

    Parameters
    ----------
    batch_size : int
        Number of matrices.
    n : int
        Matrix dimension.
    condition_range : tuple
        Range of condition numbers.

    Returns
    -------
    np.ndarray
        Batch of SPD matrices with shape (batch_size, n, n).
    """
    matrices = []
    for _ in range(batch_size):
        cond = np.random.uniform(*condition_range)
        matrices.append(generate_random_spd(n, cond))
    return np.stack(matrices)


def interpolate_spd_linear(A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between SPD matrices (not geodesic).

    Parameters
    ----------
    A, B : np.ndarray
        SPD matrices.
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    np.ndarray
        Interpolated matrix (may not be exactly SPD for all t).
    """
    return (1 - t) * A + t * B


def interpolate_spd_geodesic(A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
    """Geodesic interpolation between SPD matrices (AIRM).

    Parameters
    ----------
    A, B : np.ndarray
        SPD matrices.
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    np.ndarray
        Point on the geodesic from A to B.
    """
    # A^{1/2}
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    A_sqrt = eigvecs_A @ np.diag(np.sqrt(eigvals_A)) @ eigvecs_A.T
    A_inv_sqrt = eigvecs_A @ np.diag(1.0 / np.sqrt(eigvals_A)) @ eigvecs_A.T

    # A^{-1/2} B A^{-1/2}
    M = A_inv_sqrt @ B @ A_inv_sqrt

    # M^t
    eigvals_M, eigvecs_M = np.linalg.eigh(M)
    M_t = eigvecs_M @ np.diag(np.power(eigvals_M, t)) @ eigvecs_M.T

    # A^{1/2} M^t A^{1/2}
    return A_sqrt @ M_t @ A_sqrt


def frechet_mean(matrices: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """Compute the Fréchet mean of SPD matrices.

    Parameters
    ----------
    matrices : np.ndarray
        Batch of SPD matrices with shape (batch_size, n, n).
    n_iter : int
        Number of iterations.

    Returns
    -------
    np.ndarray
        Fréchet mean matrix.
    """
    mean = matrices.mean(axis=0)

    for _ in range(n_iter):
        # Map to tangent space at mean
        eigvals, eigvecs = np.linalg.eigh(mean)
        mean_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        mean_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        tangent_vectors = []
        for mat in matrices:
            M = mean_inv_sqrt @ mat @ mean_inv_sqrt
            eigvals_M, eigvecs_M = np.linalg.eigh(M)
            log_M = eigvecs_M @ np.diag(np.log(eigvals_M)) @ eigvecs_M.T
            tangent_vectors.append(log_M)

        # Average in tangent space
        avg_tangent = np.mean(tangent_vectors, axis=0)

        # Map back to manifold
        eigvals_T, eigvecs_T = np.linalg.eigh(avg_tangent)
        exp_T = eigvecs_T @ np.diag(np.exp(eigvals_T)) @ eigvecs_T.T
        mean = mean_sqrt @ exp_T @ mean_sqrt

    return mean


def setup_spd_plot(
    ax: plt.Axes,
    xlim: Tuple[float, float] = (-4, 4),
    ylim: Tuple[float, float] = (-4, 4),
    title: str = "",
) -> None:
    """Setup axes for SPD visualization.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    xlim, ylim : tuple
        Axis limits.
    title : str
        Plot title.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


def draw_eigenvalue_axes(
    ax: plt.Axes,
    spd_matrix: np.ndarray,
    center: Tuple[float, float] = (0, 0),
    color: str = "red",
    scale: float = 1.0,
) -> None:
    """Draw eigenvector axes scaled by eigenvalues.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    spd_matrix : np.ndarray
        A 2x2 SPD matrix.
    center : tuple
        Center point.
    color : str
        Arrow color.
    scale : float
        Scale factor for arrows.
    """
    eigvals, eigvecs = np.linalg.eigh(spd_matrix)

    for i in range(2):
        vec = eigvecs[:, i] * np.sqrt(eigvals[i]) * scale
        ax.arrow(
            center[0],
            center[1],
            vec[0],
            vec[1],
            head_width=0.1,
            head_length=0.08,
            fc=color,
            ec=color,
            linewidth=2,
            alpha=0.8,
        )


def create_manifold_grid(
    n_points: int = 5,
    eigval_range: Tuple[float, float] = (0.5, 3.0),
) -> List[np.ndarray]:
    """Create a grid of SPD matrices for visualization.

    Parameters
    ----------
    n_points : int
        Number of points along each eigenvalue axis.
    eigval_range : tuple
        Range of eigenvalues.

    Returns
    -------
    list
        List of 2x2 SPD matrices.
    """
    eigvals = np.linspace(eigval_range[0], eigval_range[1], n_points)
    matrices = []

    for e1 in eigvals:
        for e2 in eigvals:
            matrices.append(np.diag([e1, e2]))

    return matrices


# Color maps for different operations
COLORS = {
    "input": "#3498db",  # Blue
    "output": "#e74c3c",  # Red
    "intermediate": "#2ecc71",  # Green
    "mean": "#9b59b6",  # Purple
    "tangent": "#f39c12",  # Orange
    "identity": "#34495e",  # Dark gray
}


def save_animation(
    anim: animation.FuncAnimation,
    filename: str,
    fps: int = 30,
    dpi: int = 100,
) -> None:
    """Save animation to file.

    Parameters
    ----------
    anim : FuncAnimation
        Animation object.
    filename : str
        Output filename (supports .mp4, .gif).
    fps : int
        Frames per second.
    dpi : int
        Resolution.
    """
    if filename.endswith(".gif"):
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps)

    anim.save(filename, writer=writer, dpi=dpi)
    print(f"Animation saved to {filename}")
