"""
Generate an animated SPD cone manifold visualization for documentation background.

The space of 2x2 SPD matrices can be parameterized and visualized as a cone in 3D.
A 2x2 SPD matrix [[a, b], [b, c]] must satisfy: a > 0, c > 0, and det = ac - b² > 0
This forms a cone in (a, b, c) space.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap


# Set style for dark/elegant background
plt.style.use("dark_background")


def create_spd_cone_animation(
    output_path="spd_cone_animation.gif", n_frames=120, fps=30, dpi=150, figsize=(8, 8)
):
    """
    Create an animated visualization of the SPD cone manifold.

    The cone represents the space of 2x2 symmetric positive definite matrices.
    """

    # Create figure with transparent background option
    fig = plt.figure(figsize=figsize, facecolor="#0d1117")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0d1117")

    # Custom colormap - gradient from deep purple to cyan (matching docs theme)
    colors = ["#1a1a2e", "#16213e", "#0f3460", "#4a90a4", "#7fdbda", "#b8f3ff"]
    cmap = LinearSegmentedColormap.from_list("spd_cone", colors, N=256)

    # Alternative warm colormap
    warm_colors = ["#1a1a2e", "#2d1b4e", "#4a1c6e", "#7b2d8e", "#b84dae", "#f06dce"]
    cmap_warm = LinearSegmentedColormap.from_list("spd_warm", warm_colors, N=256)

    # Generate cone surface
    # For SPD: det = ac - b² > 0, so c > b²/a
    # Parameterize: a = r*cos(θ)², c = r*sin(θ)², b varies

    n_points = 80

    # Create cone using different parameterization
    # Use: x = trace, y = off-diagonal, z = sqrt(determinant)
    theta = np.linspace(0, 2 * np.pi, n_points)
    z_cone = np.linspace(0.01, 2.5, n_points)

    Theta, Z = np.meshgrid(theta, z_cone)

    # Cone surface: x² + y² = z² (simplified representation)
    R = Z * 0.8  # Radius grows with height
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Create geodesic curves on the manifold
    n_geodesics = 5
    geodesic_params = np.linspace(0, 2 * np.pi, n_geodesics, endpoint=False)

    # Sample points representing SPD matrices
    n_samples = 30
    np.random.seed(42)
    sample_z = np.random.uniform(0.5, 2.2, n_samples)
    sample_theta = np.random.uniform(0, 2 * np.pi, n_samples)
    sample_r = sample_z * 0.8 * np.random.uniform(0.3, 0.95, n_samples)
    sample_x = sample_r * np.cos(sample_theta)
    sample_y = sample_r * np.sin(sample_theta)

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()

        # Rotation angle
        angle = frame * (360 / n_frames)
        elevation = 20 + 10 * np.sin(2 * np.pi * frame / n_frames)

        ax.view_init(elev=elevation, azim=angle)

        # Pulsing effect
        pulse = 1 + 0.05 * np.sin(2 * np.pi * frame / 30)

        # Draw cone surface with transparency
        surf = ax.plot_surface(
            X * pulse,
            Y * pulse,
            Z * pulse,
            cmap=cmap,
            alpha=0.3,
            antialiased=True,
            linewidth=0,
            shade=True,
        )

        # Draw wireframe for structure
        ax.plot_wireframe(
            X * pulse,
            Y * pulse,
            Z * pulse,
            color="#4a90a4",
            alpha=0.15,
            linewidth=0.3,
            rstride=8,
            cstride=8,
        )

        # Draw geodesic-like spiral curves
        for i, phi in enumerate(geodesic_params):
            t = np.linspace(0.1, 2.4, 100)
            # Spiral geodesic
            spiral_phase = phi + frame * 0.02
            gx = t * 0.6 * np.cos(spiral_phase + t * 0.8) * pulse
            gy = t * 0.6 * np.sin(spiral_phase + t * 0.8) * pulse
            gz = t * pulse

            # Color gradient along geodesic
            alpha_geo = 0.6 + 0.3 * np.sin(2 * np.pi * frame / 60 + i)
            ax.plot(gx, gy, gz, color="#7fdbda", alpha=alpha_geo, linewidth=1.5)

        # Animate sample points (floating effect)
        float_offset = 0.1 * np.sin(2 * np.pi * frame / 40 + np.arange(n_samples) * 0.2)
        ax.scatter(
            sample_x * pulse,
            sample_y * pulse,
            (sample_z + float_offset) * pulse,
            c=sample_z,
            cmap=cmap_warm,
            s=30,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

        # Draw apex point
        ax.scatter(
            [0],
            [0],
            [0],
            c="#f06dce",
            s=100,
            alpha=0.9,
            marker="o",
            edgecolors="white",
            linewidths=2,
        )

        # Draw circular cross-sections at different heights
        for h in [0.8, 1.5, 2.2]:
            circle_theta = np.linspace(0, 2 * np.pi, 100)
            circle_r = h * 0.8 * pulse
            circle_x = circle_r * np.cos(circle_theta)
            circle_y = circle_r * np.sin(circle_theta)
            circle_z = np.ones_like(circle_theta) * h * pulse
            ax.plot(
                circle_x, circle_y, circle_z, color="#b8f3ff", alpha=0.4, linewidth=1
            )

        # Style settings
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_zlim(0, 3)

        # Remove axes for clean look
        ax.set_axis_off()

        # Set background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("none")
        ax.yaxis.pane.set_edgecolor("none")
        ax.zaxis.pane.set_edgecolor("none")

        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        return []

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=1000 / fps, blit=False
    )

    # Save as GIF
    print(f"Saving animation to {output_path}...")
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    print("Done!")

    plt.close(fig)
    return output_path


def create_simple_cone_loop(
    output_path="spd_cone_simple.gif", n_frames=60, fps=20, dpi=100, figsize=(6, 6)
):
    """
    Create a simpler, faster-loading cone animation suitable for web background.
    """

    fig = plt.figure(figsize=figsize, facecolor="none")
    ax = fig.add_subplot(111, projection="3d", facecolor="none")

    # Minimal colormap
    colors = ["#1e3a5f", "#3d6a8a", "#5c9ab5", "#7bcae0", "#9afaff"]
    cmap = LinearSegmentedColormap.from_list("spd_minimal", colors, N=256)

    # Cone surface
    n_points = 50
    theta = np.linspace(0, 2 * np.pi, n_points)
    z_cone = np.linspace(0.01, 2.0, n_points)
    Theta, Z = np.meshgrid(theta, z_cone)
    R = Z * 0.7
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    def animate(frame):
        ax.clear()
        angle = frame * (360 / n_frames)
        ax.view_init(elev=25, azim=angle)

        # Cone surface
        ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.4, antialiased=True, linewidth=0)

        # Wireframe
        ax.plot_wireframe(
            X, Y, Z, color="#5c9ab5", alpha=0.2, linewidth=0.3, rstride=5, cstride=5
        )

        # Geodesic lines
        for phi in np.linspace(0, 2 * np.pi, 4, endpoint=False):
            t = np.linspace(0.1, 1.9, 50)
            gx = t * 0.5 * np.cos(phi + t * 0.5)
            gy = t * 0.5 * np.sin(phi + t * 0.5)
            ax.plot(gx, gy, t, color="#9afaff", alpha=0.6, linewidth=1.2)

        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_zlim(0, 2.2)
        ax.set_axis_off()

        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("none")
        ax.yaxis.pane.set_edgecolor("none")
        ax.zaxis.pane.set_edgecolor("none")

        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=1000 / fps, blit=False
    )

    print(f"Saving simple animation to {output_path}...")
    writer = animation.PillowWriter(fps=fps)
    anim.save(
        output_path,
        writer=writer,
        dpi=dpi,
        savefig_kwargs={"transparent": True, "facecolor": "none"},
    )
    print("Done!")

    plt.close(fig)
    return output_path


if __name__ == "__main__":
    import os

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate both versions
    detailed_path = os.path.join(script_dir, "spd_cone_animation.gif")
    simple_path = os.path.join(script_dir, "spd_cone_simple.gif")

    print("Generating SPD cone manifold animations...")
    print("=" * 50)

    # Generate detailed version
    create_spd_cone_animation(output_path=detailed_path, n_frames=120, fps=30, dpi=150)

    # Generate simple/lightweight version
    create_simple_cone_loop(output_path=simple_path, n_frames=60, fps=20, dpi=100)

    print("=" * 50)
    print("Generated animations:")
    print(f"  - Detailed: {detailed_path}")
    print(f"  - Simple:   {simple_path}")
