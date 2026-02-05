"""
Generate SPD cone manifold animation with smiling face for documentation.
Matches the spd_learn logo style with PyTorch flame.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation


def create_cone_with_face(
    output_path="spd_cone_dark.gif",
    n_frames=90,
    fps=24,
    dpi=150,
    figsize=(8, 8),
    include_flame=True,
    light_mode=False,
):
    """
    Create SPD cone animation with smiling face and PyTorch flame, matching logo style.

    Parameters
    ----------
    output_path : str
        Path to save the GIF
    n_frames : int
        Number of frames in the animation
    fps : int
        Frames per second
    dpi : int
        Resolution of the output
    figsize : tuple
        Figure size in inches
    include_flame : bool
        Whether to include the PyTorch flame
    light_mode : bool
        If True, use white background for light theme (multiply blend mode)
        If False, use black background for dark theme (screen blend mode)
    """

    # Background color based on mode
    # Black for dark mode (screen blend makes black transparent)
    # White for light mode (multiply blend makes white transparent)
    bg_color = "#ffffff" if light_mode else "#000000"
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax = fig.add_subplot(111, projection="3d", facecolor=bg_color)

    # Logo colors
    cone_dark = "#1e2a3a"
    cone_light = "#4a5a6a"
    grid_color = "#5a6a7a"
    face_color = "#ffffff"
    flame_color = "#ee4c2c"  # PyTorch orange

    # Cone geometry
    n_theta = 100
    n_z = 60

    theta = np.linspace(0, 2 * np.pi, n_theta)
    z_vals = np.linspace(0, 2.5, n_z)
    Theta, Z = np.meshgrid(theta, z_vals)

    # Cone shape - slightly wider to match logo
    cone_slope = 0.75
    R = (2.5 - Z) * cone_slope
    X_base = R * np.cos(Theta)
    Y_base = R * np.sin(Theta)

    # Tilt to match logo orientation
    tilt_angle = np.radians(15)
    lean_angle = np.radians(10)

    def apply_tilt(x, y, z):
        """Apply rotation to match logo tilt."""
        y_tilted = y * np.cos(tilt_angle) - z * np.sin(tilt_angle)
        z_tilted = y * np.sin(tilt_angle) + z * np.cos(tilt_angle)
        x_leaned = x * np.cos(lean_angle) + z_tilted * np.sin(lean_angle)
        z_final = -x * np.sin(lean_angle) + z_tilted * np.cos(lean_angle)
        return x_leaned, y_tilted, z_final

    def draw_flame(ax, center_x, center_y, center_z, scale=0.4, rotation=0):
        """Draw a PyTorch-style flame."""
        # Flame shape points (simplified torch flame)
        t = np.linspace(0, 2 * np.pi, 50)

        # Outer flame shape
        outer_r = scale * (0.5 + 0.3 * np.sin(t * 1.5) + 0.1 * np.sin(t * 3))
        outer_x = outer_r * np.sin(t) * 0.6
        outer_y = outer_r * np.cos(t) * 0.8 + scale * 0.3

        # Inner cutout (the hole in the flame)
        inner_t = np.linspace(0, 2 * np.pi, 30)
        inner_r = scale * 0.25
        inner_x = inner_r * np.sin(inner_t) * 0.7
        inner_y = inner_r * np.cos(inner_t) * 0.9 - scale * 0.1

        # Apply rotation
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)

        # Rotate outer flame
        ox_rot = outer_x * cos_r - outer_y * sin_r
        oy_rot = outer_x * sin_r + outer_y * cos_r

        # Plot outer flame
        ax.plot(
            center_x + ox_rot,
            [center_y] * len(ox_rot),
            center_z + oy_rot,
            color=flame_color,
            linewidth=3,
            alpha=0.95,
            zorder=20,
        )

        # Fill the flame with scatter points for 3D effect
        fill_t = np.linspace(0, 2 * np.pi, 100)
        for r_factor in np.linspace(0.3, 1.0, 8):
            fr = scale * (0.5 + 0.3 * np.sin(fill_t * 1.5)) * r_factor
            fx = fr * np.sin(fill_t) * 0.6
            fy = fr * np.cos(fill_t) * 0.8 + scale * 0.3 * r_factor
            fx_rot = fx * cos_r - fy * sin_r
            fy_rot = fx * sin_r + fy * cos_r
            ax.scatter(
                center_x + fx_rot,
                [center_y] * len(fx_rot),
                center_z + fy_rot,
                c=flame_color,
                s=8,
                alpha=0.3,
                edgecolors="none",
                zorder=19,
            )

    def animate(frame):
        ax.clear()

        progress = frame / n_frames
        rotation_angle = 360 * progress

        ax.view_init(elev=20, azim=rotation_angle + 35)

        # Subtle breathing effect
        breath = 1.0 + 0.008 * np.sin(2 * np.pi * progress * 2)

        # Apply transformations
        X_tilted, Y_tilted, Z_tilted = apply_tilt(
            X_base * breath, Y_base * breath, Z * breath
        )

        # Create gradient colors for cone surface
        colors = np.zeros((*Z.shape, 4))
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                # Lighting based on angle to camera
                angle_factor = (
                    np.cos(Theta[i, j] - np.radians(rotation_angle + 45)) + 1
                ) / 2
                # Darker at top, lighter at base
                height_factor = 0.7 + 0.3 * (Z[i, j] / 2.5)
                shade = angle_factor * 0.4 + 0.3

                dark_rgb = np.array([0x1E, 0x2A, 0x3A]) / 255
                light_rgb = np.array([0x3D, 0x4E, 0x5E]) / 255
                colors[i, j, :3] = dark_rgb + shade * (light_rgb - dark_rgb)
                colors[i, j, 3] = 0.98

        # Draw cone surface
        ax.plot_surface(
            X_tilted,
            Y_tilted,
            Z_tilted,
            facecolors=colors,
            antialiased=True,
            linewidth=0,
            shade=False,
            rcount=50,
            ccount=50,
        )

        # Horizontal grid rings
        ring_theta = np.linspace(0, 2 * np.pi, 100)
        for h in [0.6, 1.2, 1.8]:
            ring_r = (2.5 - h) * cone_slope * breath
            rx = ring_r * np.cos(ring_theta)
            ry = ring_r * np.sin(ring_theta)
            rz = np.full_like(ring_theta, h * breath)
            rx_t, ry_t, rz_t = apply_tilt(rx, ry, rz)
            ax.plot(rx_t, ry_t, rz_t, color=grid_color, linewidth=1.2, alpha=0.5)

        # Vertical meridian lines
        meridian_z = np.linspace(0, 2.48, 50) * breath
        for i in range(8):
            m_theta = 2 * np.pi * i / 8
            m_r = (2.5 * breath - meridian_z) * cone_slope
            mx = m_r * np.cos(m_theta)
            my = m_r * np.sin(m_theta)
            mx_t, my_t, mz_t = apply_tilt(mx, my, meridian_z)
            ax.plot(mx_t, my_t, mz_t, color=grid_color, linewidth=1.0, alpha=0.4)

        # Base ellipse
        base_r = 2.5 * cone_slope * breath
        bx = base_r * np.cos(ring_theta)
        by = base_r * np.sin(ring_theta)
        bz = np.zeros_like(ring_theta)
        bx_t, by_t, bz_t = apply_tilt(bx, by, bz)
        ax.plot(bx_t, by_t, bz_t, color=grid_color, linewidth=2.0, alpha=0.7)

        # ===== SMILING FACE =====
        # Use 2D annotation overlay to avoid matplotlib 3D depth sorting issues
        # Position face in the center-front area of the visible cone

        # Calculate a point on the visible front of the cone (centered)
        face_height = 0.65 * breath  # Lower on the cone (more centered)
        face_r_local = (2.5 * breath - face_height) * cone_slope * 0.65

        # Face theta tracks with camera to always be on the visible side
        camera_azimuth_rad = np.radians(rotation_angle + 35)
        face_theta = camera_azimuth_rad + np.pi

        face_x_local = face_r_local * np.cos(face_theta)
        face_y_local = face_r_local * np.sin(face_theta)
        face_x, face_y, face_z = apply_tilt(face_x_local, face_y_local, face_height)

        # Convert 3D position to 2D screen coordinates
        from mpl_toolkits.mplot3d import proj3d

        # Get the 2D projection of the face center
        x2d, y2d, _ = proj3d.proj_transform(face_x, face_y, face_z, ax.get_proj())

        # Convert to figure coordinates
        face_2d_x, face_2d_y = ax.transData.transform((x2d, y2d))
        # Convert to axes fraction
        face_ax_x = (face_2d_x - ax.bbox.x0) / ax.bbox.width
        face_ax_y = (face_2d_y - ax.bbox.y0) / ax.bbox.height

        # Eye parameters in axes fraction
        eye_offset_x = 0.038  # Horizontal spacing
        eye_offset_y = 0.022  # Vertical offset above center

        # Draw eyes as white filled circles using annotation
        for dx in [-eye_offset_x, eye_offset_x]:
            ax.annotate(
                "●",
                xy=(face_ax_x + dx, face_ax_y + eye_offset_y),
                xycoords="axes fraction",
                fontsize=18,
                color=face_color,
                ha="center",
                va="center",
                zorder=1000,
            )

        # Draw smile as a curved arc (upward curving = happy smile)
        smile_width = 0.055
        smile_x_pts = np.linspace(-smile_width, smile_width, 25)
        # For a smile: ends should be higher than center
        # y = A * x^2 gives center=0, ends=positive (ends higher)
        smile_y_pts = 0.012 * (smile_x_pts / smile_width) ** 2

        # Draw smile below the eyes
        smile_center_y = face_ax_y - 0.035
        for i in range(len(smile_x_pts) - 1):
            sx1 = face_ax_x + smile_x_pts[i]
            sy1 = smile_center_y + smile_y_pts[i]
            sx2 = face_ax_x + smile_x_pts[i + 1]
            sy2 = smile_center_y + smile_y_pts[i + 1]
            ax.annotate(
                "",
                xy=(sx2, sy2),
                xytext=(sx1, sy1),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color=face_color, lw=3.5),
                zorder=1000,
            )

        # ===== PYTORCH FLAME (2D overlay - simple solid flame without hole) =====
        if include_flame:
            # Position flame above the cone apex
            apex_x, apex_y, apex_z = apply_tilt(0, 0, 2.5 * breath)

            # Get 2D projection of apex position
            apex_2d_x, apex_2d_y, _ = proj3d.proj_transform(
                apex_x, apex_y, apex_z, ax.get_proj()
            )
            apex_screen_x, apex_screen_y = ax.transData.transform(
                (apex_2d_x, apex_2d_y)
            )
            flame_base_x = (apex_screen_x - ax.bbox.x0) / ax.bbox.width
            flame_base_y = (apex_screen_y - ax.bbox.y0) / ax.bbox.height

            # Offset flame position (directly above apex)
            flame_center_x = flame_base_x
            flame_center_y = flame_base_y + 0.08

            # Subtle flame flicker animation
            flicker_x = 0.002 * np.sin(2 * np.pi * progress * 2)
            flicker_scale = 1.0 + 0.015 * np.sin(2 * np.pi * progress * 3)
            flame_center_x += flicker_x

            # Simple solid flame - teardrop shape
            scale = 0.06 * flicker_scale

            # Flame outline points - simple teardrop
            n_pts = 40
            t = np.linspace(0, 2 * np.pi, n_pts)

            # Simple teardrop shape
            outer_x = scale * 0.4 * np.sin(t)
            outer_y = scale * (0.6 * np.cos(t) + 0.35 * np.cos(t) ** 2) + scale * 0.2

            # Draw flame outline
            for i in range(len(outer_x) - 1):
                ax.annotate(
                    "",
                    xy=(
                        flame_center_x + outer_x[i + 1],
                        flame_center_y + outer_y[i + 1],
                    ),
                    xytext=(flame_center_x + outer_x[i], flame_center_y + outer_y[i]),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color=flame_color, lw=3),
                    zorder=1001,
                )

            # Fill flame solidly with orange dots (no hole - simpler and cleaner)
            for r_mult in np.linspace(0.15, 0.9, 10):
                fill_x = outer_x * r_mult
                fill_y = (outer_y - scale * 0.2) * r_mult + scale * 0.2
                for j in range(0, len(fill_x), 2):
                    ax.annotate(
                        "●",
                        xy=(flame_center_x + fill_x[j], flame_center_y + fill_y[j]),
                        xycoords="axes fraction",
                        fontsize=int(5 * r_mult + 2),
                        color=flame_color,
                        alpha=0.9,
                        ha="center",
                        va="center",
                        zorder=1000,
                    )

        # View settings
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-2.8, 2.8)
        ax.set_zlim(-0.5, 3.8)
        ax.set_axis_off()

        # Remove axis panes
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor("none")

        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return []

    print(f"Generating {n_frames} frames at {fps} fps...")
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=1000 / fps, blit=False
    )

    print(f"Saving to {output_path}...")
    writer = animation.PillowWriter(fps=fps, metadata={"loop": 0})
    anim.save(output_path, writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"Done: {output_path}")
    return output_path


def create_minimal_cone(
    output_path="spd_cone_minimal.gif", n_frames=60, fps=20, dpi=100, figsize=(6, 6)
):
    """
    Create a minimal version without flame for smaller file size.
    """
    return create_cone_with_face(
        output_path=output_path,
        n_frames=n_frames,
        fps=fps,
        dpi=dpi,
        figsize=figsize,
        include_flame=False,
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("SPD Cone with Smiling Face Generator")
    print("=" * 60)

    # Generate dark mode version (black background, for screen blend mode)
    output_path = os.path.join(script_dir, "spd_cone_dark.gif")
    create_cone_with_face(
        output_path=output_path,
        n_frames=72,
        fps=24,
        dpi=120,
        figsize=(6, 6),
        include_flame=True,
        light_mode=False,
    )

    # Generate light mode version (white background, for multiply blend mode)
    light_path = os.path.join(script_dir, "spd_cone_light.gif")
    create_cone_with_face(
        output_path=light_path,
        n_frames=72,
        fps=24,
        dpi=120,
        figsize=(6, 6),
        include_flame=True,
        light_mode=True,
    )

    # Generate minimal dark mode version without flame
    minimal_path = os.path.join(script_dir, "spd_cone_minimal.gif")
    create_cone_with_face(
        output_path=minimal_path,
        n_frames=60,
        fps=20,
        dpi=100,
        figsize=(5, 5),
        include_flame=False,
        light_mode=False,
    )

    # Generate minimal light mode version without flame
    minimal_light_path = os.path.join(script_dir, "spd_cone_minimal_light.gif")
    create_cone_with_face(
        output_path=minimal_light_path,
        n_frames=60,
        fps=20,
        dpi=100,
        figsize=(5, 5),
        include_flame=False,
        light_mode=True,
    )

    print("=" * 60)
    print(f"Generated: {output_path}")
    print(f"Generated: {light_path}")
    print(f"Generated: {minimal_path}")
    print(f"Generated: {minimal_light_path}")
    print("=" * 60)
