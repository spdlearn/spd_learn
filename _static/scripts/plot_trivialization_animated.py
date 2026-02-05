"""
Animated Trivialization Visualization - Improved Version.

Key improvements:
1. Better manifold representation (smooth blob like in the paper)
2. Correct phase 3 logic - trajectory continues from where phase 1 ended
3. More fluid animation transitions
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path


def create_smooth_blob(center=(0, 0), scale=(1.5, 1.0), n_points=100, seed=42):
    """
    Create a smooth, organic blob shape like in the paper.
    Uses Fourier-based perturbations for natural-looking curves.
    """
    np.random.seed(seed)
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Base ellipse
    x = scale[0] * np.cos(t)
    y = scale[1] * np.sin(t)

    # Add smooth perturbations using low-frequency Fourier components
    perturbation = (
        0.12 * np.sin(2 * t + 0.5)
        + 0.08 * np.cos(3 * t + 1.2)
        + 0.05 * np.sin(4 * t + 0.3)
        + 0.03 * np.cos(5 * t)
    )

    # Apply perturbation radially
    r = 1 + perturbation
    x = x * r + center[0]
    y = y * r + center[1]

    # Close the path
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    return x, y


def create_filled_blob(
    ax, center, scale, color="lightblue", edge_color="black", alpha=0.3, lw=2, zorder=1
):
    """Create a filled blob with smooth edges."""
    x, y = create_smooth_blob(center, scale)

    # Create path for filled shape
    vertices = np.column_stack([x, y])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(x) - 2) + [Path.CLOSEPOLY]
    path = Path(vertices, codes)

    # Draw filled patch
    patch = PathPatch(
        path, facecolor=color, edgecolor=edge_color, alpha=alpha, lw=lw, zorder=zorder
    )
    ax.add_patch(patch)

    return patch


def get_color_gradient(
    n_points, start_color=(0.3, 0.5, 0.9), end_color=(0.9, 0.3, 0.3)
):
    """Generate smooth color gradient."""
    colors = []
    for i in range(n_points):
        t = i / (n_points - 1) if n_points > 1 else 0
        r = start_color[0] + (end_color[0] - start_color[0]) * t
        g = start_color[1] + (end_color[1] - start_color[1]) * t
        b = start_color[2] + (end_color[2] - start_color[2]) * t
        colors.append((r, g, b))
    return colors


class ImprovedDynamicTrivializationAnimation:
    def __init__(self, scale=1.0):
        # Use a nice style
        plt.style.use("default")

        # Scale factor for making everything bigger
        self.scale = scale

        self.fig, self.ax = plt.subplots(
            1, 1, figsize=(16 * scale, 9 * scale), facecolor="white"
        )
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.patch.set_facecolor("white")

        # Configuration
        self.n_points_per_phase = 5

        # Color schemes - blue to purple for phase 1, purple to red for phase 2
        self.colors_phase1 = get_color_gradient(
            self.n_points_per_phase,
            start_color=(0.3, 0.5, 0.95),  # Blue
            end_color=(0.6, 0.3, 0.7),  # Purple
        )
        self.colors_phase2 = get_color_gradient(
            self.n_points_per_phase,
            start_color=(0.7, 0.3, 0.6),  # Purple (continues from phase 1)
            end_color=(0.95, 0.3, 0.3),  # Red
        )

        # Manifold center (positioned for compact layout without title)
        self.manifold_center = (5.5, 5.8)

        # Tangent space origins
        self.ts1_origin = (0.8, 0.8)
        self.ts2_origin = (7.2, 0.8)
        self.ts_width = 3.2
        self.ts_height = 2.8

        # Define all point positions
        self.setup_points()

        # Draw static elements
        self.draw_static_elements()

        # Initialize dynamic elements
        self.dynamic_elements = {}
        self.setup_dynamic_elements()

        # Animation state
        self.total_frames = 100

    def setup_points(self):
        """Define all point positions - ensuring trajectory continuity."""

        # Points in tangent space 1 (y_{i,0} to y_{i,4})
        # Diagonal trajectory from origin
        self.tangent_pts_1 = []
        for i in range(self.n_points_per_phase):
            t = i / (self.n_points_per_phase - 1)
            x = self.ts1_origin[0] + 0.15 + t * 1.8
            y = self.ts1_origin[1] + 0.12 + t * 1.6
            self.tangent_pts_1.append((x, y))

        # Points in tangent space 2 (y_{i+1,0} to y_{i+1,4})
        # Similar diagonal trajectory, but in the second tangent space
        self.tangent_pts_2 = []
        for i in range(self.n_points_per_phase):
            t = i / (self.n_points_per_phase - 1)
            x = self.ts2_origin[0] + 0.15 + t * 1.8
            y = self.ts2_origin[1] + 0.12 + t * 1.5
            self.tangent_pts_2.append((x, y))

        # Points on manifold for phase 1
        # These trace a curved path on the manifold surface
        self.manifold_pts_1 = []
        for i in range(self.n_points_per_phase):
            t = i / (self.n_points_per_phase - 1)
            # Curved trajectory on the manifold (inside the blob)
            x = self.manifold_center[0] - 1.0 + t * 1.2
            y = self.manifold_center[1] + 0.2 + 0.15 * np.sin(t * np.pi)
            self.manifold_pts_1.append((x, y))

        # Points on manifold for phase 2
        # KEY FIX: Start from where phase 1 ended!
        # p_{i+1} = φ_{p_i}(y_{i,4}) = last point of phase 1 on manifold
        start_pt = self.manifold_pts_1[-1]  # This is p_{i+1}

        self.manifold_pts_2 = []
        for i in range(self.n_points_per_phase):
            t = i / (self.n_points_per_phase - 1)
            # Continue the trajectory from where phase 1 ended
            x = start_pt[0] + t * 1.1
            y = start_pt[1] + 0.1 - 0.2 * t  # Slight curve downward
            self.manifold_pts_2.append((x, y))

    def draw_static_elements(self):
        """Draw elements that don't change during animation."""
        s = self.scale  # Shorthand for scale

        # Title removed - it's redundant with the documentation page title

        # Manifold - smooth filled blob
        create_filled_blob(
            self.ax,
            center=self.manifold_center,
            scale=(1.8, 1.1),
            color="#E8F4FD",  # Light blue fill
            edge_color="#2C3E50",  # Dark edge
            alpha=0.5,
            lw=3 * s,
            zorder=1,
        )

        # Manifold label
        self.ax.text(
            self.manifold_center[0],
            self.manifold_center[1] - 0.5,
            r"$\mathcal{M}$",
            fontsize=30 * s,
            style="italic",
            ha="center",
            va="center",
            fontweight="bold",
        )

        # =====================================================================
        # Left tangent space T_{p_i}M
        # =====================================================================
        # Draw axes with nice arrows
        arrow_style = dict(
            arrowstyle="->", color="#34495E", lw=2.5 * s, mutation_scale=18 * s
        )

        self.ax.annotate(
            "",
            xy=(self.ts1_origin[0] + self.ts_width, self.ts1_origin[1]),
            xytext=self.ts1_origin,
            arrowprops=arrow_style,
        )
        self.ax.annotate(
            "",
            xy=(self.ts1_origin[0], self.ts1_origin[1] + self.ts_height),
            xytext=self.ts1_origin,
            arrowprops=arrow_style,
        )

        # Label
        self.ax.text(
            self.ts1_origin[0] + self.ts_width / 2,
            self.ts1_origin[1] - 0.5,
            r"$T_{p_i}\mathcal{M} \cong \mathbb{R}^n$",
            fontsize=18 * s,
            ha="center",
            va="top",
        )

        # Point labels (semi-transparent, will be highlighted when active)
        labels_1 = [
            r"$y_{i,0}$",
            r"$y_{i,1}$",
            r"$y_{i,2}$",
            r"$y_{i,3}$",
            r"$y_{i,4}$",
        ]
        for i, (pt, label) in enumerate(zip(self.tangent_pts_1, labels_1)):
            self.ax.text(
                pt[0] - 0.25,
                pt[1] + 0.18,
                label,
                fontsize=14 * s,
                ha="right",
                va="bottom",
                color="#7F8C8D",
                alpha=0.6,
            )

        # =====================================================================
        # Right tangent space T_{p_{i+1}}M
        # =====================================================================
        self.ax.annotate(
            "",
            xy=(self.ts2_origin[0] + self.ts_width, self.ts2_origin[1]),
            xytext=self.ts2_origin,
            arrowprops=arrow_style,
        )
        self.ax.annotate(
            "",
            xy=(self.ts2_origin[0], self.ts2_origin[1] + self.ts_height),
            xytext=self.ts2_origin,
            arrowprops=arrow_style,
        )

        # Label
        self.ax.text(
            self.ts2_origin[0] + self.ts_width / 2 + 0.3,
            self.ts2_origin[1] - 0.5,
            r"$T_{p_{i+1}}\mathcal{M} \cong \mathbb{R}^n$",
            fontsize=18 * s,
            ha="center",
            va="top",
        )

        # Point labels
        labels_2 = [
            r"$y_{i+1,0}$",
            r"$y_{i+1,1}$",
            r"$y_{i+1,2}$",
            r"$y_{i+1,3}$",
            r"$y_{i+1,4}$",
        ]
        for i, (pt, label) in enumerate(zip(self.tangent_pts_2, labels_2)):
            ha = "left" if i >= 2 else "right"
            offset_x = 0.15 if i >= 2 else -0.15
            self.ax.text(
                pt[0] + offset_x,
                pt[1] + 0.18,
                label,
                fontsize=13 * s,
                ha=ha,
                va="bottom",
                color="#7F8C8D",
                alpha=0.6,
            )

        # Base point labels on manifold
        self.ax.text(
            self.manifold_pts_1[0][0] - 0.2,
            self.manifold_pts_1[0][1] + 0.35,
            r"$p_i$",
            fontsize=18 * s,
            ha="center",
            va="bottom",
            color="#2980B9",
        )
        self.ax.text(
            self.manifold_pts_2[0][0] + 0.25,
            self.manifold_pts_2[0][1] + 0.35,
            r"$p_{i+1}$",
            fontsize=18 * s,
            ha="center",
            va="bottom",
            color="#8E44AD",
        )

        # Set limits (compact layout without title)
        self.ax.set_xlim(-0.8, 11.5)
        self.ax.set_ylim(-0.5, 7.5)

    def setup_dynamic_elements(self):
        """Setup placeholders for dynamic elements."""
        s = self.scale

        # Scatter plots for points (bigger)
        self.dynamic_elements["tangent_scatter_1"] = self.ax.scatter(
            [], [], s=180 * s, zorder=10, edgecolors="white", linewidths=2 * s
        )
        self.dynamic_elements["tangent_scatter_2"] = self.ax.scatter(
            [], [], s=180 * s, zorder=10, edgecolors="white", linewidths=2 * s
        )
        self.dynamic_elements["manifold_scatter_1"] = self.ax.scatter(
            [], [], s=150 * s, zorder=10, edgecolors="white", linewidths=1.5 * s
        )
        self.dynamic_elements["manifold_scatter_2"] = self.ax.scatter(
            [], [], s=150 * s, zorder=10, edgecolors="white", linewidths=1.5 * s
        )

        # Trajectory lines (thicker)
        (self.dynamic_elements["traj_line_1"],) = self.ax.plot(
            [], [], "-", color="#3498DB", lw=3 * s, alpha=0.5, zorder=9
        )
        (self.dynamic_elements["traj_line_2"],) = self.ax.plot(
            [], [], "-", color="#9B59B6", lw=3 * s, alpha=0.5, zorder=9
        )
        (self.dynamic_elements["manifold_line_1"],) = self.ax.plot(
            [], [], "-", color="#3498DB", lw=3.5 * s, alpha=0.6, zorder=9
        )
        (self.dynamic_elements["manifold_line_2"],) = self.ax.plot(
            [], [], "-", color="#9B59B6", lw=3.5 * s, alpha=0.6, zorder=9
        )

        # Arrows (will be recreated each frame)
        self.dynamic_elements["phi1_arrow"] = None
        self.dynamic_elements["phi2_arrow"] = None
        self.dynamic_elements["transition_arrow"] = None
        self.dynamic_elements["transition_line"] = None

        # Text elements (bigger fonts)
        self.dynamic_elements["phi1_label"] = self.ax.text(
            2.0, 5.0, "", fontsize=22 * s, ha="center", va="center", fontweight="bold"
        )
        self.dynamic_elements["phi2_label"] = self.ax.text(
            9.0, 5.0, "", fontsize=22 * s, ha="center", va="center", fontweight="bold"
        )
        self.dynamic_elements["transition_label"] = self.ax.text(
            5.5, 1.7, "", fontsize=16 * s, ha="center", va="bottom"
        )

        # Status text (bigger)
        self.dynamic_elements["status"] = self.ax.text(
            5.5,
            -0.5,
            "",
            fontsize=15 * s,
            ha="center",
            va="top",
            style="italic",
            color="#7F8C8D",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#BDC3C7",
                alpha=0.8,
            ),
        )

    def clear_arrows(self):
        """Remove all dynamic arrows."""
        for key in ["phi1_arrow", "phi2_arrow", "transition_arrow"]:
            if self.dynamic_elements[key] is not None:
                self.dynamic_elements[key].remove()
                self.dynamic_elements[key] = None

        if self.dynamic_elements["transition_line"] is not None:
            self.dynamic_elements["transition_line"].remove()
            self.dynamic_elements["transition_line"] = None

    def update(self, frame):
        """Update function for animation."""
        self.clear_arrows()

        # Animation phases:
        # Phase 1: frames 0-39 (optimization in T_{p_i}M)
        # Phase 2: frames 40-59 (transition)
        # Phase 3: frames 60-89 (optimization in T_{p_{i+1}}M)
        # Phase 4: frames 90-99 (hold)

        if frame < 40:
            self._animate_phase1(frame)
        elif frame < 60:
            self._animate_phase2(frame)
        elif frame < 90:
            self._animate_phase3(frame)
        else:
            self._animate_phase4(frame)

        return list(self.dynamic_elements.values())

    def _animate_phase1(self, frame):
        """Phase 1: Optimization in first tangent space."""
        s = self.scale
        progress = frame / 39
        n_visible = min(int(1 + progress * 4.5), 5)

        # Show points in tangent space 1
        visible_pts = self.tangent_pts_1[:n_visible]
        visible_colors = self.colors_phase1[:n_visible]

        self.dynamic_elements["tangent_scatter_1"].set_offsets(visible_pts)
        self.dynamic_elements["tangent_scatter_1"].set_facecolors(visible_colors)

        # Show trajectory line
        if n_visible > 1:
            pts = np.array(visible_pts)
            self.dynamic_elements["traj_line_1"].set_data(pts[:, 0], pts[:, 1])
        else:
            self.dynamic_elements["traj_line_1"].set_data([], [])

        # Show corresponding points on manifold
        visible_manifold = self.manifold_pts_1[:n_visible]
        self.dynamic_elements["manifold_scatter_1"].set_offsets(visible_manifold)
        self.dynamic_elements["manifold_scatter_1"].set_facecolors(visible_colors)

        # Manifold trajectory line
        if n_visible > 1:
            pts = np.array(visible_manifold)
            self.dynamic_elements["manifold_line_1"].set_data(pts[:, 0], pts[:, 1])
        else:
            self.dynamic_elements["manifold_line_1"].set_data([], [])

        # Clear phase 2 elements
        self.dynamic_elements["tangent_scatter_2"].set_offsets(np.empty((0, 2)))
        self.dynamic_elements["manifold_scatter_2"].set_offsets(np.empty((0, 2)))
        self.dynamic_elements["traj_line_2"].set_data([], [])
        self.dynamic_elements["manifold_line_2"].set_data([], [])

        # Draw φ_{p_i} arrow
        phi1_start = (
            self.ts1_origin[0] + 1.8,
            self.ts1_origin[1] + self.ts_height + 0.2,
        )
        phi1_end = (self.manifold_pts_1[0][0] - 0.2, self.manifold_pts_1[0][1] - 0.6)

        arrow1 = FancyArrowPatch(
            phi1_start,
            phi1_end,
            connectionstyle="arc3,rad=0.35",
            arrowstyle="-|>",
            mutation_scale=24 * s,
            lw=3 * s,
            color="#2980B9",
            zorder=5,
        )
        self.ax.add_patch(arrow1)
        self.dynamic_elements["phi1_arrow"] = arrow1

        self.dynamic_elements["phi1_label"].set_text(r"$\phi_{p_i}$")
        self.dynamic_elements["phi1_label"].set_color("#2980B9")
        self.dynamic_elements["phi2_label"].set_text("")
        self.dynamic_elements["transition_label"].set_text("")

        self.dynamic_elements["status"].set_text(
            f"Phase 1: Optimizing in $T_{{p_i}}\\mathcal{{M}}$  —  step {n_visible}/5"
        )

    def _animate_phase2(self, frame):
        """Phase 2: Transition to new base point."""
        s = self.scale
        progress = (frame - 40) / 19

        # Keep all phase 1 points visible
        self.dynamic_elements["tangent_scatter_1"].set_offsets(self.tangent_pts_1)
        self.dynamic_elements["tangent_scatter_1"].set_facecolors(self.colors_phase1)
        self.dynamic_elements["manifold_scatter_1"].set_offsets(self.manifold_pts_1)
        self.dynamic_elements["manifold_scatter_1"].set_facecolors(self.colors_phase1)

        # Trajectory lines
        pts1 = np.array(self.tangent_pts_1)
        self.dynamic_elements["traj_line_1"].set_data(pts1[:, 0], pts1[:, 1])
        mpts1 = np.array(self.manifold_pts_1)
        self.dynamic_elements["manifold_line_1"].set_data(mpts1[:, 0], mpts1[:, 1])

        # Draw φ_{p_i} arrow (fading)
        phi1_start = (
            self.ts1_origin[0] + 1.8,
            self.ts1_origin[1] + self.ts_height + 0.2,
        )
        phi1_end = (self.manifold_pts_1[0][0] - 0.2, self.manifold_pts_1[0][1] - 0.6)

        arrow1 = FancyArrowPatch(
            phi1_start,
            phi1_end,
            connectionstyle="arc3,rad=0.35",
            arrowstyle="-|>",
            mutation_scale=24 * s,
            lw=3 * s,
            color="#2980B9",
            alpha=max(0.3, 1 - progress * 0.7),
            zorder=5,
        )
        self.ax.add_patch(arrow1)
        self.dynamic_elements["phi1_arrow"] = arrow1

        # Draw transition arrow (appearing)
        trans_start = (self.ts1_origin[0] + self.ts_width + 0.5, 2.0)
        trans_end_full = (self.ts2_origin[0] - 0.5, 2.0)

        current_end_x = trans_start[0] + progress * (trans_end_full[0] - trans_start[0])

        # Draw line
        (line,) = self.ax.plot(
            [trans_start[0], current_end_x],
            [trans_start[1], trans_start[1]],
            "-",
            color="#E74C3C",
            lw=4 * s,
            zorder=5,
        )
        self.dynamic_elements["transition_line"] = line

        # Add arrowhead when near complete
        if progress > 0.85:
            arrow_trans = FancyArrowPatch(
                (current_end_x - 0.5, trans_start[1]),
                (current_end_x, trans_start[1]),
                arrowstyle="-|>",
                mutation_scale=24 * s,
                lw=4 * s,
                color="#E74C3C",
                zorder=5,
            )
            self.ax.add_patch(arrow_trans)
            self.dynamic_elements["transition_arrow"] = arrow_trans

        self.dynamic_elements["phi1_label"].set_text(r"$\phi_{p_i}$")
        self.dynamic_elements["phi1_label"].set_color("#95A5A6")
        self.dynamic_elements["phi2_label"].set_text("")
        self.dynamic_elements["transition_label"].set_text(
            r"$p_{i+1} := \phi_{p_i}(y_{i,4})$"
        )
        self.dynamic_elements["transition_label"].set_color("#E74C3C")

        # Show first point of phase 2 appearing at the end
        if progress > 0.75:
            alpha = (progress - 0.75) / 0.25
            self.dynamic_elements["tangent_scatter_2"].set_offsets(
                [self.tangent_pts_2[0]]
            )
            self.dynamic_elements["tangent_scatter_2"].set_facecolors(
                [(*self.colors_phase2[0], alpha)]
            )
            self.dynamic_elements["manifold_scatter_2"].set_offsets(
                [self.manifold_pts_2[0]]
            )
            self.dynamic_elements["manifold_scatter_2"].set_facecolors(
                [(*self.colors_phase2[0], alpha)]
            )
        else:
            self.dynamic_elements["tangent_scatter_2"].set_offsets(np.empty((0, 2)))
            self.dynamic_elements["manifold_scatter_2"].set_offsets(np.empty((0, 2)))

        self.dynamic_elements["traj_line_2"].set_data([], [])
        self.dynamic_elements["manifold_line_2"].set_data([], [])

        self.dynamic_elements["status"].set_text(
            "Phase 2: Updating base point  —  $p_{i+1} := \\phi_{p_i}(y_{i,4})$"
        )

    def _animate_phase3(self, frame):
        """Phase 3: Optimization in second tangent space."""
        s = self.scale
        progress = (frame - 60) / 29
        n_visible = min(int(1 + progress * 4.5), 5)

        # Phase 1 points (grayed out)
        self.dynamic_elements["tangent_scatter_1"].set_offsets(self.tangent_pts_1)
        gray_colors = [(0.75, 0.75, 0.75, 0.5)] * 5
        self.dynamic_elements["tangent_scatter_1"].set_facecolors(gray_colors)
        self.dynamic_elements["manifold_scatter_1"].set_offsets(self.manifold_pts_1)
        self.dynamic_elements["manifold_scatter_1"].set_facecolors(gray_colors)

        # Gray trajectory lines for phase 1
        pts1 = np.array(self.tangent_pts_1)
        self.dynamic_elements["traj_line_1"].set_data(pts1[:, 0], pts1[:, 1])
        self.dynamic_elements["traj_line_1"].set_alpha(0.2)
        self.dynamic_elements["traj_line_1"].set_color("#95A5A6")

        mpts1 = np.array(self.manifold_pts_1)
        self.dynamic_elements["manifold_line_1"].set_data(mpts1[:, 0], mpts1[:, 1])
        self.dynamic_elements["manifold_line_1"].set_alpha(0.2)
        self.dynamic_elements["manifold_line_1"].set_color("#95A5A6")

        # Show points in tangent space 2
        visible_pts = self.tangent_pts_2[:n_visible]
        visible_colors = self.colors_phase2[:n_visible]

        self.dynamic_elements["tangent_scatter_2"].set_offsets(visible_pts)
        self.dynamic_elements["tangent_scatter_2"].set_facecolors(visible_colors)

        # Trajectory line for phase 2
        if n_visible > 1:
            pts = np.array(visible_pts)
            self.dynamic_elements["traj_line_2"].set_data(pts[:, 0], pts[:, 1])
            self.dynamic_elements["traj_line_2"].set_alpha(0.5)
            self.dynamic_elements["traj_line_2"].set_color("#9B59B6")
        else:
            self.dynamic_elements["traj_line_2"].set_data([], [])

        # Show corresponding points on manifold
        visible_manifold = self.manifold_pts_2[:n_visible]
        self.dynamic_elements["manifold_scatter_2"].set_offsets(visible_manifold)
        self.dynamic_elements["manifold_scatter_2"].set_facecolors(visible_colors)

        # Manifold trajectory line for phase 2
        if n_visible > 1:
            pts = np.array(visible_manifold)
            self.dynamic_elements["manifold_line_2"].set_data(pts[:, 0], pts[:, 1])
            self.dynamic_elements["manifold_line_2"].set_alpha(0.6)
            self.dynamic_elements["manifold_line_2"].set_color("#9B59B6")
        else:
            self.dynamic_elements["manifold_line_2"].set_data([], [])

        # Draw φ_{p_i} arrow (grayed)
        phi1_start = (
            self.ts1_origin[0] + 1.8,
            self.ts1_origin[1] + self.ts_height + 0.2,
        )
        phi1_end = (self.manifold_pts_1[0][0] - 0.2, self.manifold_pts_1[0][1] - 0.6)

        arrow1 = FancyArrowPatch(
            phi1_start,
            phi1_end,
            connectionstyle="arc3,rad=0.35",
            arrowstyle="-|>",
            mutation_scale=22 * s,
            lw=2.5 * s,
            color="#BDC3C7",
            alpha=0.4,
            zorder=3,
        )
        self.ax.add_patch(arrow1)
        self.dynamic_elements["phi1_arrow"] = arrow1

        # Draw φ_{p_{i+1}} arrow (active)
        phi2_start = (
            self.ts2_origin[0] + 1.8,
            self.ts2_origin[1] + self.ts_height + 0.2,
        )
        phi2_end = (self.manifold_pts_2[0][0] + 0.3, self.manifold_pts_2[0][1] - 0.6)

        arrow2 = FancyArrowPatch(
            phi2_start,
            phi2_end,
            connectionstyle="arc3,rad=-0.35",
            arrowstyle="-|>",
            mutation_scale=24 * s,
            lw=3 * s,
            color="#8E44AD",
            zorder=5,
        )
        self.ax.add_patch(arrow2)
        self.dynamic_elements["phi2_arrow"] = arrow2

        # Transition arrow (grayed, complete)
        trans_start = (self.ts1_origin[0] + self.ts_width + 0.5, 2.0)
        trans_end = (self.ts2_origin[0] - 0.5, 2.0)

        (line,) = self.ax.plot(
            [trans_start[0], trans_end[0]],
            [trans_start[1], trans_end[1]],
            "-",
            color="#BDC3C7",
            lw=3 * s,
            alpha=0.4,
            zorder=3,
        )
        self.dynamic_elements["transition_line"] = line

        self.dynamic_elements["phi1_label"].set_text(r"$\phi_{p_i}$")
        self.dynamic_elements["phi1_label"].set_color("#BDC3C7")
        self.dynamic_elements["phi2_label"].set_text(r"$\phi_{p_{i+1}}$")
        self.dynamic_elements["phi2_label"].set_color("#8E44AD")
        self.dynamic_elements["transition_label"].set_text(
            r"$p_{i+1} := \phi_{p_i}(y_{i,4})$"
        )
        self.dynamic_elements["transition_label"].set_color("#BDC3C7")

        self.dynamic_elements["status"].set_text(
            f"Phase 3: Optimizing in $T_{{p_{{i+1}}}}\\mathcal{{M}}$  —  step {n_visible}/5"
        )

    def _animate_phase4(self, frame):
        """Phase 4: Hold final state."""
        # Same as end of phase 3, but with "complete" message
        self._animate_phase3(89)  # Render final state of phase 3

        self.dynamic_elements["status"].set_text(
            "Complete!  Ready to continue with next base point update..."
        )

    def create_animation(self, save_path="dynamic_trivialization_animated_v2.gif"):
        """Create and save the animation."""
        print(f"Creating animation with {self.total_frames} frames...")

        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=self.total_frames,
            interval=80,  # 80ms between frames (12.5 fps)
            blit=False,
            repeat=True,
        )

        # Save as GIF
        print("Saving GIF...")
        writer = PillowWriter(fps=12)
        anim.save(save_path, writer=writer, dpi=130)
        print(f"Saved: {save_path}")

        return anim


def main():
    """Main function to create the animation."""
    import os

    print("Creating improved trivialization animation (larger version)...")

    # Output to _static/images relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "images", "dynamic_trivialization.gif")

    # Use scale=1.2 for bigger elements
    animator = ImprovedDynamicTrivializationAnimation(scale=1.2)
    anim = animator.create_animation(output_path)

    print("\nAnimation complete!")
    plt.show()


if __name__ == "__main__":
    main()
