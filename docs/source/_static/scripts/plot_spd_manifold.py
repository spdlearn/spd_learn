"""
SPD Manifold Visualization for EEG Covariance Analysis.

Creates an interactive 3D Plotly visualization showing the SPD cone
with sample EEG covariance matrices, the identity matrix reference point,
and the tangent space at the identity.

This visualization is embedded in the geometric_concepts documentation page.
"""

import os

import numpy as np
import plotly.graph_objects as go


def create_spd_cone(size=2.5, resolution=50):
    """
    Create a cone representing the SPD manifold.

    The SPD manifold S++^n can be visualized as a cone in the space of
    symmetric matrices, where the cone opens upward representing positive
    definiteness.
    """
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0.1, size, resolution)
    Theta, Z = np.meshgrid(theta, z)

    # Cone radius increases with z (representing eigenvalue constraint)
    R = Z * 0.5
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    return X, Y, Z


def create_tangent_plane(base_point, normal, size=1.5, resolution=15):
    """
    Create a tangent plane at a given point on the manifold.

    Parameters
    ----------
    base_point : array-like
        The point on the manifold where tangent plane is attached
    normal : array-like
        Normal vector to the plane (points outward from manifold)
    size : float
        Size of the tangent plane
    resolution : int
        Grid resolution
    """
    normal = np.array(normal) / np.linalg.norm(normal)

    # Find two vectors orthogonal to normal
    if abs(normal[0]) < 0.9:
        v1 = np.cross(normal, [1, 0, 0])
    else:
        v1 = np.cross(normal, [0, 1, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)

    # Create grid
    s = np.linspace(-size / 2, size / 2, resolution)
    t = np.linspace(-size / 2, size / 2, resolution)
    S, T = np.meshgrid(s, t)

    # Tangent plane points
    X = base_point[0] + S * v1[0] + T * v2[0]
    Y = base_point[1] + S * v1[1] + T * v2[1]
    Z = base_point[2] + S * v1[2] + T * v2[2]

    return X, Y, Z


def create_spd_manifold_visualization():
    """
    Create a visualization specifically for the SPD manifold used in EEG analysis.

    The SPD manifold for 2x2 matrices can be visualized as a 3D cone,
    where points represent covariance matrices.
    """
    fig = go.Figure()

    # Create SPD cone
    X, Y, Z = create_spd_cone(size=2.5, resolution=50)

    # Add cone surface
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Blues",
            showscale=False,
            opacity=0.4,
            name="SPD Manifold S++",
            hovertemplate="SPD Manifold S<sub>++</sub><sup>n</sup><extra></extra>",
        )
    )

    # Add identity matrix point (center of cone at z=1)
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[1],
            mode="markers+text",
            marker=dict(size=12, color="gold", line=dict(width=2, color="black")),
            text=["I (Identity)"],
            textposition="top center",
            name="Identity Matrix",
            hovertemplate="Identity Matrix I<br>Reference point for tangent space<extra></extra>",
        )
    )

    # Sample EEG covariance matrices (simulated as points on cone)
    np.random.seed(42)
    n_samples = 15

    # Generate points on the cone surface
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    z_vals = np.random.uniform(0.5, 2.0, n_samples)
    r_vals = z_vals * 0.5 * np.random.uniform(0.3, 0.9, n_samples)  # Inside the cone

    sample_x = r_vals * np.cos(theta)
    sample_y = r_vals * np.sin(theta)
    sample_z = z_vals

    # Color by "class" (simulated)
    colors = ["red" if i < n_samples // 2 else "blue" for i in range(n_samples)]

    fig.add_trace(
        go.Scatter3d(
            x=sample_x,
            y=sample_y,
            z=sample_z,
            mode="markers",
            marker=dict(size=6, color=colors, line=dict(width=1, color="white")),
            name="EEG Spatial Covariance Matrices",
            hovertemplate="Covariance Matrix<br>Class: %{text}<extra></extra>",
            text=["Class A" if c == "red" else "Class B" for c in colors],
        )
    )

    # Add tangent plane at identity
    base = np.array([0, 0, 1])
    normal = np.array([0, 0, 1])
    Xt, Yt, Zt = create_tangent_plane(base, normal, size=1.5, resolution=15)

    fig.add_trace(
        go.Surface(
            x=Xt,
            y=Yt,
            z=Zt,
            colorscale=[
                [0, "rgba(144, 238, 144, 0.3)"],
                [1, "rgba(144, 238, 144, 0.3)"],
            ],
            showscale=False,
            opacity=0.4,
            name="Tangent Space (Symmetric Matrices)",
            hovertemplate="Tangent Space T<sub>I</sub>S<sub>++</sub><br>= Space of Symmetric Matrices<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>SPD Manifold for EEG Covariance Analysis</b><br><sup>Covariance matrices as points on the positive definite cone</sup>",
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(title="σ₁₂ (off-diagonal)", showgrid=True),
            yaxis=dict(title="σ₁₁ - σ₂₂", showgrid=True),
            zaxis=dict(title="tr(Σ) (trace)", showgrid=True),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode="cube",
        ),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=80, b=0),
        width=900,
        height=700,
    )

    return fig


def main():
    """Main function to create and save the visualization."""
    print("Creating SPD manifold visualization...")

    # Output path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "spd_manifold_eeg.html")

    fig = create_spd_manifold_visualization()
    fig.write_html(output_path)

    print(f"Saved: {output_path}")
    print("\nOpen the HTML file in a browser to view the interactive visualization.")


if __name__ == "__main__":
    main()
