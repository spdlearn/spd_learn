"""
.. _radar-classification:

Radar Image Classification with SPD Learn
==========================================

This tutorial demonstrates how to use **spd_learn** for synthetic aperture
radar (SAR) polarimetric image classification using **real UAVSAR data**.
Polarimetric SAR data naturally produces symmetric positive definite (SPD)
covariance matrices, making this an ideal application for SPD-based
machine learning.

We use the UAVSAR dataset from NASA's Uninhabited Aerial Vehicle Synthetic
Aperture Radar system, with pseudo-labels generated via Riemannian clustering.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# Polarimetric SAR (PolSAR) :cite:p:`lee2009polarimetric` systems transmit and
# receive electromagnetic waves in multiple polarization states (typically
# HH, HV, VH, VV), capturing rich information about the scattering properties
# of terrain and objects.
#
# The data is commonly represented as coherency or covariance matrices,
# which are **inherently SPD matrices**. This makes SPD-based deep learning
# approaches particularly well-suited for PolSAR classification tasks.
#
# **The UAVSAR Dataset**
#
# The UAVSAR (Uninhabited Aerial Vehicle Synthetic Aperture Radar) is a
# NASA/JPL airborne SAR system. The data used here covers the Los Angeles
# area with:
#
# - **Source**: NASA/JPL UAVSAR, L-band
# - **Format**: Polarimetric scattering vectors (3 channels: HH, HV, VV)
# - **Hosted on**: Zenodo (open access)
#
# In this tutorial, we:
#
# 1. Load real UAVSAR PolSAR data
# 2. Visualize the radar image using Pauli RGB decomposition
#    :cite:p:`cloude1996review`
# 3. Generate pseudo-labels using Riemannian K-Means clustering
# 4. Train SPDNet :cite:p:`huang2017riemannian` and compare with pyRiemann
#    baselines
#

######################################################################
# Setup and Imports
# -----------------
#

from __future__ import annotations

import logging
import warnings

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import torch

from braindecode import EEGClassifier
from numpy.typing import NDArray
from pyriemann.classification import MDM
from pyriemann.clustering import Kmeans
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

from spd_learn.models import SPDNet


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

######################################################################
# UAVSAR Data Loading
# -------------------
#
# The UAVSAR dataset is hosted on Zenodo and contains processed PolSAR
# scattering vectors from the Los Angeles area.
#
# Data source: https://zenodo.org/records/10625505
#


def get_data_dir() -> Path:
    """Get the data directory for storing downloaded datasets."""
    data_dir = Path.home() / ".spd_learn_data" / "uavsar"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def verify_npy_file(filepath: Path, expected_shape: tuple = (2360, 600, 3, 4)) -> bool:
    """Verify that a .npy file is complete and valid.

    Parameters
    ----------
    filepath : Path
        Path to the .npy file.
    expected_shape : tuple
        Expected shape of the array.

    Returns
    -------
    bool
        True if file is valid, False otherwise.
    """
    try:
        data = np.load(filepath)
        if data.shape != expected_shape:
            logger.warning(f"Unexpected shape: {data.shape} vs {expected_shape}")
            return False
        return True
    except Exception as e:
        logger.warning(f"File verification failed: {e}")
        return False


def download_uavsar(data_path: Path, scene: int = 1) -> Path:
    """Download the UAVSAR dataset from Zenodo.

    Parameters
    ----------
    data_path : Path
        Path to the destination folder for data download.
    scene : {1, 2}
        Scene index to download.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    assert scene in [1, 2], f"Unknown scene {scene} for UAVSAR dataset"
    filename = f"scene{scene}.npy"
    src = f"https://zenodo.org/records/10625505/files/{filename}?download=1"

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    dst = data_path / filename

    # Check if file exists and is valid
    if dst.exists():
        logger.info(f"Verifying existing file: {dst}")
        if verify_npy_file(dst):
            logger.info("File verified successfully!")
            return dst
        else:
            logger.warning("File corrupted or incomplete. Re-downloading...")
            dst.unlink()  # Remove corrupted file

    logger.info(f"Downloading UAVSAR scene {scene} from Zenodo...")
    logger.info(f"Source: {src}")
    logger.info(f"Destination: {dst}")
    logger.info("This may take a few minutes for the ~136MB file...")
    urlretrieve(src, dst)

    # Verify the downloaded file
    if not verify_npy_file(dst):
        raise RuntimeError(
            "Downloaded file is corrupted. Please try again or "
            "manually download from: https://zenodo.org/records/10625505"
        )

    logger.info("Download complete and verified!")
    return dst


######################################################################
# Loading the Dataset
# -------------------
#
# We load the UAVSAR data which contains polarimetric scattering vectors.
# The data has shape (height, width, 3, n_dates) where the 3 channels
# correspond to [HH, HV, VV] polarizations.
#

print("=" * 60)
print("Loading Real UAVSAR PolSAR Data")
print("=" * 60)

data_path = get_data_dir()
file_path = download_uavsar(data_path, scene=1)

# Load the data
data = np.load(file_path)
print(f"Original data shape: {data.shape}")
# Shape: (height, width, 3, n_dates)

# Select first date
date_idx = 0
data = data[:, :, :, date_idx]
print(f"After date selection: {data.shape}")

# Store original dimensions for visualization
h_orig, w_orig, n_channels = data.shape

######################################################################
# Visualizing the SAR Image
# -------------------------
#
# We visualize the radar image using standard SAR representations:
#
# 1. **Total Power (Span)**: Sum of all polarimetric powers
# 2. **Pauli RGB**: Color composite showing scattering mechanisms
#
#    - Red: ``abs(HH - VV)`` (double-bounce)
#    - Green: ``abs(HV)`` (volume scattering)
#    - Blue: ``abs(HH + VV)`` (surface scattering)
#

# Compute intensity image (in dB)
intensity = 20 * np.log10(np.sum(np.abs(data) ** 2, axis=2) + 1e-10)

# Compute Pauli RGB components
HH = data[:, :, 0]
HV = data[:, :, 1]
VV = data[:, :, 2]

# Pauli decomposition
pauli_red = np.abs(HH - VV)  # Double-bounce
pauli_green = np.abs(HV) * np.sqrt(2)  # Volume
pauli_blue = np.abs(HH + VV)  # Surface


def normalize_for_display(img: NDArray, percentile: tuple = (2, 98)) -> NDArray:
    """Normalize image for display with percentile clipping."""
    p_low, p_high = np.percentile(img, percentile)
    img_norm = (img - p_low) / (p_high - p_low + 1e-10)
    return np.clip(img_norm, 0, 1)


# Create Pauli RGB image
pauli_rgb = np.stack(
    [
        normalize_for_display(pauli_red),
        normalize_for_display(pauli_green),
        normalize_for_display(pauli_blue),
    ],
    axis=2,
)

# Create figure with SAR visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Total power (Span)
ax = axes[0]
im = ax.imshow(intensity, cmap="gray", aspect="auto")
ax.set_title("SAR Intensity (dB)", fontsize=12, fontweight="bold")
ax.set_xlabel("Range")
ax.set_ylabel("Azimuth")
plt.colorbar(im, ax=ax, label="dB")

# Pauli RGB
ax = axes[1]
ax.imshow(pauli_rgb, aspect="auto")
ax.set_title("Pauli RGB Decomposition", fontsize=12, fontweight="bold")
ax.set_xlabel("Range")
ax.set_ylabel("Azimuth")

# Add legend for Pauli colors
from matplotlib.patches import Patch


legend_elements = [
    Patch(facecolor="red", label="Double-bounce |HH-VV|"),
    Patch(facecolor="green", label="Volume |HV|"),
    Patch(facecolor="blue", label="Surface |HH+VV|"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

# Individual channels
ax = axes[2]
ax.imshow(normalize_for_display(np.abs(HV)), cmap="viridis", aspect="auto")
ax.set_title("Cross-pol |HV| (Volume Scattering)", fontsize=12, fontweight="bold")
ax.set_xlabel("Range")
ax.set_ylabel("Azimuth")

plt.suptitle(
    "UAVSAR Los Angeles - Polarimetric SAR Visualization",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

######################################################################
# Computing Covariance Matrices
# -----------------------------
#
# For SPD-based classification, we compute local covariance matrices
# using a sliding window approach. Each pixel gets a 3x3 covariance
# matrix estimated from its neighborhood.
#

print("\n" + "=" * 60)
print("Computing Covariance Matrices")
print("=" * 60)

# Downsample for computational efficiency
downsample_h = 7
downsample_w = 4
data_ds = data[::downsample_h, ::downsample_w, :]
h_ds, w_ds, _ = data_ds.shape
print(f"Downsampled shape: {data_ds.shape}")

# Reshape for covariance estimation
# We'll compute covariance from the scattering vector at each pixel
# Shape: (n_pixels, n_channels)
data_flat = data_ds.reshape(-1, n_channels)
n_pixels = data_flat.shape[0]
print(f"Number of pixels: {n_pixels}")

# Compute covariance matrices from scattering vectors
# C = k * k^H where k = [HH, HV, VV]
print("Computing covariance matrices...")
covs = np.zeros((n_pixels, 3, 3), dtype=np.float64)

for i in range(n_pixels):
    k = data_flat[i]  # Scattering vector
    C = np.outer(k, np.conj(k))  # Outer product
    covs[i] = np.real(C)  # Take real part

# Ensure strong positive definiteness for Riemannian operations
# Riemannian mean computation requires well-conditioned matrices
print("Ensuring positive definiteness...")
min_eigenvalue = 1e-4  # Minimum eigenvalue for numerical stability
for i in range(n_pixels):
    # Symmetrize
    covs[i] = (covs[i] + covs[i].T) / 2
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(covs[i])
    # Add regularization to ensure minimum eigenvalue
    if np.min(eigvals) < min_eigenvalue:
        covs[i] += (min_eigenvalue - np.min(eigvals) + 1e-6) * np.eye(3)

X = covs
print(f"Covariance matrices shape: {X.shape}")

######################################################################
# Clustering for Pseudo-Labels
# ----------------------------
#
# Since UAVSAR doesn't have ground truth labels, we use Riemannian
# K-Means clustering to segment the image into terrain types.
# The clusters serve as pseudo-labels for classification.
#

print("\n" + "=" * 60)
print("Riemannian K-Means Clustering")
print("=" * 60)

n_clusters = 3  # Reduced from 5 for faster documentation build

print(f"Clustering into {n_clusters} classes using Riemannian K-Means...")
kmeans = Kmeans(n_clusters=n_clusters, metric="riemann", random_state=SEED)
y = kmeans.fit_predict(X)

# Reshape labels back to image
labels_image = y.reshape(h_ds, w_ds)

# Count samples per cluster
unique, counts = np.unique(y, return_counts=True)
print("\nCluster distribution:")
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} pixels ({count/len(y)*100:.1f}%)")

######################################################################
# Visualizing Clustering Results
# ------------------------------
#
# We display the clustering results as a segmentation map overlaid
# on the original SAR image, similar to pyRiemann's visualization.
#

# Downsample Pauli RGB for comparison
pauli_rgb_ds = pauli_rgb[::downsample_h, ::downsample_w, :]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original Pauli RGB (downsampled)
ax = axes[0]
ax.imshow(pauli_rgb_ds, aspect="auto")
ax.set_title("Pauli RGB (Downsampled)", fontsize=12, fontweight="bold")
ax.set_xlabel("Range")
ax.set_ylabel("Azimuth")

# Clustering result
ax = axes[1]
im = ax.imshow(labels_image, cmap="tab10", aspect="auto")
ax.set_title(
    f"Riemannian K-Means ({n_clusters} clusters)", fontsize=12, fontweight="bold"
)
ax.set_xlabel("Range")
ax.set_ylabel("Azimuth")
plt.colorbar(im, ax=ax, label="Cluster")

# Overlay: Pauli RGB with cluster boundaries
ax = axes[2]
ax.imshow(pauli_rgb_ds, aspect="auto")
ax.contour(
    labels_image, levels=n_clusters - 1, colors="white", linewidths=0.5, alpha=0.7
)
ax.set_title("Pauli RGB with Cluster Boundaries", fontsize=12, fontweight="bold")
ax.set_xlabel("Range")
ax.set_ylabel("Azimuth")

plt.suptitle("UAVSAR Segmentation Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

######################################################################
# Analyzing Cluster Characteristics
# ---------------------------------
#
# Let's examine the covariance structure of each cluster to understand
# what terrain types they might represent.
#

fig, axes = plt.subplots(2, n_clusters, figsize=(3 * n_clusters, 6))

for cluster_idx in range(n_clusters):
    # Get samples from this cluster
    mask = y == cluster_idx
    cluster_covs = X[mask]

    # Mean covariance matrix
    mean_cov = np.mean(cluster_covs, axis=0)

    # Plot mean covariance
    ax = axes[0, cluster_idx]
    im = ax.imshow(mean_cov, cmap="viridis", aspect="equal")
    ax.set_title(f"Cluster {cluster_idx}\nMean Cov", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["HH", "HV", "VV"], fontsize=8)
    ax.set_yticklabels(["HH", "HV", "VV"], fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Eigenvalue distribution
    ax = axes[1, cluster_idx]
    eigenvalues = np.array([np.linalg.eigvalsh(c) for c in cluster_covs])
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for eig_idx in range(3):
        ax.hist(
            eigenvalues[:, eig_idx],
            bins=30,
            alpha=0.6,
            color=colors[eig_idx],
            label=f"λ{eig_idx+1}",
        )
    ax.set_xlabel("Eigenvalue", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Eigenvalues", fontsize=10)
    if cluster_idx == n_clusters - 1:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Cluster Analysis: Covariance Structure", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

######################################################################
# Data Splitting for Classification
# ---------------------------------
#
# We split the data into training and test sets for classification.
#

n_classes = n_clusters

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

######################################################################
# Classification with pyRiemann Baselines
# ---------------------------------------
#
# We compare SPDNet against pyRiemann classifiers:
#
# 1. **MDM**: Minimum Distance to Mean
# 2. **Tangent Space + Logistic Regression**
#

print("\n" + "=" * 60)
print("Training Classifiers")
print("=" * 60)

# MDM Classifier
print("\n[1] Training MDM classifier...")
clf_mdm = MDM(metric="riemann")
clf_mdm.fit(X_train, y_train)
y_pred_mdm = clf_mdm.predict(X_test)
acc_mdm = accuracy_score(y_test, y_pred_mdm)
bal_acc_mdm = balanced_accuracy_score(y_test, y_pred_mdm)
print(f"    MDM Accuracy: {acc_mdm*100:.2f}%")
print(f"    MDM Balanced Accuracy: {bal_acc_mdm*100:.2f}%")

# Tangent Space + Logistic Regression
print("\n[2] Training Tangent Space + LR classifier...")
clf_ts = make_pipeline(
    TangentSpace(metric="riemann"),
    LogisticRegression(random_state=SEED, max_iter=1000),
)
clf_ts.fit(X_train, y_train)
y_pred_ts = clf_ts.predict(X_test)
acc_ts = accuracy_score(y_test, y_pred_ts)
bal_acc_ts = balanced_accuracy_score(y_test, y_pred_ts)
print(f"    TS+LR Accuracy: {acc_ts*100:.2f}%")
print(f"    TS+LR Balanced Accuracy: {bal_acc_ts*100:.2f}%")

######################################################################
# SPDNet for Radar Classification
# -------------------------------
#

print("\n[3] Training SPDNet...")
model = SPDNet(
    input_type="cov",
    n_chans=3,
    n_outputs=n_classes,
    subspacedim=3,
    threshold=1e-4,
)

print("\nSPDNet Architecture:")
print(model)

clf_spd = EEGClassifier(
    module=model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=1e-2,
    optimizer__weight_decay=1e-4,
    train_split=ValidSplit(0.1, stratified=True, random_state=SEED),
    batch_size=64,
    max_epochs=20,  # Reduced from 100 for faster documentation build
    callbacks=[
        (
            "train_acc",
            EpochScoring(
                "accuracy", lower_is_better=False, on_train=True, name="train_acc"
            ),
        ),
        EarlyStopping(monitor="valid_loss", patience=15),
    ],
    device=device,
    verbose=1,
)

clf_spd.fit(X_train, y_train)

y_pred_spd = clf_spd.predict(X_test)
acc_spd = accuracy_score(y_test, y_pred_spd)
bal_acc_spd = balanced_accuracy_score(y_test, y_pred_spd)
print(f"\nSPDNet Accuracy: {acc_spd*100:.2f}%")
print(f"SPDNet Balanced Accuracy: {bal_acc_spd*100:.2f}%")

######################################################################
# Classification Maps
# -------------------
#
# We create classification maps by predicting the class for each pixel
# and displaying as an image.
#

print("\n" + "=" * 60)
print("Generating Classification Maps")
print("=" * 60)

# Predict for all pixels
y_pred_all_mdm = clf_mdm.predict(X)
y_pred_all_ts = clf_ts.predict(X)
y_pred_all_spd = clf_spd.predict(X)

# Reshape to images
map_mdm = y_pred_all_mdm.reshape(h_ds, w_ds)
map_ts = y_pred_all_ts.reshape(h_ds, w_ds)
map_spd = y_pred_all_spd.reshape(h_ds, w_ds)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Top row: Classification maps
for ax, (name, pred_map, acc) in zip(
    axes[0],
    [
        ("MDM", map_mdm, acc_mdm),
        ("TS + LogReg", map_ts, acc_ts),
        ("SPDNet", map_spd, acc_spd),
    ],
):
    im = ax.imshow(pred_map, cmap="tab10", aspect="auto")
    ax.set_title(f"{name}\nAccuracy: {acc*100:.1f}%", fontsize=12, fontweight="bold")
    ax.set_xlabel("Range")
    ax.set_ylabel("Azimuth")
    plt.colorbar(im, ax=ax, label="Class")

# Bottom row: Overlays on Pauli RGB
for ax, (name, pred_map) in zip(
    axes[1],
    [("MDM", map_mdm), ("TS + LogReg", map_ts), ("SPDNet", map_spd)],
):
    ax.imshow(pauli_rgb_ds, aspect="auto")
    ax.contour(
        pred_map, levels=n_clusters - 1, colors="yellow", linewidths=0.5, alpha=0.8
    )
    ax.set_title(f"{name} Boundaries on Pauli RGB", fontsize=12, fontweight="bold")
    ax.set_xlabel("Range")
    ax.set_ylabel("Azimuth")

plt.suptitle("Classification Results Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

######################################################################
# Results Summary
# ---------------
#

results = {
    "MDM": {"accuracy": acc_mdm, "balanced_accuracy": bal_acc_mdm},
    "TS + LogReg": {"accuracy": acc_ts, "balanced_accuracy": bal_acc_ts},
    "SPDNet": {"accuracy": acc_spd, "balanced_accuracy": bal_acc_spd},
}

print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)
print(f"\n{'Method':<20} {'Accuracy':<15} {'Balanced Acc':<15}")
print("-" * 50)
for method, metrics in results.items():
    print(
        f"{method:<20} {metrics['accuracy']*100:>6.2f}%        {metrics['balanced_accuracy']*100:>6.2f}%"
    )

######################################################################
# Final Visualization
# -------------------
#

fig = plt.figure(figsize=(16, 10))

# Accuracy comparison
ax1 = fig.add_subplot(2, 2, 1)
methods = list(results.keys())
accuracies = [results[m]["accuracy"] for m in methods]
bal_accuracies = [results[m]["balanced_accuracy"] for m in methods]

x = np.arange(len(methods))
width = 0.35

bars1 = ax1.bar(
    x - width / 2,
    accuracies,
    width,
    label="Accuracy",
    color="#3498db",
    edgecolor="black",
)
bars2 = ax1.bar(
    x + width / 2,
    bal_accuracies,
    width,
    label="Balanced Acc",
    color="#2ecc71",
    edgecolor="black",
)

ax1.set_xlabel("Method", fontsize=12)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_title("Classification Performance", fontsize=14, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylim([0, 1.1])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis="y")

for bar in bars1 + bars2:
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.2%}",
        ha="center",
        fontsize=9,
    )

# Training history
ax2 = fig.add_subplot(2, 2, 2)
history = clf_spd.history
epochs = range(1, len(history) + 1)
ax2.plot(epochs, history[:, "train_loss"], "b-", label="Train Loss", linewidth=2)
ax2.plot(epochs, history[:, "valid_loss"], "r--", label="Valid Loss", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("SPDNet Training History", fontsize=14, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Confusion matrices
cluster_labels = [f"C{i}" for i in range(n_classes)]

ax3 = fig.add_subplot(2, 3, 4)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_mdm,
    ax=ax3,
    display_labels=cluster_labels,
    cmap="Blues",
    colorbar=False,
)
ax3.set_title("MDM", fontsize=12, fontweight="bold")

ax4 = fig.add_subplot(2, 3, 5)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_ts,
    ax=ax4,
    display_labels=cluster_labels,
    cmap="Blues",
    colorbar=False,
)
ax4.set_title("TS + LogReg", fontsize=12, fontweight="bold")

ax5 = fig.add_subplot(2, 3, 6)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_spd,
    ax=ax5,
    display_labels=cluster_labels,
    cmap="Blues",
    colorbar=False,
)
ax5.set_title("SPDNet", fontsize=12, fontweight="bold")

plt.suptitle(
    "UAVSAR Classification with SPD Methods", fontsize=16, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.show()

######################################################################
# Summary
# -------
#
# In this tutorial, we demonstrated:
#
# 1. **Real PolSAR data**: Loaded NASA UAVSAR data from Zenodo
#
# 2. **SAR visualization**: Displayed radar intensity and Pauli RGB
#
# 3. **Covariance estimation**: Computed 3x3 SPD matrices from
#    polarimetric scattering vectors
#
# 4. **Riemannian clustering**: Generated pseudo-labels using K-Means
#    with the Riemannian metric
#
# 5. **Classification comparison**: Compared MDM, Tangent Space + LR,
#    and SPDNet on real radar data
#
# 6. **Classification maps**: Visualized spatial prediction results
#
# **Key takeaways:**
#
# - PolSAR covariance matrices are naturally SPD
# - Riemannian geometry captures meaningful terrain differences
# - SPDNet learns features directly on the SPD manifold
# - Classification maps reveal spatial patterns in the radar image
#
