"""
.. _skada-integration:

Domain Adaptation with skada and SPD Learn
==========================================

This tutorial demonstrates how to integrate SPD Learn with
`skada <https://scikit-adaptation.github.io/>`_ (scikit-learn domain adaptation)
for cross-subject EEG classification. We show how to combine SPD-based feature
extraction with various domain adaptation strategies.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# **Domain adaptation** is a machine learning technique that addresses the
# challenge of applying a model trained on data from one distribution (source
# domain) to data from a different but related distribution (target domain).
#
# In EEG-based Brain-Computer Interfaces (BCIs), domain adaptation is crucial
# because:
#
# - **Subject variability**: EEG signals differ significantly between subjects
#   due to anatomical differences, electrode placement, and neural patterns
# - **Session variability**: Even the same subject shows drift over time due to
#   electrode impedance changes and mental state variations
# - **Limited labeled data**: Collecting labeled EEG is expensive and
#   time-consuming, requiring subjects to perform specific tasks
#
# This tutorial combines two powerful approaches:
#
# 1. **SPD Learn**: Extracts geometric features from covariance matrices that
#    lie on the SPD (Symmetric Positive Definite) manifold
# 2. **skada**: Provides domain adaptation algorithms to align feature
#    distributions between source and target domains
#
# **Methods demonstrated:**
#
# - **SPDBatchNormMeanVar**: Native Riemannian domain adaptation
#   :cite:p:`kobler2022spd`
# - **CORAL**: Correlation Alignment :cite:p:`sun2016coral`
# - **Subspace Alignment**: Linear subspace mapping
#   :cite:p:`fernando2013subspace`
# - **Optimal Transport**: Sample transport between domains
#   :cite:p:`courty2017optimal`
#
#

######################################################################
# Setup and Imports
# -----------------
#

from __future__ import annotations

import warnings

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, SPDBatchNormMeanVar


warnings.filterwarnings("ignore")

from skada import (
    CORALAdapter,
    EntropicOTMapping,
    SubspaceAlignment,
    make_da_pipeline,
)


######################################################################
# Configuration
# -------------
#

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

######################################################################
# Loading EEG Data
# ----------------
#
# We use the BNCI2014_001 dataset (BCI Competition IV 2a) with:
#
# - **Source domain**: Subject 1 (training data with labels)
# - **Target domain**: Subject 2 (testing/adaptation - labels only for
#   evaluation)
#
# This simulates a common BCI scenario: train a model on one subject and
# deploy it on another subject without collecting new labeled data.
#

dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=2)  # Left vs Right hand

# Cache configuration for faster loading
cache_config = dict(
    save_raw=True,
    save_epochs=True,
    save_array=True,
    use=True,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
)

# Define source and target subjects
source_subject = 1
target_subject = 2

print(f"Loading data for subjects {source_subject} and {target_subject}...")
X, labels, meta = paradigm.get_data(
    dataset=dataset,
    subjects=[source_subject, target_subject],
    cache_config=cache_config,
)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Split by subject
source_mask = meta["subject"] == source_subject
target_mask = meta["subject"] == target_subject

X_source: np.ndarray = X[source_mask]
y_source: np.ndarray = y[source_mask]
X_target: np.ndarray = X[target_mask]
y_target: np.ndarray = y[target_mask]

print(f"\nSource domain (Subject {source_subject}): {len(X_source)} samples")
print(f"Target domain (Subject {target_subject}): {len(X_target)} samples")
print(f"Number of channels: {X.shape[1]}")
print(f"Number of time points: {X.shape[2]}")
print(f"Classes: {le.classes_}")

######################################################################
# SPD Feature Extraction
# ----------------------
#
# We create an SPD-based feature extractor that transforms raw EEG signals
# into geometric features suitable for domain adaptation:
#
# 1. **CovLayer**: Computes covariance matrices from raw EEG
# 2. **BiMap**: Learns a bilinear mapping for dimensionality reduction
# 3. **ReEig**: Applies a non-linearity to eigenvalues for numerical stability
# 4. **LogEig**: Projects SPD matrices to tangent space at identity (Euclidean)
#
# The tangent space features are then suitable for standard ML classifiers
# and domain adaptation methods.
#


class SPDFeatureExtractor(torch.nn.Module):
    """Extract tangent space features from raw EEG using SPD operations.

    This module computes covariance matrices from EEG signals and projects
    them to the tangent space at the identity matrix, producing Euclidean
    features suitable for standard classifiers.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels in the input data.
    subspace_dim : int, optional
        Dimension of the BiMap output (reduced SPD matrix size).
        If None, uses the full channel dimension. Smaller values reduce
        feature dimensionality but may lose discriminative information.
    threshold : float, default=1e-4
        ReEig threshold for numerical stability. Eigenvalues below this
        threshold are clipped to prevent numerical issues in matrix
        operations.

    Attributes
    ----------
    output_dim : int
        Number of output features (upper triangular elements of the
        subspace_dim x subspace_dim matrix).

    Examples
    --------
    >>> extractor = SPDFeatureExtractor(n_channels=22, subspace_dim=10)
    >>> X = torch.randn(32, 22, 500)  # batch of 32 EEG trials
    >>> features = extractor(X)
    >>> features.shape
    torch.Size([32, 55])  # 10*(10+1)/2 = 55 features
    """

    def __init__(
        self,
        n_channels: int,
        subspace_dim: Optional[int] = None,
        threshold: float = 1e-4,
    ) -> None:
        super().__init__()
        if subspace_dim is None:
            subspace_dim = n_channels

        self.n_channels = n_channels
        self.subspace_dim = subspace_dim
        self.threshold = threshold

        self.cov = CovLayer()
        self.bimap = BiMap(n_channels, subspace_dim)
        self.reeig = ReEig(threshold=threshold)
        self.logeig = LogEig(upper=True, flatten=True)

        # Output dimension: upper triangular elements
        self.output_dim = subspace_dim * (subspace_dim + 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from raw EEG.

        Parameters
        ----------
        x : torch.Tensor
            Raw EEG of shape (batch, channels, time).

        Returns
        -------
        torch.Tensor
            Tangent space features of shape (batch, output_dim).
        """
        cov = self.cov(x)
        spd = self.bimap(cov)
        spd = self.reeig(spd)
        features = self.logeig(spd)
        return features


class SPDFeatureExtractorWithBN(torch.nn.Module):
    """SPD feature extractor with batch normalization for domain adaptation.

    This extractor includes SPDBatchNormMeanVar, which normalizes SPD matrices
    using the Fréchet mean on the Riemannian manifold. The running
    statistics can be adapted to a target domain without retraining.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    subspace_dim : int, optional
        Dimension of BiMap output. If None, uses full channel dimension.
    threshold : float, default=1e-4
        ReEig threshold for numerical stability.
    bn_momentum : float, default=0.1
        Momentum for SPDBatchNormMeanVar running statistics update.
        Higher values adapt faster but may be less stable.

    Notes
    -----
    For domain adaptation:

    1. Train the extractor on source domain (train mode)
    2. Reset running stats and adapt on target domain (train mode for BN only)
    3. Extract target features (eval mode)
    """

    def __init__(
        self,
        n_channels: int,
        subspace_dim: Optional[int] = None,
        threshold: float = 1e-4,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if subspace_dim is None:
            subspace_dim = n_channels

        self.n_channels = n_channels
        self.subspace_dim = subspace_dim
        self.threshold = threshold

        self.cov = CovLayer()
        self.bimap = BiMap(n_channels, subspace_dim)
        self.reeig = ReEig(threshold=threshold)
        self.spdbn = SPDBatchNormMeanVar(
            subspace_dim, momentum=bn_momentum, affine=True
        )
        self.logeig = LogEig(upper=True, flatten=True)

        self.output_dim = subspace_dim * (subspace_dim + 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with batch normalization.

        Parameters
        ----------
        x : torch.Tensor
            Raw EEG of shape (batch, channels, time).

        Returns
        -------
        torch.Tensor
            Normalized tangent space features of shape (batch, output_dim).
        """
        cov = self.cov(x)
        spd = self.bimap(cov)
        spd = self.reeig(spd)
        spd = self.spdbn(spd)
        features = self.logeig(spd)
        return features


def extract_features(
    extractor: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract features from data using the given extractor.

    Parameters
    ----------
    extractor : torch.nn.Module
        Feature extractor module (SPDFeatureExtractor or
        SPDFeatureExtractorWithBN).
    X : np.ndarray
        EEG data of shape (n_samples, n_channels, n_times).
    batch_size : int, default=64
        Batch size for processing.

    Returns
    -------
    np.ndarray
        Feature array of shape (n_samples, feature_dim).
    """
    extractor.eval()
    X_tensor = torch.from_numpy(X).float()

    features_list = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            features = extractor(batch)
            features_list.append(features)

    return torch.cat(features_list, dim=0).numpy()


def train_extractor_on_source(
    extractor: torch.nn.Module,
    X_source: np.ndarray,
    n_epochs: int = 5,
    batch_size: int = 32,
) -> torch.nn.Module:
    """Train extractor on source domain to learn BiMap weights and BN stats.

    This function passes source data through the extractor multiple times
    to update the BiMap weights (if in train mode) and SPDBatchNormMeanVar running
    statistics.

    Parameters
    ----------
    extractor : torch.nn.Module
        Feature extractor with SPDBatchNormMeanVar.
    X_source : np.ndarray
        Source domain EEG data.
    n_epochs : int, default=5
        Number of passes through the source data.
    batch_size : int, default=32
        Batch size for training.

    Returns
    -------
    torch.nn.Module
        Extractor with updated running statistics.
    """
    extractor.train()
    X_tensor = torch.from_numpy(X_source).float()

    for epoch in range(n_epochs):
        # Shuffle data
        perm = torch.randperm(len(X_tensor))
        X_shuffled = X_tensor[perm]

        for i in range(0, len(X_shuffled), batch_size):
            batch = X_shuffled[i : i + batch_size]
            with torch.no_grad():
                _ = extractor(batch)

    return extractor


def adapt_spdbn_to_target(
    extractor: torch.nn.Module,
    X_target: np.ndarray,
    n_passes: int = 10,
    batch_size: int = 32,
    reset_stats: bool = True,
) -> torch.nn.Module:
    """Adapt SPDBatchNormMeanVar statistics to target domain.

    This implements Source-Free Unsupervised Domain Adaptation (SFUDA)
    by updating only the SPDBatchNormMeanVar running statistics on unlabeled
    target data. No labels are required from the target domain.

    Parameters
    ----------
    extractor : torch.nn.Module
        Feature extractor with SPDBatchNormMeanVar layer(s).
    X_target : np.ndarray
        Target domain data (unlabeled).
    n_passes : int, default=10
        Number of passes through target data for statistics update.
        More passes provide better statistics estimation.
    batch_size : int, default=32
        Batch size for adaptation.
    reset_stats : bool, default=True
        If True, reset running statistics before adaptation.
        This allows the model to fully adapt to the target domain.

    Returns
    -------
    torch.nn.Module
        Extractor with SPDBatchNormMeanVar adapted to target domain.

    Notes
    -----
    The adaptation works by:

    1. Setting the model to eval mode (freezing BiMap weights)
    2. Setting SPDBatchNormMeanVar to train mode (updating running stats)
    3. Passing target data through to update the Fréchet mean
    4. The running mean adapts to center target data at identity
    """
    # Freeze all layers except SPDBatchNormMeanVar
    extractor.eval()

    # Find and configure SPDBatchNormMeanVar layers
    spdbn_modules = []
    for module in extractor.modules():
        if isinstance(module, SPDBatchNormMeanVar):
            spdbn_modules.append(module)
            if reset_stats:
                module.reset_running_stats()
            module.train()  # Enable running stats update

    if len(spdbn_modules) == 0:
        print("  Warning: No SPDBatchNormMeanVar layers found!")
        return extractor

    print(f"  Adapting {len(spdbn_modules)} SPDBatchNormMeanVar layer(s)...")

    X_tensor = torch.from_numpy(X_target).float()

    # Multiple passes through target data
    with torch.no_grad():
        for pass_idx in range(n_passes):
            perm = torch.randperm(len(X_tensor))
            X_shuffled = X_tensor[perm]
            for i in range(0, len(X_shuffled), batch_size):
                batch = X_shuffled[i : i + batch_size]
                _ = extractor(batch)

    # Set everything back to eval mode
    extractor.eval()
    return extractor


######################################################################
# Feature Extraction Setup
# ------------------------
#
# We experiment with different subspace dimensions to find the optimal
# trade-off between feature richness and generalization.
#

n_channels = X.shape[1]

# Try different subspace dimensions
# Larger subspace = more features but may overfit
# Smaller subspace = fewer features but better generalization
subspace_dim = 12  # Reduced from 22 to 12 for better generalization

print(f"\nUsing subspace dimension: {subspace_dim}")
print(f"Output feature dimension: {subspace_dim * (subspace_dim + 1) // 2}")

######################################################################
# Baseline: No Domain Adaptation
# ------------------------------
#
# First, we establish a baseline by training on source features and
# testing on target features without any adaptation. This shows the
# performance drop due to domain shift.
#

print("\n" + "=" * 60)
print("Extracting SPD Features (Baseline - No BatchNorm)")
print("=" * 60)

# Create extractor WITHOUT batch normalization for baseline
extractor_baseline = SPDFeatureExtractor(n_channels, subspace_dim)
extractor_baseline.eval()

# Extract features (same extractor for both domains to ensure fair comparison)
features_source = extract_features(extractor_baseline, X_source)
features_target = extract_features(extractor_baseline, X_target)

print(f"Source features shape: {features_source.shape}")
print(f"Target features shape: {features_target.shape}")

# Standardize features
scaler = StandardScaler()
features_source_scaled = scaler.fit_transform(features_source)
features_target_scaled = scaler.transform(features_target)

# Train classifier on source
clf_baseline = LogisticRegression(max_iter=1000, random_state=SEED)
clf_baseline.fit(features_source_scaled, y_source)

# Evaluate
y_pred_source = clf_baseline.predict(features_source_scaled)
y_pred_target_baseline = clf_baseline.predict(features_target_scaled)

source_acc = accuracy_score(y_source, y_pred_source)
target_acc_baseline = accuracy_score(y_target, y_pred_target_baseline)
target_bal_acc_baseline = balanced_accuracy_score(y_target, y_pred_target_baseline)

print("\n" + "-" * 50)
print("Baseline Results (No Domain Adaptation)")
print("-" * 50)
print(f"Source Accuracy:            {source_acc * 100:.2f}%")
print(f"Target Accuracy:            {target_acc_baseline * 100:.2f}%")
print(f"Target Balanced Accuracy:   {target_bal_acc_baseline * 100:.2f}%")
print(f"Performance Drop:           {(source_acc - target_acc_baseline) * 100:.2f}%")

# Store results for summary
results: Dict[str, Dict[str, float]] = {
    "No Adaptation": {
        "accuracy": target_acc_baseline,
        "balanced_accuracy": target_bal_acc_baseline,
        "improvement": 0.0,
    }
}

######################################################################
# Visualizing Domain Shift
# ------------------------
#
# Before applying domain adaptation, let's visualize the distribution
# shift between source and target domains using PCA projection.
#


def plot_domain_shift_comprehensive(
    features_source: np.ndarray,
    features_target: np.ndarray,
    y_source: np.ndarray,
    y_target: np.ndarray,
    class_names: List[str],
    title: str = "Domain Shift Visualization",
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Comprehensive visualization of domain shift.

    Creates a 2x3 grid showing:
    - Domain distributions (PCA)
    - Source and target by class
    - Feature histograms per domain
    - Class-conditional distributions

    Parameters
    ----------
    features_source : np.ndarray
        Source domain features.
    features_target : np.ndarray
        Target domain features.
    y_source : np.ndarray
        Source domain labels.
    y_target : np.ndarray
        Target domain labels.
    class_names : List[str]
        Names of the classes.
    title : str
        Overall figure title.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    # Combine features for PCA
    features_all = np.vstack([features_source, features_target])
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_all)

    n_source = len(features_source)
    source_2d = features_2d[:n_source]
    target_2d = features_2d[n_source:]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # --- Row 1 ---
    # Plot 1: All data by domain
    ax1 = axes[0, 0]
    ax1.scatter(
        source_2d[:, 0],
        source_2d[:, 1],
        c="blue",
        alpha=0.5,
        label="Source",
        marker="o",
        s=40,
    )
    ax1.scatter(
        target_2d[:, 0],
        target_2d[:, 1],
        c="red",
        alpha=0.5,
        label="Target",
        marker="s",
        s=40,
    )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax1.set_title("Domain Distribution", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Source domain by class
    ax2 = axes[0, 1]
    n_classes = len(np.unique(y_source))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 4)))[:n_classes]
    for label_idx, label in enumerate(np.unique(y_source)):
        mask = y_source == label
        ax2.scatter(
            source_2d[mask, 0],
            source_2d[mask, 1],
            c=colors[label_idx],
            alpha=0.6,
            label=f"{class_names[label_idx]}",
            marker="o",
            s=40,
        )
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax2.set_title("Source Domain (by class)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Target domain by class
    ax3 = axes[0, 2]
    for label_idx, label in enumerate(np.unique(y_target)):
        mask = y_target == label
        ax3.scatter(
            target_2d[mask, 0],
            target_2d[mask, 1],
            c=colors[label_idx],
            alpha=0.6,
            label=f"{class_names[label_idx]}",
            marker="s",
            s=40,
        )
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax3.set_title("Target Domain (by class)", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Row 2 ---
    # Plot 4: Feature histogram (first 3 features)
    ax4 = axes[1, 0]
    for feat_idx in range(min(3, features_source.shape[1])):
        ax4.hist(
            features_source[:, feat_idx],
            bins=20,
            alpha=0.5,
            label=f"Source feat {feat_idx}",
            density=True,
        )
        ax4.hist(
            features_target[:, feat_idx],
            bins=20,
            alpha=0.5,
            label=f"Target feat {feat_idx}",
            density=True,
            linestyle="--",
        )
    ax4.set_xlabel("Feature Value")
    ax4.set_ylabel("Density")
    ax4.set_title("Feature Distribution (first 3)", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Class-conditional distributions (Source)
    ax5 = axes[1, 1]
    for label_idx, label in enumerate(np.unique(y_source)):
        mask_s = y_source == label
        ax5.hist(
            source_2d[mask_s, 0],
            bins=15,
            alpha=0.5,
            label=f"Source-{class_names[label_idx]}",
            color=colors[label_idx],
            density=True,
        )
    ax5.set_xlabel("PC1")
    ax5.set_ylabel("Density")
    ax5.set_title("Class Distribution (Source)", fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Class-conditional distributions (Target)
    ax6 = axes[1, 2]
    for label_idx, label in enumerate(np.unique(y_target)):
        mask_t = y_target == label
        ax6.hist(
            target_2d[mask_t, 0],
            bins=15,
            alpha=0.5,
            label=f"Target-{class_names[label_idx]}",
            color=colors[label_idx],
            density=True,
        )
    ax6.set_xlabel("PC1")
    ax6.set_ylabel("Density")
    ax6.set_title("Class Distribution (Target)", fontweight="bold")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# Visualize original domain shift
class_names = [str(c) for c in le.classes_]
fig = plot_domain_shift_comprehensive(
    features_source,
    features_target,
    y_source,
    y_target,
    class_names=class_names,
    title="SPD Feature Space - Before Domain Adaptation",
)
plt.show()

######################################################################
# SPDBatchNormMeanVar Domain Adaptation
# ------------------------------
#
# SPDBatchNormMeanVar provides a native Riemannian approach to domain adaptation
# that operates directly on SPD matrices. The key insight is that
# subject-to-subject variability manifests as a shift in the Fréchet mean
# of the SPD distribution.
#
# **How SPDBatchNormMeanVar adaptation works:**
#
# 1. During training on source, SPDBatchNormMeanVar learns running statistics
#    (Fréchet mean and dispersion) of the source distribution
# 2. For adaptation, we reset the running statistics and pass unlabeled
#    target data through the layer
# 3. The new running mean captures the target distribution's center
# 4. At inference, target data is normalized to have the same geometric
#    center as source data was during training
#
# This is **Source-Free Unsupervised Domain Adaptation (SFUDA)** because:
#
# - No source data is needed during adaptation (source-free)
# - No target labels are needed (unsupervised)
#

print("\n" + "=" * 60)
print("SPDBatchNormMeanVar Domain Adaptation")
print("=" * 60)

# Create extractor WITH batch normalization
extractor_bn = SPDFeatureExtractorWithBN(n_channels, subspace_dim, bn_momentum=0.1)

# Step 1: Train on source domain to establish baseline statistics
print("\nStep 1: Training on source domain...")
extractor_bn = train_extractor_on_source(
    extractor_bn, X_source, n_epochs=5, batch_size=32
)

# Extract source features (for training classifier)
features_source_bn = extract_features(extractor_bn, X_source)

# Step 2: Extract target features WITHOUT adaptation (for comparison)
print("Step 2: Extracting target features without adaptation...")
features_target_no_adapt = extract_features(extractor_bn, X_target)

# Step 3: Adapt to target domain
print("Step 3: Adapting SPDBatchNormMeanVar to target domain...")
extractor_bn = adapt_spdbn_to_target(
    extractor_bn, X_target, n_passes=10, reset_stats=True
)

# Step 4: Extract target features AFTER adaptation
print("Step 4: Extracting target features after adaptation...")
features_target_bn = extract_features(extractor_bn, X_target)

# Train classifier on source features (with BN)
scaler_bn = StandardScaler()
features_source_bn_scaled = scaler_bn.fit_transform(features_source_bn)
features_target_no_adapt_scaled = scaler_bn.transform(features_target_no_adapt)
features_target_bn_scaled = scaler_bn.transform(features_target_bn)

clf_spdbn = LogisticRegression(max_iter=1000, random_state=SEED)
clf_spdbn.fit(features_source_bn_scaled, y_source)

# Evaluate without adaptation
y_pred_target_no_adapt = clf_spdbn.predict(features_target_no_adapt_scaled)
acc_no_adapt = accuracy_score(y_target, y_pred_target_no_adapt)

# Evaluate with adaptation
y_pred_spdbn = clf_spdbn.predict(features_target_bn_scaled)
spdbn_acc = accuracy_score(y_target, y_pred_spdbn)
spdbn_bal_acc = balanced_accuracy_score(y_target, y_pred_spdbn)

print("\n" + "-" * 50)
print("SPDBatchNormMeanVar Results")
print("-" * 50)
print(f"Before adaptation:        {acc_no_adapt * 100:.2f}%")
print(f"After adaptation:         {spdbn_acc * 100:.2f}%")
print(f"Balanced Accuracy:        {spdbn_bal_acc * 100:.2f}%")
print(f"Improvement:              {(spdbn_acc - acc_no_adapt) * 100:+.2f}%")
print(f"vs Baseline (no BN):      {(spdbn_acc - target_acc_baseline) * 100:+.2f}%")

results["SPDBatchNormMeanVar"] = {
    "accuracy": spdbn_acc,
    "balanced_accuracy": spdbn_bal_acc,
    "improvement": spdbn_acc - target_acc_baseline,
}

######################################################################
# Visualizing Adapted Features
# ----------------------------
#

fig = plot_domain_shift_comprehensive(
    features_source_bn,
    features_target_bn,
    y_source,
    y_target,
    class_names=class_names,
    title="SPD Feature Space - After SPDBatchNormMeanVar Adaptation",
)
plt.show()

######################################################################
# Domain Adaptation with skada
# ----------------------------
#
# If skada is installed, we can compare SPDBatchNormMeanVar with other domain
# adaptation methods. These methods operate on the Euclidean features
# (after LogEig projection) rather than directly on SPD matrices.
#
# **Key concept:** skada uses `sample_domain` to distinguish domains:
#
# - **Positive values (1)**: Source domain samples
# - **Negative values (-1)**: Target domain samples
#

print("\n" + "=" * 60)
print("skada Domain Adaptation Methods")
print("=" * 60)

# Prepare data in skada format
X_combined = np.vstack([features_source, features_target])
y_combined = np.concatenate([y_source, -np.ones(len(y_target))])
sample_domain = np.concatenate(
    [np.ones(len(features_source)), -np.ones(len(features_target))]
)

print("\nData prepared for skada:")
print(f"  Combined features shape: {X_combined.shape}")
print(f"  Source samples (domain=1): {int(np.sum(sample_domain > 0))}")
print(f"  Target samples (domain=-1): {int(np.sum(sample_domain < 0))}")

######################################################################
# CORAL (Correlation Alignment)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# CORAL :cite:p:`sun2016coral` aligns the second-order statistics (covariance)
# of the
# source and target feature distributions. This is particularly
# suitable for SPD-derived features since they already capture
# covariance structure.
#
# .. math::
#
#     \min_A \|A^T C_S A - C_T\|_F^2
#
# where :math:`C_S` and :math:`C_T` are the covariance matrices of
# source and target features, respectively.
#

print("\n" + "-" * 50)
print("CORAL (Correlation Alignment)")
print("-" * 50)

coral_pipeline = make_da_pipeline(
    StandardScaler(),
    CORALAdapter(reg=1e-3),
    LogisticRegression(max_iter=1000, random_state=SEED),
)

coral_pipeline.fit(X_combined, y_combined, sample_domain=sample_domain)
y_pred_coral = coral_pipeline.predict(features_target)

coral_acc = accuracy_score(y_target, y_pred_coral)
coral_bal_acc = balanced_accuracy_score(y_target, y_pred_coral)

print(f"Target Accuracy:          {coral_acc * 100:.2f}%")
print(f"Target Balanced Accuracy: {coral_bal_acc * 100:.2f}%")
print(f"Improvement over baseline: {(coral_acc - target_acc_baseline) * 100:+.2f}%")

results["CORAL"] = {
    "accuracy": coral_acc,
    "balanced_accuracy": coral_bal_acc,
    "improvement": coral_acc - target_acc_baseline,
}

######################################################################
# Subspace Alignment
# ~~~~~~~~~~~~~~~~~~
#
# Subspace Alignment :cite:p:`fernando2013subspace` learns a linear
# transformation that aligns
# the principal subspaces of source and target domains.
#
# The method:
#
# 1. Computes PCA bases for source and target domains
# 2. Finds optimal rotation to align the subspaces
# 3. Projects source data into the aligned space
#

print("\n" + "-" * 50)
print("Subspace Alignment")
print("-" * 50)

sa_clf = SubspaceAlignment(
    base_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
    n_components=min(10, features_source.shape[1]),
)

sa_clf.fit(X_combined, y_combined, sample_domain=sample_domain)
y_pred_sa = sa_clf.predict(features_target)

sa_acc = accuracy_score(y_target, y_pred_sa)
sa_bal_acc = balanced_accuracy_score(y_target, y_pred_sa)

print(f"Target Accuracy:          {sa_acc * 100:.2f}%")
print(f"Target Balanced Accuracy: {sa_bal_acc * 100:.2f}%")
print(f"Improvement over baseline: {(sa_acc - target_acc_baseline) * 100:+.2f}%")

results["Subspace Alignment"] = {
    "accuracy": sa_acc,
    "balanced_accuracy": sa_bal_acc,
    "improvement": sa_acc - target_acc_baseline,
}

######################################################################
# Entropic Optimal Transport
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Optimal Transport :cite:p:`courty2017optimal` finds the minimum cost mapping
# between source
# and target distributions. The entropic regularization makes the
# optimization problem tractable and provides smoother mappings.
#

print("\n" + "-" * 50)
print("Entropic Optimal Transport")
print("-" * 50)

try:
    ot_clf = EntropicOTMapping(
        base_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        reg_e=1.0,
    )

    ot_clf.fit(X_combined, y_combined, sample_domain=sample_domain)
    y_pred_ot = ot_clf.predict(features_target)

    ot_acc = accuracy_score(y_target, y_pred_ot)
    ot_bal_acc = balanced_accuracy_score(y_target, y_pred_ot)

    print(f"Target Accuracy:          {ot_acc * 100:.2f}%")
    print(f"Target Balanced Accuracy: {ot_bal_acc * 100:.2f}%")
    print(f"Improvement over baseline: {(ot_acc - target_acc_baseline) * 100:+.2f}%")

    results["Entropic OT"] = {
        "accuracy": ot_acc,
        "balanced_accuracy": ot_bal_acc,
        "improvement": ot_acc - target_acc_baseline,
    }
except Exception as e:
    print(f"OT method failed: {e}")

######################################################################
# Confusion Matrices
# ------------------
#
# Let's visualize the confusion matrices to understand where each
# method makes mistakes.
#


def plot_confusion_matrices(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    class_names: List[str],
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """Plot confusion matrices for multiple methods.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    predictions : Dict[str, np.ndarray]
        Dictionary mapping method names to predicted labels.
    class_names : List[str]
        Names of classes.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    n_methods = len(predictions)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)

    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted",
            ylabel="True",
        )

        # Add text annotations
        thresh = cm_normalized.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})",
                    ha="center",
                    va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=9,
                )

        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f"{method_name}\nAcc: {acc * 100:.1f}%", fontweight="bold")

    plt.suptitle("Confusion Matrices - Target Domain", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# Prepare predictions for confusion matrix plot
predictions_dict = {
    "No Adaptation": y_pred_target_baseline,
    "SPDBatchNormMeanVar": y_pred_spdbn,
}

predictions_dict["CORAL"] = y_pred_coral
predictions_dict["Subspace Align"] = y_pred_sa
if "Entropic OT" in results:
    predictions_dict["Entropic OT"] = y_pred_ot

fig = plot_confusion_matrices(y_target, predictions_dict, class_names)
plt.show()

######################################################################
# Results Summary
# ---------------
#
# Compare all domain adaptation methods in a comprehensive table.
#

print("\n" + "=" * 70)
print("Domain Adaptation Results Summary")
print("=" * 70)
print(f"{'Method':<25} {'Accuracy':>12} {'Balanced Acc':>14} {'Improvement':>12}")
print("-" * 70)

for method_name, method_results in results.items():
    acc = method_results["accuracy"] * 100
    bal_acc = method_results["balanced_accuracy"] * 100
    imp = method_results["improvement"] * 100
    imp_str = f"{imp:+.2f}%" if method_name != "No Adaptation" else "-"
    print(f"{method_name:<25} {acc:>10.2f}% {bal_acc:>12.2f}% {imp_str:>12}")

print("-" * 70)
print(f"{'Chance Level':<25} {'50.00%':>12}")
print(f"{'Source Accuracy':<25} {source_acc * 100:>10.2f}%")

# Find best method
best_method = max(results.keys(), key=lambda k: results[k]["accuracy"])
print(f"\nBest method: {best_method} ({results[best_method]['accuracy'] * 100:.2f}%)")

######################################################################
# Accuracy Comparison Bar Chart
# -----------------------------
#


def plot_accuracy_comparison(
    results: Dict[str, Dict[str, float]],
    source_acc: float,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot accuracy comparison bar chart.

    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Results dictionary with accuracy for each method.
    source_acc : float
        Source domain accuracy.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    methods = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in methods]
    improvements = [results[m]["improvement"] * 100 for m in methods]

    # Color scheme
    colors = []
    for imp in improvements:
        if imp == 0:
            colors.append("#e74c3c")  # Red for baseline
        elif imp > 0:
            colors.append("#2ecc71")  # Green for positive
        else:
            colors.append("#f39c12")  # Orange for negative

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(methods, accuracies, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar, acc, imp in zip(bars, accuracies, improvements):
        label = f"{acc:.1f}%"
        if imp != 0:
            label += f"\n({imp:+.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Reference lines
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="Chance (50%)")
    ax.axhline(
        y=source_acc * 100,
        color="blue",
        linestyle=":",
        alpha=0.7,
        label=f"Source Acc ({source_acc * 100:.1f}%)",
    )

    ax.set_ylabel("Target Accuracy (%)", fontsize=12)
    ax.set_xlabel("Domain Adaptation Method", fontsize=12)
    ax.set_title(
        "Domain Adaptation Method Comparison\n(Cross-Subject EEG Classification)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim([0, 110])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate x-axis labels if many methods
    if len(methods) > 4:
        plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    return fig


fig = plot_accuracy_comparison(results, source_acc)
plt.show()

######################################################################
# Before vs After Adaptation Comparison
# -------------------------------------
#
# Direct visual comparison of feature distributions before and after
# SPDBatchNormMeanVar adaptation.
#


def plot_before_after_comparison(
    features_source: np.ndarray,
    features_target_before: np.ndarray,
    features_target_after: np.ndarray,
    y_source: np.ndarray,
    y_target: np.ndarray,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot before/after comparison of domain adaptation.

    Parameters
    ----------
    features_source : np.ndarray
        Source domain features.
    features_target_before : np.ndarray
        Target features before adaptation.
    features_target_after : np.ndarray
        Target features after adaptation.
    y_source : np.ndarray
        Source labels.
    y_target : np.ndarray
        Target labels.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Fit PCA on all data
    all_features = np.vstack(
        [features_source, features_target_before, features_target_after]
    )
    pca = PCA(n_components=2)
    pca.fit(all_features)

    source_2d = pca.transform(features_source)
    target_before_2d = pca.transform(features_target_before)
    target_after_2d = pca.transform(features_target_after)

    # Plot 1: Source distribution
    ax1 = axes[0]
    ax1.scatter(
        source_2d[:, 0],
        source_2d[:, 1],
        c="blue",
        alpha=0.6,
        label="Source",
        s=30,
    )
    ax1.scatter(
        source_2d[:, 0].mean(),
        source_2d[:, 1].mean(),
        c="blue",
        s=200,
        marker="*",
        edgecolors="black",
        label="Source mean",
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Source Domain", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Before adaptation
    ax2 = axes[1]
    ax2.scatter(
        source_2d[:, 0],
        source_2d[:, 1],
        c="blue",
        alpha=0.3,
        label="Source",
        s=20,
    )
    ax2.scatter(
        target_before_2d[:, 0],
        target_before_2d[:, 1],
        c="red",
        alpha=0.6,
        label="Target",
        s=30,
    )
    # Mark means
    ax2.scatter(
        source_2d[:, 0].mean(),
        source_2d[:, 1].mean(),
        c="blue",
        s=200,
        marker="*",
        edgecolors="black",
    )
    ax2.scatter(
        target_before_2d[:, 0].mean(),
        target_before_2d[:, 1].mean(),
        c="red",
        s=200,
        marker="*",
        edgecolors="black",
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Before Adaptation", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Compute domain shift (distance between means)
    shift_before = np.linalg.norm(
        source_2d.mean(axis=0) - target_before_2d.mean(axis=0)
    )
    ax2.text(
        0.05,
        0.95,
        f"Shift: {shift_before:.2f}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Plot 3: After adaptation
    ax3 = axes[2]
    ax3.scatter(
        source_2d[:, 0],
        source_2d[:, 1],
        c="blue",
        alpha=0.3,
        label="Source",
        s=20,
    )
    ax3.scatter(
        target_after_2d[:, 0],
        target_after_2d[:, 1],
        c="green",
        alpha=0.6,
        label="Target (adapted)",
        s=30,
    )
    # Mark means
    ax3.scatter(
        source_2d[:, 0].mean(),
        source_2d[:, 1].mean(),
        c="blue",
        s=200,
        marker="*",
        edgecolors="black",
    )
    ax3.scatter(
        target_after_2d[:, 0].mean(),
        target_after_2d[:, 1].mean(),
        c="green",
        s=200,
        marker="*",
        edgecolors="black",
    )
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("After Adaptation", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    shift_after = np.linalg.norm(source_2d.mean(axis=0) - target_after_2d.mean(axis=0))
    ax3.text(
        0.05,
        0.95,
        f"Shift: {shift_after:.2f}",
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    plt.suptitle(
        "SPDBatchNormMeanVar Domain Adaptation Effect",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


# We need to re-extract features for proper comparison
# Reset and re-train for clean comparison
extractor_compare = SPDFeatureExtractorWithBN(n_channels, subspace_dim, bn_momentum=0.1)
extractor_compare = train_extractor_on_source(extractor_compare, X_source, n_epochs=5)

# Get features before adaptation
features_source_compare = extract_features(extractor_compare, X_source)
features_target_before_compare = extract_features(extractor_compare, X_target)

# Adapt and get features after
extractor_compare = adapt_spdbn_to_target(extractor_compare, X_target, n_passes=10)
features_target_after_compare = extract_features(extractor_compare, X_target)

fig = plot_before_after_comparison(
    features_source_compare,
    features_target_before_compare,
    features_target_after_compare,
    y_source,
    y_target,
)
plt.show()

######################################################################
# Summary Table
# -------------
#
# Comprehensive summary of all methods with their characteristics.
#

print("\n" + "=" * 80)
print("Method Summary Table")
print("=" * 80)
print(f"{'Method':<20} {'Type':<15} {'Requires':<15} {'Accuracy':>10} {'Notes':<20}")
print("-" * 80)

method_info = {
    "No Adaptation": ("Baseline", "Nothing", "Direct transfer"),
    "SPDBatchNormMeanVar": ("Riemannian", "Unlabeled target", "Native SPD adaptation"),
}
method_info.update(
    {
        "CORAL": ("Statistical", "Unlabeled target", "Covariance alignment"),
        "Subspace Alignment": ("Geometric", "Unlabeled target", "PCA-based"),
        "Entropic OT": ("Optimal Transport", "Unlabeled target", "Sample mapping"),
    }
)

for method_name, (method_type, requires, notes) in method_info.items():
    if method_name in results:
        acc = results[method_name]["accuracy"] * 100
        print(
            f"{method_name:<20} {method_type:<15} {requires:<15} {acc:>8.1f}% {notes}"
        )

print("-" * 80)

######################################################################
# Discussion
# ----------
#
# **Key insights from this tutorial:**
#
# **SPD Features for Domain Adaptation:**
#
# - SPD-based features capture the geometric structure of EEG covariance
#   matrices, which encode both power and connectivity information
# - The tangent space projection (LogEig) provides Euclidean features
#   suitable for standard domain adaptation methods
# - SPDBatchNormMeanVar provides a native Riemannian approach that operates
#   directly on the SPD manifold
#
# **Method Comparison:**
#
# - **SPDBatchNormMeanVar** adapts the Fréchet mean of the SPD distribution,
#   which is the natural center on the Riemannian manifold
# - **CORAL** aligns covariance matrices in the Euclidean feature space,
#   complementing the geometric features
# - **Subspace Alignment** finds a shared linear subspace, useful when
#   domains have similar structure but different orientations
# - **Optimal Transport** provides principled sample-to-sample mapping
#
# **Practical Recommendations:**
#
# 1. **Start with SPDBatchNormMeanVar** for SPD networks (native, no extra
#    dependencies, efficient)
# 2. **Use CORAL** when second-order alignment is sufficient
# 3. **Try Subspace Alignment** for high-dimensional scenarios
# 4. **Combine multiple methods** for robust performance
#
# **Limitations:**
#
# - Domain adaptation assumes source and target share the same classes
# - Large distribution shifts may require more sophisticated methods
# - Performance depends on the quality of the source model
#
