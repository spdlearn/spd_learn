"""
.. _tsmnet-domain-adaptation:

Cross-Session Transfer with TSMNet
==================================

This tutorial demonstrates how to use TSMNet for cross-session motor imagery
classification with domain adaptation. TSMNet's SPDBatchNormMeanVar layer enables
adaptation to new sessions without labeled data from the target session.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# In EEG-based BCIs, a common challenge is **session-to-session variability**:
# models trained on one day often perform poorly on another day
# due to changes in electrode impedance, mental state, and environment.
#
# TSMNet :cite:p:`kobler2022spd` addresses this through **SPDBatchNormMeanVar**,
# which:
#
# 1. Normalizes SPD matrices using the Fréchet mean
# 2. Maintains running statistics that can be updated on new data
# 3. Enables **Source-Free Unsupervised Domain Adaptation (SFUDA)**
#
# This means we can adapt to a new subject using only unlabeled data!
#

######################################################################
# Setup and Imports
# -----------------
#

import warnings

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from braindecode import EEGClassifier
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from skada import (
    CORALAdapter,
    EntropicOTMapping,
    SubspaceAlignment,
    make_da_pipeline,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skorch.callbacks import EpochScoring, GradientNormClipping
from skorch.dataset import ValidSplit

from spd_learn.models import TSMNet


warnings.filterwarnings("ignore")

######################################################################
# Loading the Dataset
# -------------------
#
# BNCI2014_001 contains EEG recordings from 9 subjects performing
# - **22 EEG channels**: Standard 10-20 montage
# - **250 Hz sampling rate**: After resampling
#
# We'll demonstrate **cross-session transfer**:
#
# - **Source domain**: Subject 1, Session 1 (training)
# - **Target domain**: Subject 1, Session 2 (testing/adaptation)
#
# Cross-session transfer is a realistic BCI scenario where we want to avoid
# recalibration for a returning user.
#

dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

print(f"Dataset: {dataset.code}")
print("Cross-subject transfer: Subject 1 (source) -> Subject 2 (target)")

######################################################################
# Creating the TSMNet Model
# -------------------------
#
# TSMNet architecture:
#
# 1. **Temporal Conv**: Learns temporal filters
# 2. **Spatial Conv**: Learns spatial combinations
# 3. **CovLayer**: Computes covariance matrices
# 4. **BiMap + ReEig**: SPD dimensionality reduction
# 5. **SPDBatchNormMeanVar**: Riemannian batch normalization (key for adaptation)
# 6. **LogEig**: Projects to tangent space
# 7. **Linear**: Classification head
#

n_chans = 22
n_outputs = 4

model = TSMNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    n_temp_filters=8,  # Temporal filters
    temp_kernel_length=50,  # ~200ms at 250Hz
    n_spatiotemp_filters=32,  # Spatiotemporal features
    n_bimap_filters=16,  # BiMap output dimension
    reeig_threshold=1e-4,  # ReEig threshold
)

print("TSMNet Architecture:")
print(model)

######################################################################
# Training on Source Domain
# -------------------------
#
# First, we train TSMNet on Session 1 (source domain).
#

source_subject = 1
target_subject = 1  # Same subject, different session
batch_size = 32
max_epochs = 300
learning_rate = 1e-4  # Optimal learning rate from grid search
weight_decay = 1e-4  # L2 regularization for better generalization

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# Cache configuration
cache_config = dict(
    save_raw=True,
    save_epochs=True,
    save_array=True,
    use=True,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
)

# Load data for both subjects
X, labels, meta = paradigm.get_data(
    dataset=dataset,
    subjects=[source_subject, target_subject],
    cache_config=cache_config,
)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Split by session
# Session '0train' is the first session (source)
# Session '1test' is the second session (target)
source_idx = meta.query("session == '0train'").index.to_numpy()
target_idx = meta.query("session == '1test'").index.to_numpy()

X_source, y_source = X[source_idx], y[source_idx]
X_target, y_target = X[target_idx], y[target_idx]

print(f"\nSource domain (Session 1): {len(source_idx)} samples")
print(f"Target domain (Session 2): {len(target_idx)} samples")

# Create classifier
# Note: SPD networks benefit from gradient clipping to prevent
# divergence during training on the Riemannian manifold.
clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=learning_rate,
    optimizer__weight_decay=weight_decay,
    train_split=ValidSplit(0.1, stratified=True, random_state=42),
    batch_size=batch_size,
    max_epochs=max_epochs,
    callbacks=[
        (
            "train_acc",
            EpochScoring(
                "accuracy", lower_is_better=False, on_train=True, name="train_acc"
            ),
        ),
        ("gradient_clip", GradientNormClipping(gradient_clip_value=1.0)),
    ],
    device=device,
    verbose=1,
)

# Train on source domain
print("\n" + "=" * 50)
print("Training on Source Domain")
print("=" * 50)
clf.fit(X_source, y_source)

######################################################################
# Evaluating Without Adaptation
# -----------------------------
#
# Let's first see how the model performs on the target domain
# WITHOUT any adaptation.
#

# Evaluate on source (should be high)
y_pred_source = clf.predict(X_source)
source_acc = accuracy_score(y_source, y_pred_source)

# Evaluate on target WITHOUT adaptation
y_pred_target_no_adapt = clf.predict(X_target)
target_acc_no_adapt = accuracy_score(y_target, y_pred_target_no_adapt)

print(f"\n{'='*50}")
print("Results WITHOUT Domain Adaptation")
print(f"{'='*50}")
print(f"Source Domain Accuracy: {source_acc*100:.2f}%")
print(f"Target Domain Accuracy: {target_acc_no_adapt*100:.2f}%")
print(f"Performance Drop: {(source_acc - target_acc_no_adapt)*100:.2f}%")

######################################################################
# Domain Adaptation via SPDBatchNormMeanVar
# -----------------------------------------
#
# Now we perform **Source-Free Unsupervised Domain Adaptation (SFUDA)**:
#
# 1. Put the model in eval mode (freeze all parameters)
# 2. Put SPDBatchNormMeanVar in train mode (update running statistics)
# 3. Pass target domain data through the model (no labels needed!)
# 4. The running mean adapts to the target domain distribution
#


def adapt_spdbn(
    model,
    X_target,
    n_passes=10,
    reset_stats=False,
    adapt_momentum=0.8,
    batch_size=64,
):
    """Adapt SPDBatchNormMeanVar statistics to target domain.

    Parameters
    ----------
    model : nn.Module
        TSMNet model with SPDBatchNormMeanVar layer.
    X_target : array
        Target domain data (unlabeled).
    n_passes : int
        Number of passes through the data for statistics update.
    reset_stats : bool
        If True, reset running statistics before adaptation.
        Default is False to preserve source domain knowledge.
    adapt_momentum : float
        Momentum to use during adaptation (higher = faster adaptation).
    batch_size : int
        Batch size for adaptation (larger = more stable statistics).

    Returns
    -------
    model : nn.Module
        The adapted model with updated SPDBatchNormMeanVar statistics.
    """
    model.eval()  # Freeze other layers

    # Find SPDBatchNormMeanVar layers and configure for adaptation
    spdbn_modules = []
    original_momentums = []
    for module in model.modules():
        class_name = module.__class__.__name__
        if "SPDBatchNormMeanVar" in class_name:
            spdbn_modules.append(module)
            original_momentums.append(module.momentum)

            if reset_stats:
                # Reset to identity mean and unit variance for fresh adaptation
                module.reset_running_stats()

            module.train()  # Enable running stats update

    print(f"Found {len(spdbn_modules)} SPDBatchNormMeanVar layer(s) to adapt")

    # Convert to tensor
    X_tensor = torch.tensor(X_target, dtype=torch.float32)
    if next(model.parameters()).is_cuda:
        X_tensor = X_tensor.cuda()

    # Pass data through model multiple times to update statistics
    with torch.no_grad():
        for pass_idx in range(n_passes):
            # Set momentum for this pass
            for module in spdbn_modules:
                module.momentum = adapt_momentum

            # Reshuffle each pass for better statistics
            perm = torch.randperm(len(X_tensor))
            X_shuffled = X_tensor[perm]

            # Process in batches
            for i in range(0, len(X_shuffled), batch_size):
                batch = X_shuffled[i : i + batch_size]
                _ = model(batch)

            if (pass_idx + 1) % 10 == 0 or pass_idx == 0:
                print(f"  Adaptation pass {pass_idx + 1}/{n_passes}")

    # Restore original momentum values
    for module, orig_momentum in zip(spdbn_modules, original_momentums):
        module.momentum = orig_momentum

    model.eval()  # Set everything back to eval
    return model


def predict_with_domain_specific_bn(model, X_data):
    """Predict using domain-specific batch normalization (SPDDSMBN approach).

    This implements the key idea from Kobler et al. (NeurIPS 2022):
    Compute domain-specific statistics on the target domain and use them
    directly for normalization. This is different from standard adaptation
    which tries to blend source and target statistics.

    Parameters
    ----------
    model : nn.Module
        TSMNet model with SPDBatchNormMeanVar layer.
    X_data : array
        Target domain data to predict on.

    Returns
    -------
    predictions : array
        Predicted class labels.
    """
    # Find SPDBatchNormMeanVar layers
    spdbn_modules = []
    for module in model.modules():
        class_name = module.__class__.__name__
        if "SPDBatchNormMeanVar" in class_name:
            spdbn_modules.append(module)

    model.eval()

    # Convert to tensor
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    if next(model.parameters()).is_cuda:
        X_tensor = X_tensor.cuda()

    # Key insight from SPDDSMBN: Compute domain-specific statistics
    # by processing ALL target data with momentum=1.0 (full batch stats)
    # This estimates the target domain's Fréchet mean and variance
    for module in spdbn_modules:
        module.reset_running_stats()  # Start fresh for target domain
        module.momentum = 1.0  # Use full batch statistics
        module.train()  # Enable running stats update

    # Single pass to compute target domain statistics using all data
    with torch.no_grad():
        _ = model(X_tensor)  # This updates running_mean and running_var

    # Now predict using the target domain statistics
    model.eval()  # Back to eval mode - uses the updated running stats

    with torch.no_grad():
        logits = model(X_tensor)
        predictions = logits.argmax(dim=1).cpu().numpy()

    return predictions


def extract_features_from_tsmnet(model, X, batch_size=32):
    """Extract tangent space features from TSMNet (before classification head).

    This extracts the Euclidean features from the tangent space projection,
    which can be used with standard domain adaptation methods like CORAL,
    Subspace Alignment, and Optimal Transport.

    Parameters
    ----------
    model : nn.Module
        TSMNet model.
    X : array
        Input EEG data of shape (n_samples, n_channels, n_times).
    batch_size : int
        Batch size for processing.

    Returns
    -------
    features : np.ndarray
        Tangent space features of shape (n_samples, n_features).
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    device = next(model.parameters()).device
    X_tensor = X_tensor.to(device)

    features_list = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            # Process through TSMNet layers up to LogEig (before classification head)
            x = batch[:, None, ...]  # Add channel dim for CNN
            x = model.cnn(x)
            x = model.covpool(x)
            x = model.spdnet(x)
            x = model.spdbnorm(x)
            x = model.logeig(x)  # Tangent space features
            features_list.append(x.cpu())

    return torch.cat(features_list, dim=0).numpy()


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
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
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
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
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
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
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


# Get the underlying model from the classifier
underlying_model = clf.module_

# Use Domain-Specific Batch Normalization (SPDDSMBN approach from Kobler et al.)
# This computes target-domain-specific statistics and uses them for normalization
print("\n" + "=" * 50)
print("Using Domain-Specific Batch Normalization (SPDDSMBN)")
print("=" * 50)
print("Computing target domain statistics (Fréchet mean and variance)")

######################################################################
# Evaluating After Adaptation
# ---------------------------
#
# .. note::
#
#    Cross-session transfer typically shows distribution shifts
#    that SPDBatchNormMeanVar can correct.
#    The improvement depends on:
#
#    - Non-stationarity between sessions
#    - Training convergence on the source session
#    - How well the learned features generalize
#
#    Typical improvements range from 3-10% for cross-session transfer.
#

# Predict using domain-specific batch normalization
# Key insight: Use target domain statistics directly (not blended with source)
y_pred_target_adapted = predict_with_domain_specific_bn(underlying_model, X_target)

target_acc_adapted = accuracy_score(y_target, y_pred_target_adapted)
improvement = target_acc_adapted - target_acc_no_adapt

print(f"\n{'='*50}")
print("Results WITH Domain Adaptation")
print(f"{'='*50}")
print(f"Target Accuracy (No Adaptation):   {target_acc_no_adapt*100:.2f}%")
print(f"Target Accuracy (With Adaptation): {target_acc_adapted*100:.2f}%")
if improvement >= 0:
    print(f"Improvement: +{improvement*100:.2f}%")
else:
    print(f"Improvement: {improvement*100:.2f}%")

######################################################################
# Domain Adaptation with SKADA
# ----------------------------
#
# Now we compare SPDBatchNormMeanVar/TTBN with domain adaptation methods from
# `skada <https://scikit-adaptation.github.io/>`_ (scikit-learn domain
# adaptation). These methods operate on the Euclidean tangent space
# features extracted from TSMNet.
#
# **Methods compared:**
#
# - **CORAL**: Correlation Alignment - aligns second-order statistics
# - **Subspace Alignment**: Linear subspace mapping between domains
# - **Entropic Optimal Transport**: Sample-to-sample mapping
#

print("\n" + "=" * 50)
print("SKADA Domain Adaptation Methods")
print("=" * 50)

# Extract tangent space features for SKADA methods
features_source = extract_features_from_tsmnet(underlying_model, X_source)
features_target = extract_features_from_tsmnet(underlying_model, X_target)

print(f"Source features shape: {features_source.shape}")
print(f"Target features shape: {features_target.shape}")

######################################################################
# Visualizing Domain Shift
# ------------------------
#
# Before applying domain adaptation, let's visualize the distribution
# shift between source and target sessions using PCA projection.
#

class_names = [str(c) for c in le.classes_]
fig = plot_domain_shift_comprehensive(
    features_source,
    features_target,
    y_source,
    y_target,
    class_names=class_names,
    title="Riemannian Feature Space - Cross-Session Distribution",
)
plt.show()

# Prepare data in SKADA format
# SKADA uses sample_domain to distinguish domains:
# - Positive values (1): Source domain
# - Negative values (-1): Target domain
X_combined = np.vstack([features_source, features_target])
y_combined = np.concatenate([y_source, -np.ones(len(y_target))])
sample_domain = np.concatenate(
    [np.ones(len(features_source)), -np.ones(len(features_target))]
)

# Initialize results dictionary
results = {
    "No Adaptation": target_acc_no_adapt,
    "SPDBatchNormMeanVar (TTBN)": target_acc_adapted,
}

######################################################################
# CORAL (Correlation Alignment)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# CORAL aligns the second-order statistics (covariance) of source and
# target feature distributions. This is particularly suitable for
# SPD-derived features since they already capture covariance structure.
#

print("\n" + "-" * 50)
print("CORAL (Correlation Alignment)")
print("-" * 50)

coral_pipeline = make_da_pipeline(
    StandardScaler(),
    CORALAdapter(reg=1e-3),
    LogisticRegression(max_iter=1000),
)
coral_pipeline.fit(X_combined, y_combined, sample_domain=sample_domain)
y_pred_coral = coral_pipeline.predict(features_target)
coral_acc = accuracy_score(y_target, y_pred_coral)

print(f"CORAL Accuracy: {coral_acc*100:.2f}%")
print(f"Improvement over baseline: {(coral_acc - target_acc_no_adapt)*100:+.2f}%")
results["CORAL"] = coral_acc

######################################################################
# Subspace Alignment
# ~~~~~~~~~~~~~~~~~~
#
# Subspace Alignment learns a linear transformation that aligns the
# principal subspaces of source and target domains.
#

print("\n" + "-" * 50)
print("Subspace Alignment")
print("-" * 50)

sa_clf = SubspaceAlignment(
    base_estimator=LogisticRegression(max_iter=1000),
    n_components=min(10, features_source.shape[1]),
)
sa_clf.fit(X_combined, y_combined, sample_domain=sample_domain)
y_pred_sa = sa_clf.predict(features_target)
sa_acc = accuracy_score(y_target, y_pred_sa)

print(f"Subspace Alignment Accuracy: {sa_acc*100:.2f}%")
print(f"Improvement over baseline: {(sa_acc - target_acc_no_adapt)*100:+.2f}%")
results["Subspace Alignment"] = sa_acc

######################################################################
# Entropic Optimal Transport
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Optimal Transport finds the minimum cost mapping between source and
# target distributions. Entropic regularization makes the optimization
# tractable and provides smoother mappings.
#

print("\n" + "-" * 50)
print("Entropic Optimal Transport")
print("-" * 50)

try:
    ot_clf = EntropicOTMapping(
        base_estimator=LogisticRegression(max_iter=1000),
        reg_e=1.0,
    )
    ot_clf.fit(X_combined, y_combined, sample_domain=sample_domain)
    y_pred_ot = ot_clf.predict(features_target)
    ot_acc = accuracy_score(y_target, y_pred_ot)

    print(f"Entropic OT Accuracy: {ot_acc*100:.2f}%")
    print(f"Improvement over baseline: {(ot_acc - target_acc_no_adapt)*100:+.2f}%")
    results["Entropic OT"] = ot_acc
except Exception as e:
    print(f"Entropic OT failed: {e}")

######################################################################
# Results Summary
# ---------------
#

print("\n" + "=" * 60)
print("Domain Adaptation Results Summary")
print("=" * 60)
print(f"{'Method':<25} {'Accuracy':>12} {'vs Baseline':>14}")
print("-" * 55)
for method, acc in results.items():
    if method == "No Adaptation":
        print(f"{method:<25} {acc*100:>10.2f}% {'-':>14}")
    else:
        imp = acc - target_acc_no_adapt
        print(f"{method:<25} {acc*100:>10.2f}% {imp*100:>+12.2f}%")
print("-" * 55)
print("Chance level: 25.00% (4 classes)")

# Find best method
best_method = max(results.keys(), key=lambda k: results[k])
print(f"\nBest method: {best_method} ({results[best_method]*100:.2f}%)")

######################################################################
# Visualizing Results
# -------------------
#

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Domain Adaptation Methods Comparison
ax1 = axes[0]
methods = list(results.keys())
accuracies = [results[m] * 100 for m in methods]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"][: len(methods)]
bars = ax1.bar(methods, accuracies, color=colors, edgecolor="black", linewidth=1.5)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_title("Domain Adaptation Comparison", fontsize=14)
ax1.set_ylim([0, 100])
ax1.axhline(y=25, color="gray", linestyle="--", alpha=0.5, label="Chance (25%)")
ax1.axhline(
    y=source_acc * 100,
    color="blue",
    linestyle=":",
    alpha=0.5,
    label=f"Source ({source_acc*100:.1f}%)",
)

# Add value labels
for bar, acc in zip(bars, accuracies):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax1.legend(loc="lower right", fontsize=8)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

# 2. Training history
ax2 = axes[1]
history = clf.history
epochs_hist = range(1, len(history) + 1)
ax2.plot(epochs_hist, history[:, "train_loss"], "b-", label="Train Loss", linewidth=2)
ax2.plot(epochs_hist, history[:, "valid_loss"], "r--", label="Valid Loss", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Training History", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Feature space PCA visualization
ax3 = axes[2]
pca = PCA(n_components=2)
features_all = np.vstack([features_source, features_target])
features_2d = pca.fit_transform(features_all)
n_source = len(features_source)

ax3.scatter(
    features_2d[:n_source, 0],
    features_2d[:n_source, 1],
    c="blue",
    alpha=0.5,
    label="Source",
    marker="o",
    s=30,
)
ax3.scatter(
    features_2d[n_source:, 0],
    features_2d[n_source:, 1],
    c="red",
    alpha=0.5,
    label="Target",
    marker="s",
    s=30,
)
ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
ax3.set_title("Feature Space (PCA)", fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

######################################################################
# Understanding SPDBatchNormMeanVar Adaptation
# --------------------------------------------
#
# The key insight is that **session variability manifests as a shift in
# the distribution of SPD matrices**. SPDBatchNormMeanVar counters this by:
#
# 1. **Centering**: Removes the batch mean (Fréchet mean on SPD manifold)
#
#    .. math::
#
#       \tilde{P}_i = G^{-1/2} P_i G^{-1/2}
#
# 2. **Scaling**: Normalizes dispersion
#
#    .. math::
#
#       \hat{P}_i = \tilde{P}_i^{w/\sqrt{\sigma^2 + \varepsilon}}
#
# When we adapt, we update the running mean :math:`G` and variance
# :math:`\sigma^2` to match the target domain, aligning the distributions
# without any labeled data.
#

######################################################################
# Summary
# -------
#
# In this tutorial, we demonstrated:
#
# 1. Training TSMNet on source session
# 2. Observing performance drop on target session
# 3. Adapting using SPDBatchNormMeanVar (Test-Time Batch Normalization)
# 4. Comparing with SKADA domain adaptation methods:
#
#    - **CORAL**: Correlation Alignment
#    - **Subspace Alignment**: Linear subspace mapping
#    - **Entropic Optimal Transport**: Sample-to-sample mapping
#
# **Key insights:**
#
# - SPDBatchNormMeanVar provides a native Riemannian approach that operates
#   directly on SPD matrices
# - SKADA methods operate on Euclidean tangent space features and can
#   complement or outperform SPDBatchNormMeanVar depending on the domain shift
# - Combining multiple approaches allows practitioners to select the best
#   method for their specific use case
#
# Cross-session non-stationarity is a key challenge in BCI. Domain
# adaptation methods compensate for these shifts by aligning feature
# distributions between source and target domains.
#
# This **unsupervised domain adaptation** is particularly valuable in BCI
# applications where:
#
# - Calibration time should be minimized
# - Users return for multiple sessions
# - Signal properties drift over time
#
