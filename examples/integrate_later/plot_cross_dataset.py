"""
.. _cross-dataset-example:

Cross-Dataset EEG Classification with SPDNet
=============================================

This tutorial demonstrates **cross-dataset generalization** using SPDNet
for motor imagery EEG classification. We use Leave-One-Dataset-Out (LODO)
cross-validation to evaluate how well the model transfers between
different EEG recording setups.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# Cross-dataset generalization is one of the most challenging problems
# in EEG-based brain-computer interfaces. Different datasets vary in:
#
# - **Recording equipment**: Different amplifiers, electrode types
# - **Electrode configurations**: Varying channel counts and placements
# - **Subject populations**: Age, experience, health status
# - **Experimental protocols**: Task instructions, timing, feedback
#
# SPDNet's geometric approach :cite:p:`huang2017riemannian` operating on the
# SPD manifold can help learn representations that are more robust to these
# variations, since covariance matrices capture second-order statistics that
# are somewhat invariant to amplitude scaling differences.
#
# .. note::
#
#    This is a **challenging benchmark**. Even small positive transfer
#    indicates that the model has learned generalizable features rather
#    than dataset-specific artifacts.
#

######################################################################
# Setup and Imports
# -----------------
#

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Pick,
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.util import set_random_seeds
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.utils import check_random_state
from skorch.callbacks import EpochScoring, GradientNormClipping
from skorch.dataset import ValidSplit
from skorch.helper import SliceDataset
from torch.utils.data import ConcatDataset

from spd_learn.models import SPDNet


warnings.filterwarnings("ignore")

######################################################################
# Configuration
# -------------
#
# We configure the experiment with fixed random seeds for reproducibility.
#

seed = 42
set_random_seeds(seed, cuda=False)
random_state = check_random_state(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

######################################################################
# Loading the Datasets
# --------------------
#
# We load two motor imagery datasets from MOABB:
#
# - **BNCI2014_001** (BCI Competition IV 2a) :cite:p:`tangermann2012review`:
#   9 subjects, 22 channels, 4-class motor imagery (left hand, right hand,
#   feet, tongue)
# - **Zhou2016** :cite:p:`zhou2016fully`: 4 subjects, 14 channels, 3-class
#   motor imagery
#
# For cross-dataset transfer, we need to:
#
# 1. Select **common channels** available in both datasets
# 2. Use **common classes** (left hand vs right hand)
# 3. Apply **consistent preprocessing** (same frequency band, sampling rate)
#

cache_config = dict(
    use=True,
    save_raw=True,
    save_epochs=False,
    save_array=False,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
    verbose=False,
)

# Load BNCI2014_001 dataset
# Using subset of subjects for faster demonstration
print("Loading BNCI2014_001 dataset...")
bnci2014_01 = MOABBDataset(
    "BNCI2014_001",
    subject_ids=[1, 2],  # 2 subjects for demonstration
    dataset_load_kwargs={"cache_config": cache_config},
)

# Load Zhou2016 dataset
print("Loading Zhou2016 dataset...")
# Note: Zhou2016 has different subject ID numbering, loading all available subjects
zhou2016 = MOABBDataset(
    "Zhou2016",
    subject_ids=None,  # Load all available subjects
    dataset_load_kwargs={"cache_config": cache_config},
)

######################################################################
# Preprocessing Pipeline
# ----------------------
#
# We apply a standardized preprocessing pipeline to both datasets:
#
# 1. **Channel selection**: Pick channels common to both datasets (C3, Cz, C4)
#    - These are central motor cortex electrodes, most relevant for
#    motor imagery classification
#
# 2. **Resampling**: Downsample to 125 Hz
#    - Reduces computational cost while preserving relevant frequencies
#
# 3. **Amplitude scaling**: Convert to microvolts
#    - Standardizes units across datasets
#
# 4. **Bandpass filtering**: 4-38 Hz
#    - Includes mu (8-12 Hz) and beta (13-30 Hz) bands relevant for
#    motor imagery
#
# 5. **Exponential moving standardization**
#    - Adaptive normalization that handles non-stationarity
#


def preprocess_dataset(dataset, target_sfreq=125):
    """Apply standardized preprocessing pipeline.

    Parameters
    ----------
    dataset : BaseConcatDataset
        Raw braindecode dataset.
    target_sfreq : float
        Target sampling frequency in Hz.

    Returns
    -------
    BaseConcatDataset
        Preprocessed dataset.

    Notes
    -----
    The preprocessing steps are chosen to maximize compatibility
    across different recording setups while preserving the neural
    information relevant for motor imagery classification.
    """
    preprocessors = [
        # Select motor cortex channels common to both datasets
        Pick(picks=["C3", "Cz", "C4"]),
        # Resample to common frequency
        Preprocessor("resample", sfreq=target_sfreq),
        # Convert to microvolts (standardize amplitude units)
        Preprocessor(lambda data: np.multiply(data, 1e6)),
        # Bandpass filter for motor imagery bands (mu + beta)
        Preprocessor("filter", l_freq=4.0, h_freq=38.0, verbose=False),
        # Exponential moving standardization for non-stationarity
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000,
        ),
    ]
    return preprocess(dataset, preprocessors, n_jobs=1)


print("\nPreprocessing datasets...")
print("  - Selecting channels: C3, Cz, C4")
print("  - Resampling to 125 Hz")
print("  - Bandpass filtering: 4-38 Hz")

bnci2014_01 = preprocess_dataset(bnci2014_01)
zhou2016 = preprocess_dataset(zhou2016)

print("Preprocessing complete.")

######################################################################
# Creating Windows
# ----------------
#
# We extract 1-second windows from the continuous EEG data.
# Using fixed-size windows ensures consistent input dimensions
# for the neural network.
#

# Label mapping: only use left/right hand (common across datasets)
mapping = {"left_hand": 0, "right_hand": 1}

window_size_samples = 125  # 1 second at 125 Hz
window_stride_samples = 125  # Non-overlapping windows

print(
    f"\nCreating windows (size={window_size_samples} samples, stride={window_stride_samples})..."
)

windows_bnci2014_01 = create_windows_from_events(
    bnci2014_01,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    preload=True,
    mapping=mapping,
    n_jobs=1,
)

windows_zhou2016 = create_windows_from_events(
    zhou2016,
    preload=True,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    mapping=mapping,
    n_jobs=1,
)

# Prepare dataset list for cross-validation
dataset_list = [windows_bnci2014_01, windows_zhou2016]
dataset_names = ["BNCI2014_001", "Zhou2016"]

print(f"  BNCI2014_001: {len(windows_bnci2014_01)} windows")
print(f"  Zhou2016: {len(windows_zhou2016)} windows")

######################################################################
# SPDNet Model Architecture
# -------------------------
#
# SPDNet processes data through the following stages:
#
# 1. **Input**: Raw EEG signal :math:`X \in \mathbb{R}^{C \times T}`
#
# 2. **Covariance computation**: Inside the network, covariance is computed
#
#    .. math::
#
#       \Sigma = \frac{1}{T-1} X X^T \in \text{SPD}(C)
#
# 3. **BiMap layers**: Dimensionality reduction while preserving SPD structure
#
#    .. math::
#
#       Y = W^T \Sigma W, \quad W \in \mathbb{R}^{C_{in} \times C_{out}}
#
# 4. **ReEig layers**: Non-linear activation via eigenvalue rectification
#
#    .. math::
#
#       Y = U \max(\Lambda, \epsilon) U^T
#
# 5. **LogEig layer**: Project to tangent space for linear classification
#
#    .. math::
#
#       y = \text{vec}(\log(Y))
#
# For cross-dataset transfer, the geometric operations on the SPD manifold
# help learn features that are more invariant to dataset-specific variations.
#

n_chans = 3  # C3, Cz, C4
n_classes = 2  # Left hand vs right hand

print("\nModel configuration:")
print(f"  Input channels: {n_chans}")
print(f"  Output classes: {n_classes}")

######################################################################
# Leave-One-Dataset-Out Cross-Validation
# --------------------------------------
#
# We perform Leave-One-Dataset-Out (LODO) cross-validation:
#
# - **Fold 1**: Train on Zhou2016, test on BNCI2014_001
# - **Fold 2**: Train on BNCI2014_001, test on Zhou2016
#
# This evaluates whether the model can learn features that generalize
# across recording setups, rather than overfitting to dataset-specific
# characteristics.
#

# Training hyperparameters
batch_size = 64
max_epochs = 50  # More epochs for better convergence
learning_rate = 1e-3

results = []
fold_histories = []

for fold_idx in range(len(dataset_list)):
    # Split datasets
    test_set = dataset_list[fold_idx]
    train_sets = [ds for j, ds in enumerate(dataset_list) if j != fold_idx]

    # Combine training sets
    train_set = ConcatDataset(train_sets)
    y_train = np.array(list(SliceDataset(train_set, 1)))
    y_test = np.array(list(SliceDataset(test_set, 1)))

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}: Test on {dataset_names[fold_idx]}")
    print(f"{'='*60}")
    print(
        f"Train: {len(train_set)} samples from {[n for j, n in enumerate(dataset_names) if j != fold_idx]}"
    )
    print(f"Test:  {len(test_set)} samples from {dataset_names[fold_idx]}")

    # Create fresh SPDNet model for each fold
    model = SPDNet(
        n_chans=n_chans,
        n_outputs=n_classes,
    )

    # Create classifier with braindecode
    clf = EEGClassifier(
        module=model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=learning_rate,
        optimizer__weight_decay=1e-4,
        train_split=ValidSplit(0.2, stratified=True, random_state=random_state),
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
        classes=list(range(n_classes)),
        verbose=1,
    )

    # Train the model
    clf.fit(train_set, y=y_train)
    fold_histories.append(clf.history)

    # Evaluate on test set
    y_pred = clf.predict(test_set)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    test_confusion = confusion_matrix(y_test, y_pred)

    results.append(
        {
            "fold": fold_idx + 1,
            "test_dataset": dataset_names[fold_idx],
            "train_datasets": [n for j, n in enumerate(dataset_names) if j != fold_idx],
            "accuracy": test_accuracy,
            "balanced_accuracy": test_balanced_accuracy,
            "confusion_matrix": test_confusion,
            "n_train": len(train_set),
            "n_test": len(test_set),
        }
    )

    print(f"\nFold {fold_idx + 1} Results:")
    print(f"  Accuracy:          {test_accuracy:.4f}")
    print(f"  Balanced Accuracy: {test_balanced_accuracy:.4f}")

######################################################################
# Results Summary
# ---------------
#
# We summarize the cross-dataset classification results.
#

print("\n" + "=" * 60)
print("Cross-Dataset Classification Results")
print("=" * 60)

for r in results:
    print(
        f"\nFold {r['fold']}: Train on {r['train_datasets']} → Test on {r['test_dataset']}"
    )
    print(f"  Accuracy:          {r['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {r['balanced_accuracy']:.4f}")

mean_acc = np.mean([r["accuracy"] for r in results])
mean_bal_acc = np.mean([r["balanced_accuracy"] for r in results])
print("\nOverall Performance:")
print(f"  Mean Accuracy:          {mean_acc:.4f}")
print(f"  Mean Balanced Accuracy: {mean_bal_acc:.4f}")
print("  Chance Level:           0.5000")

######################################################################
# Visualizing Results
# -------------------
#

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training curves for each fold
ax1 = axes[0, 0]
for fold_idx, history in enumerate(fold_histories):
    epochs = range(1, len(history) + 1)
    ax1.plot(
        epochs,
        history[:, "train_loss"],
        linestyle="-",
        label=f"Fold {fold_idx + 1} Train",
        alpha=0.8,
    )
    ax1.plot(
        epochs,
        history[:, "valid_loss"],
        linestyle="--",
        label=f"Fold {fold_idx + 1} Valid",
        alpha=0.8,
    )
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training and Validation Loss", fontsize=14)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy comparison
ax2 = axes[0, 1]
x_pos = np.arange(len(results))
bar_width = 0.35

bars1 = ax2.bar(
    x_pos - bar_width / 2,
    [r["accuracy"] for r in results],
    bar_width,
    label="Accuracy",
    color="#3498db",
    edgecolor="black",
)
bars2 = ax2.bar(
    x_pos + bar_width / 2,
    [r["balanced_accuracy"] for r in results],
    bar_width,
    label="Balanced Accuracy",
    color="#2ecc71",
    edgecolor="black",
)

ax2.axhline(y=0.5, color="red", linestyle="--", label="Chance", alpha=0.7)
ax2.set_xlabel("Test Dataset", fontsize=12)
ax2.set_ylabel("Score", fontsize=12)
ax2.set_title("Cross-Dataset Classification Performance", fontsize=14)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([r["test_dataset"] for r in results])
ax2.set_ylim([0, 1])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars2:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Plot 3 & 4: Confusion matrices for each fold
for fold_idx, r in enumerate(results):
    ax = axes[1, fold_idx]
    cm = r["confusion_matrix"]
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        f"Fold {fold_idx + 1}: Test on {r['test_dataset']}\nAcc: {r['accuracy']:.2%}",
        fontsize=12,
    )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Left Hand", "Right Hand"])
    ax.set_yticklabels(["Left Hand", "Right Hand"])

plt.tight_layout()
plt.suptitle("Cross-Dataset Transfer Learning Results", fontsize=16, y=1.02)
plt.show()

######################################################################
# Discussion
# ----------
#
# Cross-dataset generalization is inherently challenging due to the
# significant differences between recording setups. Key observations:
#
# **Why SPDNet helps with cross-dataset transfer:**
#
# - **Geometric invariance**: Operating on the SPD manifold provides
#   some invariance to amplitude scaling differences between setups
# - **Second-order statistics**: Covariance matrices capture signal
#   relationships that are more stable across conditions than raw amplitudes
# - **Riemannian operations**: The geometric operations respect the
#   intrinsic structure of covariance matrices
#
# **Factors affecting cross-dataset performance:**
#
# 1. **Channel overlap**: We used only 3 common channels (C3, Cz, C4),
#    which limits the spatial information available
# 2. **Subject variability**: Even within datasets, subjects vary significantly
# 3. **Task differences**: Subtle protocol variations affect the recorded signals
#
# **Recommendations for improving cross-dataset transfer:**
#
# - Use **more subjects** for training to learn more generalizable features
# - Apply **domain adaptation techniques** (see TSMNet example)
# - Use **channel interpolation** to leverage more electrodes
# - Consider **data augmentation** on the SPD manifold
#

######################################################################
# Summary
# -------
#
# In this tutorial, we demonstrated:
#
# 1. Loading and preprocessing multiple EEG datasets for cross-dataset transfer
# 2. Selecting common channels and classes across datasets
# 3. Training SPDNet with Leave-One-Dataset-Out cross-validation
# 4. Evaluating cross-dataset generalization performance
#
# Key takeaways:
#
# - Cross-dataset transfer is challenging but important for practical BCIs
# - SPDNet's geometric approach provides some robustness to dataset variations
# - Standardized preprocessing is crucial for fair cross-dataset comparison
# - Performance above chance indicates learned generalizable features
#
