"""
.. _eegspdnet-example:

EEG Classification with EEGSPDNet
=================================

This tutorial demonstrates how to use EEGSPDNet for motor imagery
EEG classification. EEGSPDNet combines channel-specific convolution
with SPD matrix learning for robust EEG decoding.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# EEGSPDNet :cite:p:`wilson2025deep` is a deep Riemannian network designed
# specifically for
# EEG decoding. It extends the SPDNet architecture with:
#
# - **Channel-specific convolution**: Learns temporal filters independently
#   for each EEG channel using grouped convolutions
# - **Covariance pooling**: Computes SPD covariance matrices from the
#   filtered signals
# - **Scalable BiMap layers**: Progressively reduces dimensionality on
#   the SPD manifold
# - **SPD Dropout**: Structured dropout that maintains positive definiteness
#

######################################################################
# Setup and Imports
# -----------------
#

import warnings

import matplotlib.pyplot as plt
import torch

from braindecode import EEGClassifier
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import EpochScoring, GradientNormClipping
from skorch.dataset import ValidSplit

from spd_learn.models import EEGSPDNet


warnings.filterwarnings("ignore")

######################################################################
# Loading the Dataset
# -------------------
#
# We use the BCI Competition IV Dataset 2a for motor imagery classification.
#

dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

print(f"Dataset: {dataset.code}")
print(f"Number of subjects: {len(dataset.subject_list)}")

######################################################################
# Understanding EEGSPDNet Architecture
# ------------------------------------
#
# EEGSPDNet processes EEG in the following stages:
#
# 1. **Channel-specific Conv1d**: Each channel gets its own set of filters
#
#    .. math::
#
#       X_{conv} = \text{GroupedConv1d}(X), \quad X \in \mathbb{R}^{C \times T}
#
#    Output shape: ``(n_chans * n_filters, time - filter_length + 1)``
#
# 2. **Covariance Pooling**: Compute SPD covariance matrix
#
#    .. math::
#
#       \Sigma = \frac{1}{T-1} X_{conv} X_{conv}^T
#
# 3. **BiMap + ReEig blocks**: Learn spatial filters while preserving SPD
#
#    .. math::
#
#       Y = W^T \Sigma W, \quad Y = U \max(\Lambda, \epsilon) U^T
#
# 4. **LogEig**: Project to tangent space for classification
#

######################################################################
# Creating the EEGSPDNet Model
# ----------------------------
#
# Key parameters:
#
# - ``n_filters``: Number of temporal filters per channel
# - ``bimap_sizes``: Tuple (k, n_layers) defining scaling factor and depth
# - ``filter_time_length``: Length of temporal convolution kernel
# - ``spd_drop_prob``: Dropout probability for SPD dropout layers
#

n_chans = 22
n_outputs = 4

# Create EEGSPDNet model
model = EEGSPDNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    n_filters=4,  # 4 filters per channel → 88 total
    bimap_sizes=(2, 2),  # Scale by 2x, 2 BiMap layers: 88→44→22
    filter_time_length=25,  # 100ms filter at 250Hz
    spd_drop_prob=0.0,  # No SPD dropout (can cause instability)
    spd_drop_scaling=True,  # Scale remaining channels
    final_layer_drop_prob=0.5,  # 50% dropout before classifier
)

print("EEGSPDNet Architecture:")
print(model)

# Show BiMap layer dimensions
print("\nBiMap Layer Dimensions:")
print("Input: 88 x 88 (22 channels x 4 filters)")
print("→ BiMap0: 88 → 44")
print("→ BiMap1: 44 → 22")
print("→ LogEig: 22 x 22 → 253 (upper triangular)")

######################################################################
# Setting up the Classifier
# -------------------------
#

batch_size = 32
max_epochs = 100
learning_rate = 1e-4  # Low learning rate for stable SPD learning

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# Note: SPD networks benefit from gradient clipping to prevent
# divergence during training on the Riemannian manifold.
clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=learning_rate,
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

######################################################################
# Training and Evaluation
# -----------------------
#

subject_id = 1

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

# Load data
X, labels, meta = paradigm.get_data(
    dataset=dataset, subjects=[subject_id], cache_config=cache_config
)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

print(f"\nData shape: {X.shape}")
print(f"Classes: {le.classes_}")

# Split by session
train_idx = meta.query("session == '0train'").index.to_numpy()
test_idx = meta.query("session == '1test'").index.to_numpy()

print(f"Training samples: {len(train_idx)}")
print(f"Test samples: {len(test_idx)}")

# Train
clf.fit(X[train_idx], y[train_idx])

# Evaluate
y_pred_train = clf.predict(X[train_idx])
y_pred_test = clf.predict(X[test_idx])

train_acc = accuracy_score(y[train_idx], y_pred_train)
test_acc = accuracy_score(y[test_idx], y_pred_test)
test_bal_acc = balanced_accuracy_score(y[test_idx], y_pred_test)

print(f"\n{'='*50}")
print(f"Results for Subject {subject_id}")
print(f"{'='*50}")
print(f"Train Accuracy:    {train_acc*100:.2f}%")
print(f"Test Accuracy:     {test_acc*100:.2f}%")
print(f"Test Balanced Acc: {test_bal_acc*100:.2f}%")

######################################################################
# Visualizing Results
# -------------------
#

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

history = clf.history
epochs = range(1, len(history) + 1)

# Loss
ax1 = axes[0]
ax1.plot(epochs, history[:, "train_loss"], "b-", label="Train Loss", linewidth=2)
ax1.plot(epochs, history[:, "valid_loss"], "r--", label="Valid Loss", linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training and Validation Loss", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Accuracy
ax2 = axes[1]
ax2.plot(epochs, history[:, "train_acc"], "b-", label="Train Acc", linewidth=2)
ax2.plot(epochs, history[:, "valid_acc"], "r--", label="Valid Acc", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.set_title("Training and Validation Accuracy", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# Confusion Matrix
ax3 = axes[2]
cm = confusion_matrix(y[test_idx], y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax3, cmap="Blues", values_format="d")
ax3.set_title(f"Test Confusion Matrix\nAccuracy: {test_acc*100:.1f}%", fontsize=14)

plt.tight_layout()
plt.show()

######################################################################
# Comparing with Different Configurations
# ---------------------------------------
#
# Let's compare different EEGSPDNet configurations to understand
# the impact of hyperparameters.
#

configs = {
    "Small (k=2, depth=1)": {"bimap_sizes": (2, 1), "n_filters": 4},
    "Medium (k=2, depth=2)": {"bimap_sizes": (2, 2), "n_filters": 4},
    "Large (k=2, depth=3)": {"bimap_sizes": (2, 3), "n_filters": 6},
}

print("\nModel Size Comparison:")
print("-" * 60)
for name, config in configs.items():
    temp_model = EEGSPDNet(n_chans=n_chans, n_outputs=n_outputs, **config)
    n_params = sum(p.numel() for p in temp_model.parameters())
    print(f"{name}: {n_params:,} parameters")

######################################################################
# Summary
# -------
#
# In this tutorial, we demonstrated how to:
#
# 1. Create and configure an EEGSPDNet model
# 2. Understand the architecture's channel-specific processing
# 3. Train and evaluate on motor imagery data
# 4. Compare different model configurations
#
# EEGSPDNet's channel-specific convolution allows it to learn
# independent temporal features for each electrode, which is
# particularly useful for spatially distributed brain signals.
#
