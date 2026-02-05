"""
.. _phase-spdnet:

Phase-Space Embedding with PhaseSPDNet
======================================

This tutorial demonstrates how to use PhaseSPDNet for EEG classification.
PhaseSPDNet applies phase-space embedding (time-delay coordinates) to
capture nonlinear dynamical structure before SPDNet processing.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# PhaseSPDNet :cite:p:`carrara2024eegspd` leverages **phase-space embedding**
# from dynamical
# systems theory. The key idea is that a single time series can be
# "unfolded" into a higher-dimensional space that reveals the underlying
# dynamics of the system.
#
# **Takens' Embedding Theorem**: For a dynamical system, a time-delayed
# embedding can reconstruct the topology of the original state space:
#
# .. math::
#
#    \mathbf{x}(t) \rightarrow [\mathbf{x}(t), \mathbf{x}(t-\tau), \mathbf{x}(t-2\tau), \ldots]
#
# This is particularly useful for EEG, where signals reflect complex
# brain dynamics that may not be fully captured by linear methods.
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
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import EpochScoring, GradientNormClipping
from skorch.dataset import ValidSplit

from spd_learn.models import PhaseSPDNet


warnings.filterwarnings("ignore")

######################################################################
# Loading the Dataset
# -------------------
#

dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

print(f"Dataset: {dataset.code}")
print("Sampling rate: 250 Hz")

######################################################################
# Understanding Phase-Space Embedding
# -----------------------------------
#
# For a signal with ``n_chans`` channels, phase-space embedding with
# ``order=m`` and ``lag=τ`` creates:
#
# .. math::
#
#    X_{embedded}(t) = [X(t), X(t-\tau), X(t-2\tau), \ldots, X(t-(m-1)\tau)]
#
# This increases the channel dimension by a factor of ``m``:
# ``n_chans → n_chans * order``
#
# **Choosing parameters**:
#
# - ``order``: Embedding dimension (typically 2-5)
# - ``lag``: Time delay in samples (often chosen via autocorrelation)
#

# Visualize embedding concept
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Original 1D signal
t = np.linspace(0, 4 * np.pi, 200)
x = np.sin(t) + 0.5 * np.sin(2 * t)

ax1 = axes[0]
ax1.plot(t, x, "b-", linewidth=2)
ax1.set_xlabel("Time", fontsize=12)
ax1.set_ylabel("Amplitude", fontsize=12)
ax1.set_title("Original Signal", fontsize=14)
ax1.grid(True, alpha=0.3)

# Phase-space embedding (2D)
lag = 15  # samples
x_delayed = x[:-lag]
x_original = x[lag:]

ax2 = axes[1]
ax2.plot(x_original, x_delayed, "b-", linewidth=1, alpha=0.7)
ax2.scatter(x_original[::10], x_delayed[::10], c=t[lag::10], cmap="viridis", s=30)
ax2.set_xlabel("x(t)", fontsize=12)
ax2.set_ylabel("x(t - τ)", fontsize=12)
ax2.set_title("Phase-Space Embedding (2D)", fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_aspect("equal")

plt.tight_layout()
plt.show()

print("Phase-space embedding reveals the underlying attractor structure!")

######################################################################
# Creating the PhaseSPDNet Model
# ------------------------------
#
# PhaseSPDNet architecture:
#
# 1. **PhaseDelay**: Applies time-delay embedding
# 2. **SPDNet**: Processes the embedded signals
#
# The embedding expands channels: 22 channels x order 3 = 66 channels
#

n_chans = 22
n_outputs = 4

# Phase-space parameters
order = 2  # Embedding dimension (lower for stability)
lag = 10  # Time delay (~40ms at 250Hz)

model = PhaseSPDNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    order=order,
    lag=lag,
    subspacedim=22,  # BiMap output dimension (half of embedded channels)
    threshold=1e-4,
)

print("PhaseSPDNet Configuration:")
print(f"  Original channels: {n_chans}")
print(f"  Embedding order: {order}")
print(f"  Time lag: {lag} samples ({lag/250*1000:.1f} ms)")
print(f"  Embedded channels: {n_chans * order}")
print("  Subspace dimension: 22")
print("\nModel Architecture:")
print(model)

######################################################################
# Training the Model
# ------------------
#

subject_id = 1
batch_size = 32
max_epochs = 100
learning_rate = 1e-4  # Low learning rate for stable SPD learning

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

# Load data
X, labels, meta = paradigm.get_data(
    dataset=dataset, subjects=[subject_id], cache_config=cache_config
)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Split by session
train_idx = meta.query("session == '0train'").index.to_numpy()
test_idx = meta.query("session == '1test'").index.to_numpy()

print(f"\nData shape: {X.shape}")
print(
    f"After embedding: ({X.shape[0]}, {n_chans * order}, {X.shape[2] - (order-1)*lag})"
)
print(f"Training samples: {len(train_idx)}")
print(f"Test samples: {len(test_idx)}")

# Create classifier
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

# Train
clf.fit(X[train_idx], y[train_idx])

# Evaluate
y_pred_train = clf.predict(X[train_idx])
y_pred_test = clf.predict(X[test_idx])

train_acc = accuracy_score(y[train_idx], y_pred_train)
test_acc = accuracy_score(y[test_idx], y_pred_test)

print(f"\n{'='*50}")
print(f"Results for Subject {subject_id}")
print(f"{'='*50}")
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy:  {test_acc*100:.2f}%")

######################################################################
# Visualizing Results
# -------------------
#

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Training curves
history = clf.history
epochs = range(1, len(history) + 1)

ax1 = axes[0]
ax1.plot(epochs, history[:, "train_loss"], "b-", label="Train Loss", linewidth=2)
ax1.plot(epochs, history[:, "valid_loss"], "r--", label="Valid Loss", linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training and Validation Loss", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(epochs, history[:, "train_acc"], "b-", label="Train Acc", linewidth=2)
ax2.plot(epochs, history[:, "valid_acc"], "r--", label="Valid Acc", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.set_title("Training and Validation Accuracy", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# Confusion matrix
ax3 = axes[2]
cm = confusion_matrix(y[test_idx], y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax3, cmap="Blues", values_format="d")
ax3.set_title(f"Confusion Matrix\nAccuracy: {test_acc*100:.1f}%", fontsize=14)

plt.tight_layout()
plt.show()

######################################################################
# Comparing Different Embedding Parameters
# ----------------------------------------
#
# The choice of ``order`` and ``lag`` affects performance.
# Let's compare different configurations.
#

print("\nComparing embedding parameters:")
print("-" * 50)

configs = [
    {"order": 2, "lag": 5, "name": "order=2, lag=5"},
    {"order": 2, "lag": 10, "name": "order=2, lag=10"},
    {"order": 3, "lag": 5, "name": "order=3, lag=5"},
    {"order": 3, "lag": 10, "name": "order=3, lag=10"},
]

for config in configs:
    embedded_chans = n_chans * config["order"]
    reduced_time = X.shape[2] - (config["order"] - 1) * config["lag"]
    cov_size = embedded_chans * (embedded_chans + 1) // 2
    print(
        f"{config['name']:20s}: {embedded_chans} channels, "
        f"{reduced_time} time points, {cov_size} features"
    )

######################################################################
# When to Use Phase-Space Embedding
# ---------------------------------
#
# PhaseSPDNet is particularly effective when:
#
# 1. **Nonlinear dynamics**: The underlying system has complex,
#    nonlinear behavior (e.g., neural oscillations, chaos)
#
# 2. **Limited channels**: Embedding can extract more information
#    from fewer channels
#
# 3. **Temporal structure**: Important features span across time
#    (captured by delay coordinates)
#
# **Considerations**:
#
# - Higher ``order`` increases model capacity but also parameters
# - ``lag`` should be chosen based on the signal's autocorrelation
# - Reduces effective time dimension: ``T_new = T - (order-1) * lag``
#

######################################################################
# Summary
# -------
#
# In this tutorial, we demonstrated:
#
# 1. Phase-space embedding theory and visualization
# 2. Creating PhaseSPDNet with different embedding parameters
# 3. Training and evaluating on motor imagery data
#
# PhaseSPDNet offers a principled way to incorporate dynamical
# systems perspectives into EEG classification, potentially
# capturing nonlinear brain dynamics that linear methods miss.
#
