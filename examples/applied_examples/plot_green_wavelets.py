"""
.. _green-wavelets:

Learnable Wavelets with GREEN for EEG Classification
====================================================

This tutorial demonstrates how to use the GREEN (Gabor Riemann EEGNet)
model for EEG classification. GREEN combines learnable Gabor wavelets
with Riemannian geometry for biomarker exploration.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# GREEN :cite:p:`paillard2024green` is a lightweight architecture that learns
# optimal time-frequency
# representations directly from EEG data. Unlike traditional approaches that
# use fixed filter banks, GREEN employs parametrized Gabor wavelets with
# learnable center frequencies and bandwidths.
#
# Key features of GREEN:
#
# - **Learnable wavelets**: Center frequencies and FWHM are optimized during training
# - **Riemannian geometry**: SPD covariance matrices with BiMap and LogEig layers
# - **Lightweight**: Efficient architecture suitable for clinical applications
#

######################################################################
# Setup and Imports
# -----------------
#
# First, we import the necessary libraries.
#

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from braindecode import EEGClassifier
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import EpochScoring
from skorch.dataset import ValidSplit

from spd_learn.models import Green


warnings.filterwarnings("ignore")

######################################################################
# Loading the Dataset
# -------------------
#
# We use the BCI Competition IV Dataset 2a (BNCI2014_001), which contains
# motor imagery EEG recordings from 9 subjects.
#

dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

print(f"Dataset: {dataset.code}")
print(f"Subjects: {dataset.subject_list}")

######################################################################
# Creating the GREEN Model
# ------------------------
#
# GREEN processes EEG through the following stages:
#
# 1. **Wavelet Convolution**: Learnable Gabor wavelets extract time-frequency features
# 2. **Covariance Pooling**: Compute SPD covariance matrices
# 3. **Shrinkage**: Ledoit-Wolf regularization for stable covariance estimation
# 4. **BiMap Layers**: Optional spatial filtering on the SPD manifold
# 5. **LogEig + BatchReNorm**: Project to tangent space with normalization
# 6. **MLP Head**: Classification with dropout
#
# Key parameters:
#
# - ``n_freqs_init``: Number of wavelet center frequencies (default: 10)
# - ``kernel_width_s``: Wavelet kernel width in seconds
# - ``oct_min/oct_max``: Frequency range in octaves (relative to 1 Hz)
# - ``shrinkage_init``: Initial shrinkage coefficient (sigmoid input)
#

# Model hyperparameters
n_chans = 22
n_outputs = 4
sfreq = 250  # Sampling frequency of BNCI2014_001

# Create GREEN model
model = Green(
    n_outputs=n_outputs,
    n_chans=n_chans,
    sfreq=sfreq,
    n_freqs_init=10,  # Number of learnable wavelets
    kernel_width_s=0.5,  # 500ms wavelet width
    oct_min=0,  # ~1 Hz minimum
    oct_max=5,  # ~32 Hz maximum (2^5)
    shrinkage_init=-3.0,  # Initial shrinkage (sigmoid(-3) ≈ 0.05)
    hidden_dim=(16,),  # Hidden layer in MLP head
    dropout=0.5,
)

print("\nGREEN Model Architecture:")
print(model)

######################################################################
# Setting up the Classifier
# -------------------------
#
# We use Braindecode's EEGClassifier wrapper for scikit-learn compatibility.
#

# Training hyperparameters
batch_size = 32
max_epochs = 50
learning_rate = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=learning_rate,
    optimizer__weight_decay=1e-4,
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
        (
            "bal_acc",
            EpochScoring(
                "balanced_accuracy",
                lower_is_better=False,
                on_train=False,
                name="bal_acc",
            ),
        ),
    ],
    device=device,
    verbose=1,
)

######################################################################
# Training and Evaluation
# -----------------------
#
# We train on a single subject for demonstration.
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
# Visualizing Learned Wavelets
# ----------------------------
#
# One advantage of GREEN is that we can inspect the learned wavelet
# parameters to understand which frequencies are most discriminative.
#

# Extract learned wavelet parameters
wavelet_conv = model.conv_layers[0]
foi_learned = wavelet_conv.foi.detach().cpu().numpy()  # Center frequencies (octaves)
fwhm_learned = wavelet_conv.fwhm.detach().cpu().numpy()  # Bandwidth (octaves)

# Convert from octaves to Hz
foi_hz = 2**foi_learned
bandwidth_hz = 2 ** np.abs(fwhm_learned)

print("\nLearned Wavelet Parameters:")
print("-" * 40)
for i, (f, bw) in enumerate(zip(foi_hz, bandwidth_hz)):
    print(f"Wavelet {i+1}: Center = {f:.1f} Hz, Bandwidth = {bw:.1f} Hz")

# Plot wavelet frequencies
fig, ax = plt.subplots(figsize=(10, 4))

# Sort by center frequency for visualization
sort_idx = np.argsort(foi_hz)
foi_sorted = foi_hz[sort_idx]
bw_sorted = bandwidth_hz[sort_idx]

x_pos = np.arange(len(foi_sorted))
ax.bar(x_pos, foi_sorted, yerr=bw_sorted / 2, capsize=5, color="steelblue", alpha=0.7)
ax.set_xlabel("Wavelet Index (sorted by frequency)", fontsize=12)
ax.set_ylabel("Center Frequency (Hz)", fontsize=12)
ax.set_title("Learned Gabor Wavelet Center Frequencies", fontsize=14)
ax.set_xticks(x_pos)
ax.grid(True, alpha=0.3, axis="y")

# Add frequency band annotations
ax.axhline(y=8, color="green", linestyle="--", alpha=0.5, label="Mu band (8-12 Hz)")
ax.axhline(y=12, color="green", linestyle="--", alpha=0.5)
ax.axhline(
    y=13, color="orange", linestyle="--", alpha=0.5, label="Beta band (13-30 Hz)"
)
ax.axhline(y=30, color="orange", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

plt.tight_layout()
plt.show()

######################################################################
# Training History
# ----------------
#
# Let's visualize the training progress.
#

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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
ax2.plot(epochs, history[:, "bal_acc"], "r--", label="Valid Balanced Acc", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.set_title("Training and Validation Accuracy", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.show()

######################################################################
# Summary
# -------
#
# In this tutorial, we demonstrated how to:
#
# 1. Create a GREEN model with learnable Gabor wavelets
# 2. Train and evaluate on motor imagery EEG data
# 3. Visualize the learned wavelet parameters
#
# GREEN's learnable wavelets allow the model to discover optimal
# time-frequency representations for the classification task, often
# focusing on the mu (8-12 Hz) and beta (13-30 Hz) rhythms known
# to be modulated during motor imagery.
#
