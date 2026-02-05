"""
.. _matt-attention:

Manifold Attention with MAtt
============================

This tutorial demonstrates how to use MAtt (Manifold Attention Network)
for EEG classification. MAtt applies attention mechanisms on the SPD
manifold to weight temporal segments by their discriminative importance.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# MAtt :cite:p:`pan2022matt` introduces **attention mechanisms on the SPD
# manifold**:
#
# 1. **Patch-based Processing**: Segments the signal into temporal patches
# 2. **Covariance per Patch**: Computes SPD matrices for each segment
# 3. **Manifold Attention**: Weights patches using Log-Euclidean distances
# 4. **Aggregation**: Combines weighted SPD matrices for classification
#
# This allows the model to focus on the most discriminative time periods
# within each trial, improving classification and interpretability.
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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import EpochScoring, GradientNormClipping
from skorch.dataset import ValidSplit

from spd_learn.models import MAtt


warnings.filterwarnings("ignore")

######################################################################
# Loading the Dataset
# -------------------
#

dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

print(f"Dataset: {dataset.code}")
print("Paradigm: 4-class motor imagery")

######################################################################
# Creating the MAtt Model
# -----------------------
#
# MAtt architecture:
#
# 1. **Spatial Conv**: Learns spatial filters
# 2. **Temporal Conv**: Extracts temporal features
# 3. **Patch Embedding**: Segments into n_patches temporal windows
# 4. **Covariance + TraceNorm**: SPD matrix per patch
# 5. **AttentionManifold**: Queries, keys, values on SPD manifold
# 6. **ReEig + LogEig**: Project to tangent space
# 7. **Linear**: Classification
#
# The attention mechanism computes:
#
# .. math::
#
#    \text{attention}(Q, K) = \text{softmax}\left(\frac{1}{1 + \log(1 + d_{LE}(Q, K))}\right)
#
# where :math:`d_{LE}` is the Log-Euclidean distance.
#

n_chans = 22
n_outputs = 4

model = MAtt(
    n_chans=n_chans,
    n_outputs=n_outputs,
    n_patches=6,  # Number of temporal segments
    temporal_out_channels=32,  # Temporal feature dimension
    temporal_kernel_size=25,  # ~100ms at 250Hz
    temporal_padding=12,  # Keep time dimension
    attention_in_features=32,  # Input to attention (must match temporal_out_channels)
    attention_out_features=24,  # Output from attention
)

print("MAtt Architecture:")
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

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training history
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

plt.tight_layout()
plt.show()

######################################################################
# Understanding Manifold Attention
# --------------------------------
#
# The attention mechanism in MAtt operates differently from standard attention:
#
# **Standard Attention** (Euclidean):
#
# .. math::
#
#    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
#
# **Manifold Attention** (Log-Euclidean):
#
# .. math::
#
#    \text{energy}_{ij} = d_{LE}(Q_i, K_j) = \|\log(Q_i) - \log(K_j)\|_F
#
# .. math::
#
#    \text{weights}_{ij} = \frac{1}{1 + \log(1 + \text{energy}_{ij})}
#
# .. math::
#
#    \text{output}_i = \sum_j \text{softmax}(\text{weights})_{ij} \odot V_j
#
# This respects the Riemannian geometry of SPD matrices, computing
# meaningful distances on the manifold rather than in Euclidean space.
#

######################################################################
# Summary
# -------
#
# In this tutorial, we demonstrated:
#
# 1. Creating a MAtt model with patch-based temporal segmentation
# 2. Training for motor imagery classification
# 3. Understanding the manifold attention mechanism
#
# MAtt is particularly useful when:
#
# - Different time segments have varying discriminative power
# - You want interpretable attention weights
# - The data has complex temporal dynamics
#
