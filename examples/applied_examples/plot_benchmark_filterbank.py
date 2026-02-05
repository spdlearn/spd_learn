"""
.. _filterbank-tensorcspnet:

Filter Bank Motor Imagery with TensorCSPNet
===========================================

This tutorial demonstrates how to use TensorCSPNet for motor imagery
classification with filter bank features. TensorCSPNet is designed to
process multi-frequency EEG data by stacking covariance matrices from
different frequency bands into a tensor structure.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# Motor imagery (MI) is a mental process where a person imagines performing
# a motor action without actually executing it. EEG-based brain-computer
# interfaces (BCIs) can decode these imagined movements to control devices.
#
# Filter bank approaches decompose the EEG signal into multiple frequency
# bands, allowing the model to capture frequency-specific spatial patterns.
# TensorCSPNet :cite:p:`ju2022tensor` leverages this by creating SPD (Symmetric
# Positive
# Definite) covariance matrices for each frequency band and processing
# them through a geometry-aware neural network.
#

######################################################################
# Setup and Imports
# -----------------
#
# First, we import the necessary libraries. We use:
#
# - **MOABB**: For loading standardized EEG datasets
# - **Braindecode**: For the EEGClassifier wrapper
# - **SPD Learn**: For the TensorCSPNet model
# - **scikit-learn**: For evaluation metrics and pipelines
#

import warnings

import matplotlib.pyplot as plt
import moabb
import torch

from braindecode import EEGClassifier
from einops.layers.torch import Rearrange
from moabb.datasets import BNCI2014_001
from moabb.paradigms import FilterBankMotorImagery
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import EpochScoring
from skorch.dataset import ValidSplit
from torch import nn

from spd_learn.models import TensorCSPNet


# Set logging and ignore warnings for cleaner output
moabb.set_log_level("info")
warnings.filterwarnings("ignore")

######################################################################
# Loading the Dataset
# -------------------
#
# We use the BCI Competition IV Dataset 2a (BNCI2014_001)
# :cite:p:`tangermann2012review`, which
# contains EEG recordings from 9 subjects performing 4 different motor
# imagery tasks:
#
# - Left hand movement
# - Right hand movement
# - Both feet movement
# - Tongue movement
#
# The dataset has 22 EEG channels and was recorded at 250 Hz.
#

dataset = BNCI2014_001()
print(f"Dataset: {dataset.code}")
print(f"Number of subjects: {len(dataset.subject_list)}")
print("Number of sessions per subject: 2 (train + test)")

######################################################################
# Defining the Filter Bank
# ------------------------
#
# We define a filter bank covering the mu (8-12 Hz) and beta (12-30 Hz)
# rhythms, which are known to be modulated during motor imagery
# :cite:p:`pfurtscheller1999event`.
# Each filter extracts a specific frequency band from the EEG signal.
#

filters = [
    [4, 8],  # Theta band
    [8, 12],  # Mu/Alpha band
    [12, 16],  # Low beta
    [16, 20],  # Mid beta
    [20, 24],  # High beta
    [24, 28],  # Beta/Gamma transition
    [28, 32],  # Low gamma
    [32, 36],  # Gamma
    [36, 40],  # High gamma
]

print(f"Number of frequency bands: {len(filters)}")
print("Frequency bands (Hz):")
for i, (low, high) in enumerate(filters):
    print(f"  Band {i+1}: {low}-{high} Hz")

######################################################################
# Setting up the Paradigm
# -----------------------
#
# The FilterBankMotorImagery paradigm from MOABB handles:
#
# - Filtering the data into multiple frequency bands
# - Extracting epochs around motor imagery events
# - Organizing data in the format (n_trials, n_channels, n_times, n_filters)
#

paradigm = FilterBankMotorImagery(n_classes=4, filters=filters)
print(f"\nParadigm: {paradigm}")
print("Number of classes: 4 (left hand, right hand, feet, tongue)")

######################################################################
# Creating the TensorCSPNet Model
# -------------------------------
#
# TensorCSPNet :cite:p:`ju2022tensor` is a deep learning architecture designed
# for filter
# bank EEG classification. The architecture consists of:
#
# 1. **Tensor Stacking**: Organizes multi-band covariance matrices
# 2. **BiMap Layers**: Learns spatial filters on the SPD manifold
# 3. **Temporal Convolution**: Captures temporal dynamics
# 4. **Classification Head**: Final prediction layer
#
# .. note::
#    The input to TensorCSPNet has shape (batch, channels, time, frequencies).
#    We use einops to rearrange from MOABB's format (batch, channels, time, freq)
#    to the expected format.
#

# Training hyperparameters
batch_size = 16
max_epochs = 15  # Reduced from 50 for faster documentation build
learning_rate = 1e-3

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# Create the model pipeline
# We wrap TensorCSPNet with a Rearrange layer to handle the input format
model = nn.Sequential(
    Rearrange("b c t f -> b f c t"),  # Rearrange to (batch, freq, channels, time)
    TensorCSPNet(
        n_chans=22,  # Number of EEG channels
        n_outputs=4,  # Number of classes
        n_freqs=len(filters),  # Number of frequency bands
    ),
)

print("\nModel architecture:")
print(model)

######################################################################
# Setting up the Classifier
# -------------------------
#
# We use Braindecode's EEGClassifier, which is built on top of skorch
# and provides a scikit-learn compatible interface for training
# PyTorch models.
#

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
    ],
    device=device,
    verbose=1,
)

######################################################################
# Training and Evaluation Function
# --------------------------------
#
# We define a function to train and evaluate the model on a single subject.
# This function:
#
# 1. Loads the data for the subject
# 2. Splits into training and test sets (using session info)
# 3. Trains the model
# 4. Evaluates on both train and test sets
#


def evaluate_subject(subject: int) -> dict:
    """Train and evaluate TensorCSPNet on a single subject.

    Parameters
    ----------
    subject : int
        Subject ID to evaluate.

    Returns
    -------
    dict
        Dictionary containing accuracy scores and predictions.
    """
    print(f"\n{'='*50}")
    print(f"Evaluating Subject {subject}")
    print(f"{'='*50}")

    # Cache configuration for faster repeated runs
    cache_config = dict(
        save_raw=True,
        save_epochs=True,
        save_array=True,
        use=True,
        overwrite_raw=False,
        overwrite_epochs=False,
        overwrite_array=False,
    )

    # Load data for this subject
    X, labels, meta = paradigm.get_data(
        dataset=dataset, subjects=[subject], cache_config=cache_config
    )

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"Data shape: {X.shape}")
    print(f"Labels: {le.classes_}")

    # Split into train and test using session information
    # Session '0train' is for training, '1test' is for testing
    train_idx = meta.query("session == '0train'").index.to_numpy()
    test_idx = meta.query("session == '1test'").index.to_numpy()

    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")

    # Train the model
    clf.fit(X[train_idx], y[train_idx])

    # Get predictions
    y_pred_train = clf.predict(X[train_idx])
    y_pred_test = clf.predict(X[test_idx])

    # Calculate accuracies
    train_acc = accuracy_score(y[train_idx], y_pred_train)
    test_acc = accuracy_score(y[test_idx], y_pred_test)

    print(f"\nResults for Subject {subject}:")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")

    return {
        "subject": subject,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "y_true_test": y[test_idx],
        "y_pred_test": y_pred_test,
        "label_encoder": le,
        "history": clf.history,
    }


######################################################################
# Running the Evaluation
# ----------------------
#
# For demonstration purposes, we evaluate on a single subject.
# In practice, you would loop over all subjects for a complete benchmark.
#
# .. note::
#    Training deep learning models on EEG data can take several minutes
#    per subject, depending on your hardware.
#

# Evaluate on subject 1 (you can change this or loop over all subjects)
subject_id = 1
results = evaluate_subject(subject_id)

######################################################################
# Visualizing Training History
# ----------------------------
#
# Let's plot the training and validation loss curves to understand
# how the model learned over epochs.
#

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Extract history
history = results["history"]
epochs = range(1, len(history) + 1)

# Plot loss
ax1 = axes[0]
ax1.plot(epochs, history[:, "train_loss"], "b-", label="Train Loss", linewidth=2)
ax1.plot(epochs, history[:, "valid_loss"], "r--", label="Valid Loss", linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training and Validation Loss", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot accuracy
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
# Confusion Matrix
# ----------------
#
# The confusion matrix shows how well the model distinguishes between
# different motor imagery classes.
#

fig, ax = plt.subplots(figsize=(8, 6))

# Get class names
class_names = results["label_encoder"].classes_

# Compute confusion matrix
cm = confusion_matrix(results["y_true_test"], results["y_pred_test"])

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title(
    f"Confusion Matrix - Subject {subject_id}\n"
    f"Test Accuracy: {results['test_acc']*100:.2f}%",
    fontsize=14,
)
plt.tight_layout()
plt.show()

######################################################################
# Summary
# -------
#
# In this tutorial, we demonstrated how to:
#
# 1. Load and prepare filter bank motor imagery data using MOABB
# 2. Create a TensorCSPNet model for multi-frequency EEG classification
# 3. Train and evaluate the model using Braindecode's EEGClassifier
# 4. Visualize training history and confusion matrices
#
# TensorCSPNet leverages the geometry of SPD matrices to learn
# discriminative spatial filters across multiple frequency bands,
# making it well-suited for motor imagery classification.
#
