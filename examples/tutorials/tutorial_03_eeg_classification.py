"""
.. _tutorial-eeg-classification:

End-to-End EEG Classification Tutorial
======================================

This comprehensive tutorial demonstrates how to build an end-to-end EEG
classification pipeline for motor imagery using SPD Learn. We cover everything
from data loading to model selection, training, and evaluation.

.. contents:: This tutorial covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction to EEG and Motor Imagery Classification
# ----------------------------------------------------
#
# Electroencephalography (EEG) measures electrical activity in the brain
# through electrodes placed on the scalp. Motor imagery (MI) is a mental
# process where a person imagines performing a motor action without
# actually executing it.
#
# **Why Motor Imagery Classification?**
#
# - **Brain-Computer Interfaces (BCIs)**: Allows paralyzed patients to
#   control devices through thought alone
# - **Rehabilitation**: Helps stroke patients recover motor function
# - **Gaming and Entertainment**: Enables hands-free control
#
# **The SPD Approach**
#
# Traditional approaches use spatial filters like Common Spatial Patterns
# (CSP) to extract discriminative features. SPD Learn takes this further
# by operating directly on the manifold of Symmetric Positive Definite
# (SPD) matrices (covariance matrices), preserving their geometric structure.
#
# This tutorial uses the BNCI2014_001 dataset :cite:p:`tangermann2012review`
# (BCI Competition IV 2a), which contains 4-class motor imagery data from
# 9 subjects.
#

######################################################################
# Setup and Imports
# -----------------
#
# First, we import the necessary libraries:
#
# - **MOABB**: For loading standardized EEG datasets with proper preprocessing
# - **Braindecode**: For the EEGClassifier wrapper (scikit-learn compatible)
# - **SPD Learn**: For geometric deep learning models
# - **scikit-learn**: For evaluation metrics and cross-validation
#

import warnings

import matplotlib.pyplot as plt
import numpy as np
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
from skorch.callbacks import EarlyStopping, EpochScoring, GradientNormClipping
from skorch.dataset import ValidSplit

from spd_learn.models import EEGSPDNet, SPDNet, TSMNet


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

######################################################################
# Loading Data from MOABB
# -----------------------
#
# MOABB (Mother of All BCI Benchmarks) provides standardized access to
# many EEG datasets. We use the BNCI2014_001 dataset:
#
# - **9 subjects**: Each recorded on 2 days (sessions)
# - **4 classes**: Left hand, right hand, feet, tongue
# - **22 EEG channels**: Standard 10-20 montage
# - **250 Hz sampling rate**: After resampling
#
# The dataset is split into training (session 1) and testing (session 2),
# simulating real-world cross-session transfer.
#

# Load the dataset
dataset = BNCI2014_001()

# Create paradigm with 4 motor imagery classes
# MOABB handles filtering (8-35 Hz bandpass is applied by default for MI)
paradigm = MotorImagery(n_classes=4)

print("=" * 60)
print("Dataset Information")
print("=" * 60)
print(f"Dataset: {dataset.code}")
print(f"Number of subjects: {len(dataset.subject_list)}")
print("Sessions per subject: 2 (train + test)")
print("Classes: left_hand, right_hand, feet, tongue")

######################################################################
# Preprocessing Recommendations
# -----------------------------
#
# Proper preprocessing is crucial for good BCI performance. MOABB handles
# most preprocessing automatically, but here are key considerations:
#
# **Filtering**
#
# - Motor imagery is characterized by Event-Related Desynchronization (ERD)
#   and Synchronization (ERS) in the mu (8-12 Hz) and beta (13-30 Hz) bands
# - Default: 8-35 Hz bandpass filter (captures both mu and beta)
# - For multi-frequency analysis, use FilterBankMotorImagery
#
# **Epoching**
#
# - Motor imagery effects typically occur 0.5-4 seconds after cue onset
# - Default: 0 to 4 seconds post-cue
# - Baseline correction is applied automatically
#
# **Artifact Handling**
#
# - Eye blinks and muscle artifacts can contaminate signals
# - MOABB applies basic artifact rejection
# - For production: Consider ICA or other artifact removal methods
#
# .. tip::
#    For SPD methods, signal quality is crucial because noise affects
#    the covariance matrix estimation. Ensure clean data before training.
#

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

# Load data for a single subject (we'll do proper evaluation later)
subject_id = 1
X, labels, meta = paradigm.get_data(
    dataset=dataset, subjects=[subject_id], cache_config=cache_config
)

# Encode labels to integers
le = LabelEncoder()
y = le.fit_transform(labels)

print(f"\nData loaded for Subject {subject_id}:")
print(f"  Shape: {X.shape} (trials, channels, timepoints)")
print("  Sampling rate: 250 Hz")
print(f"  Epoch length: {X.shape[2] / 250:.1f} seconds")
print(f"  Classes: {le.classes_}")

# Split by session (simulates real-world scenario)
train_idx = meta.query("session == '0train'").index.to_numpy()
test_idx = meta.query("session == '1test'").index.to_numpy()

print("\nData split:")
print(f"  Training (Session 1): {len(train_idx)} trials")
print(f"  Testing (Session 2): {len(test_idx)} trials")

######################################################################
# Model Selection Guide
# ---------------------
#
# SPD Learn provides several models optimized for different scenarios.
# Here's a guide to help you choose:
#
# **SPDNet** :cite:p:`huang2017riemannian` - The Classic Choice
#
# - Best for: Simple pipelines, pre-computed covariance matrices
# - Pros: Simple architecture, fast training, interpretable
# - Cons: No temporal feature learning
# - Use when: You want a baseline or have limited data
#
# **TSMNet** :cite:p:`kobler2022spd` - Best for Session Transfer
#
# - Best for: Cross-session/cross-subject scenarios
# - Pros: SPDBatchNormMeanVar enables domain adaptation without labels
# - Cons: More parameters, requires more data
# - Use when: You need to transfer to new sessions/subjects
#
# **EEGSPDNet** :cite:p:`wilson2025deep` - Channel-Specific Processing
#
# - Best for: When spatial information is important
# - Pros: Learns channel-specific temporal filters
# - Cons: More memory intensive
# - Use when: Channels have different characteristics
#
# **TensorCSPNet** :cite:p:`ju2022tensor` - Multi-Frequency Analysis
#
# - Best for: Filter bank approaches with multiple frequency bands
# - Pros: Captures frequency-specific spatial patterns
# - Cons: Requires FilterBankMotorImagery paradigm
# - Use when: Different frequency bands carry complementary information
#
# .. note::
#    For this tutorial, we'll compare SPDNet, TSMNet, and EEGSPDNet on
#    standard (single-band) motor imagery data.
#

n_chans = X.shape[1]  # 22 channels
n_outputs = len(le.classes_)  # 4 classes

print("\n" + "=" * 60)
print("Model Architectures")
print("=" * 60)

# SPDNet: Simple but effective
spdnet = SPDNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    subspacedim=n_chans,  # Keep full dimensionality
    threshold=1e-4,  # ReEig threshold
)
print("\nSPDNet:")
print(f"  Parameters: {sum(p.numel() for p in spdnet.parameters()):,}")
print("  Architecture: CovLayer -> BiMap -> ReEig -> LogEig -> Linear")

# TSMNet: With built-in feature extraction and batch normalization
tsmnet = TSMNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    n_temp_filters=8,  # Temporal filters
    temp_kernel_length=50,  # ~200ms at 250Hz
    n_spatiotemp_filters=32,  # Spatiotemporal features
    n_bimap_filters=16,  # BiMap output dimension
    reeig_threshold=1e-4,
)
print("\nTSMNet:")
print(f"  Parameters: {sum(p.numel() for p in tsmnet.parameters()):,}")
print(
    "  Architecture: TempConv -> SpatialConv -> CovLayer -> BiMap -> SPDBatchNormMeanVar -> LogEig"
)

# EEGSPDNet: Channel-specific convolution
eegspdnet = EEGSPDNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    n_filters=4,  # 4 filters per channel
    bimap_sizes=(2, 2),  # Scaling factor and depth
    filter_time_length=25,  # ~100ms at 250Hz
    spd_drop_prob=0.0,  # Disable SPD dropout for stability
)
print("\nEEGSPDNet:")
print(f"  Parameters: {sum(p.numel() for p in eegspdnet.parameters()):,}")
print("  Architecture: GroupedConv1D -> CovPool -> BiMap -> ReEig -> LogEig -> Linear")

######################################################################
# Training Configuration
# ----------------------
#
# SPD networks require careful hyperparameter selection for stable training.
# Here are the key settings:
#
# **Learning Rate**
#
# - Use low learning rates (1e-4 to 5e-4)
# - Riemannian optimization is sensitive to step size
# - Too high: Training diverges, NaN losses
# - Too low: Slow convergence
#
# **Gradient Clipping**
#
# - Essential for SPD networks
# - Prevents exploding gradients during eigenvalue operations
# - Recommended: gradient_clip_value=1.0
#
# **Optimizer**
#
# - Adam works well (adaptive learning rate)
# - SGD with momentum can also work but needs more tuning
# - AdamW with weight decay for regularization
#
# **Batch Size**
#
# - 16-64 trials typically works well
# - Larger batches give more stable covariance estimates
# - Limited by GPU memory
#
# **Early Stopping**
#
# - Monitor validation loss
# - Patience of 10-20 epochs is reasonable
#

# Training hyperparameters
batch_size = 32
max_epochs = 10  # Reduced for documentation build speed (was 200)
learning_rate = 1e-4  # CRITICAL: Use low learning rate

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*60}")
print("Training Configuration")
print("=" * 60)
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"Max epochs: {max_epochs}")
print(f"Learning rate: {learning_rate}")
print("Gradient clipping: 1.0")

######################################################################
# Training the Models
# -------------------
#
# We'll train each model and compare their performance. The key components
# of our training setup:
#
# - **EEGClassifier**: Braindecode wrapper for scikit-learn compatibility
# - **GradientNormClipping**: Prevents gradient explosion
# - **EarlyStopping**: Stops training when validation loss plateaus
# - **EpochScoring**: Tracks accuracy during training
#


def create_classifier(model, learning_rate=1e-4, max_epochs=100, batch_size=32):
    """Create an EEGClassifier with proper settings for SPD networks.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    learning_rate : float
        Learning rate (use low values like 1e-4).
    max_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Batch size for training.

    Returns
    -------
    EEGClassifier
        Configured classifier ready for training.
    """
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=learning_rate,
        # Validation split for early stopping and monitoring
        train_split=ValidSplit(0.2, stratified=True, random_state=42),
        batch_size=batch_size,
        max_epochs=max_epochs,
        callbacks=[
            # Track training accuracy
            (
                "train_acc",
                EpochScoring(
                    "accuracy", lower_is_better=False, on_train=True, name="train_acc"
                ),
            ),
            # CRITICAL: Gradient clipping for SPD network stability
            ("gradient_clip", GradientNormClipping(gradient_clip_value=1.0)),
            # Early stopping to prevent overfitting
            (
                "early_stop",
                EarlyStopping(
                    monitor="valid_loss",
                    patience=15,
                    threshold=1e-4,
                    lower_is_better=True,
                ),
            ),
        ],
        device=device,
        verbose=0,  # Set to 1 for training progress
    )
    return clf


# Store results for comparison
results = {}

######################################################################
# Training SPDNet
# ^^^^^^^^^^^^^^^
#

print("\n" + "=" * 60)
print("Training SPDNet")
print("=" * 60)

# Create fresh model instance
spdnet = SPDNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    subspacedim=n_chans,
    threshold=1e-4,
)

clf_spdnet = create_classifier(spdnet, learning_rate=1e-4)
clf_spdnet.fit(X[train_idx], y[train_idx])

# Evaluate
y_pred_train_spdnet = clf_spdnet.predict(X[train_idx])
y_pred_test_spdnet = clf_spdnet.predict(X[test_idx])

results["SPDNet"] = {
    "train_acc": accuracy_score(y[train_idx], y_pred_train_spdnet),
    "test_acc": accuracy_score(y[test_idx], y_pred_test_spdnet),
    "test_bal_acc": balanced_accuracy_score(y[test_idx], y_pred_test_spdnet),
    "y_pred": y_pred_test_spdnet,
    "history": clf_spdnet.history,
}

print(f"Train Accuracy: {results['SPDNet']['train_acc']*100:.2f}%")
print(f"Test Accuracy:  {results['SPDNet']['test_acc']*100:.2f}%")
print(f"Test Balanced:  {results['SPDNet']['test_bal_acc']*100:.2f}%")

######################################################################
# Training TSMNet
# ^^^^^^^^^^^^^^^
#

print("\n" + "=" * 60)
print("Training TSMNet")
print("=" * 60)

# Create fresh model instance
tsmnet = TSMNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    n_temp_filters=8,
    temp_kernel_length=50,
    n_spatiotemp_filters=32,
    n_bimap_filters=16,
    reeig_threshold=1e-4,
)

clf_tsmnet = create_classifier(tsmnet, learning_rate=1e-4)
clf_tsmnet.fit(X[train_idx], y[train_idx])

# Evaluate
y_pred_train_tsmnet = clf_tsmnet.predict(X[train_idx])
y_pred_test_tsmnet = clf_tsmnet.predict(X[test_idx])

results["TSMNet"] = {
    "train_acc": accuracy_score(y[train_idx], y_pred_train_tsmnet),
    "test_acc": accuracy_score(y[test_idx], y_pred_test_tsmnet),
    "test_bal_acc": balanced_accuracy_score(y[test_idx], y_pred_test_tsmnet),
    "y_pred": y_pred_test_tsmnet,
    "history": clf_tsmnet.history,
}

print(f"Train Accuracy: {results['TSMNet']['train_acc']*100:.2f}%")
print(f"Test Accuracy:  {results['TSMNet']['test_acc']*100:.2f}%")
print(f"Test Balanced:  {results['TSMNet']['test_bal_acc']*100:.2f}%")

######################################################################
# Training EEGSPDNet
# ^^^^^^^^^^^^^^^^^^
#

print("\n" + "=" * 60)
print("Training EEGSPDNet")
print("=" * 60)

# Create fresh model instance
eegspdnet = EEGSPDNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    n_filters=4,
    bimap_sizes=(2, 2),
    filter_time_length=25,
    spd_drop_prob=0.0,
)

clf_eegspdnet = create_classifier(eegspdnet, learning_rate=1e-4)
clf_eegspdnet.fit(X[train_idx], y[train_idx])

# Evaluate
y_pred_train_eegspdnet = clf_eegspdnet.predict(X[train_idx])
y_pred_test_eegspdnet = clf_eegspdnet.predict(X[test_idx])

results["EEGSPDNet"] = {
    "train_acc": accuracy_score(y[train_idx], y_pred_train_eegspdnet),
    "test_acc": accuracy_score(y[test_idx], y_pred_test_eegspdnet),
    "test_bal_acc": balanced_accuracy_score(y[test_idx], y_pred_test_eegspdnet),
    "y_pred": y_pred_test_eegspdnet,
    "history": clf_eegspdnet.history,
}

print(f"Train Accuracy: {results['EEGSPDNet']['train_acc']*100:.2f}%")
print(f"Test Accuracy:  {results['EEGSPDNet']['test_acc']*100:.2f}%")
print(f"Test Balanced:  {results['EEGSPDNet']['test_bal_acc']*100:.2f}%")

######################################################################
# Visualization of Results
# ------------------------
#
# Let's visualize the training progress and compare model performance.
#

fig = plt.figure(figsize=(16, 10))

# 1. Model Comparison Bar Chart
ax1 = fig.add_subplot(2, 3, 1)
models = list(results.keys())
test_accs = [results[m]["test_acc"] * 100 for m in models]
train_accs = [results[m]["train_acc"] * 100 for m in models]

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(
    x - width / 2, train_accs, width, label="Train", color="#3498db", alpha=0.8
)
bars2 = ax1.bar(
    x + width / 2, test_accs, width, label="Test", color="#e74c3c", alpha=0.8
)

ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_title("Model Comparison", fontsize=14, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.legend(fontsize=10)
ax1.set_ylim([0, 100])
ax1.axhline(y=25, color="gray", linestyle="--", alpha=0.5, label="Chance level")
ax1.grid(True, alpha=0.3, axis="y")

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(
        f"{height:.1f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(
        f"{height:.1f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# 2-4. Training Loss Curves
for idx, (model_name, color) in enumerate(
    [("SPDNet", "#3498db"), ("TSMNet", "#2ecc71"), ("EEGSPDNet", "#9b59b6")]
):
    ax = fig.add_subplot(2, 3, idx + 2)
    history = results[model_name]["history"]
    epochs = range(1, len(history) + 1)

    ax.plot(
        epochs, history[:, "train_loss"], "-", color=color, label="Train", linewidth=2
    )
    ax.plot(
        epochs,
        history[:, "valid_loss"],
        "--",
        color=color,
        alpha=0.7,
        label="Valid",
        linewidth=2,
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(f"{model_name} Training Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

# 5-6. Confusion Matrices (best and worst performing models)
sorted_models = sorted(
    results.keys(), key=lambda m: results[m]["test_acc"], reverse=True
)
best_model = sorted_models[0]

ax5 = fig.add_subplot(2, 3, 5)
cm = confusion_matrix(y[test_idx], results[best_model]["y_pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax5, cmap="Blues", values_format="d")
ax5.set_title(
    f"Best: {best_model}\nTest Acc: {results[best_model]['test_acc']*100:.1f}%",
    fontsize=12,
    fontweight="bold",
)

# Summary statistics
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis("off")
summary_text = "Results Summary\n" + "=" * 30 + "\n\n"
for model in sorted_models:
    summary_text += f"{model}:\n"
    summary_text += f"  Train: {results[model]['train_acc']*100:.1f}%\n"
    summary_text += f"  Test:  {results[model]['test_acc']*100:.1f}%\n"
    summary_text += f"  Balanced: {results[model]['test_bal_acc']*100:.1f}%\n\n"
summary_text += "=" * 30 + "\n"
summary_text += "Chance level: 25.0%"
ax6.text(
    0.1, 0.5, summary_text, fontsize=12, family="monospace", verticalalignment="center"
)

plt.tight_layout()
plt.show()

######################################################################
# Cross-Validation Evaluation
# ---------------------------
#
# For a more robust evaluation, we can use cross-validation. Since we have
# session information, we use session-based splits which better simulate
# real-world scenarios.
#
# .. note::
#    Due to computation time, we demonstrate with the simplest model.
#    For production, evaluate all models with proper cross-validation.
#

print("\n" + "=" * 60)
print("Cross-Validation Evaluation")
print("=" * 60)


def evaluate_cross_session(model_class, model_kwargs, subjects=[1, 2, 3]):
    """Evaluate model using cross-session validation.

    Parameters
    ----------
    model_class : class
        Model class to instantiate.
    model_kwargs : dict
        Keyword arguments for model initialization.
    subjects : list
        List of subject IDs to evaluate.

    Returns
    -------
    dict
        Dictionary with mean and std accuracy across subjects.
    """
    accuracies = []

    for subj in subjects:
        # Load data
        X_subj, labels_subj, meta_subj = paradigm.get_data(
            dataset=dataset, subjects=[subj], cache_config=cache_config
        )
        y_subj = le.fit_transform(labels_subj)

        # Split by session
        train_idx_subj = meta_subj.query("session == '0train'").index.to_numpy()
        test_idx_subj = meta_subj.query("session == '1test'").index.to_numpy()

        # Create fresh model
        model = model_class(**model_kwargs)
        clf = create_classifier(model, max_epochs=50)  # Fewer epochs for speed

        # Train and evaluate
        clf.fit(X_subj[train_idx_subj], y_subj[train_idx_subj])
        y_pred = clf.predict(X_subj[test_idx_subj])
        acc = accuracy_score(y_subj[test_idx_subj], y_pred)
        accuracies.append(acc)
        print(f"  Subject {subj}: {acc*100:.2f}%")

    return {"mean": np.mean(accuracies), "std": np.std(accuracies), "all": accuracies}


# Evaluate SPDNet on multiple subjects
print("\nSPDNet Cross-Session Evaluation:")
spdnet_cv = evaluate_cross_session(
    SPDNet,
    {
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "subspacedim": n_chans,
        "threshold": 1e-4,
    },
    subjects=[1],  # Reduced for documentation build speed (was [1, 2, 3])
)
print(f"\nSPDNet: {spdnet_cv['mean']*100:.1f}% +/- {spdnet_cv['std']*100:.1f}%")

######################################################################
# Troubleshooting Tips
# --------------------
#
# Here are common issues and solutions when training SPD networks:
#
# **Problem: NaN losses or diverging training**
#
# - Solution 1: Reduce learning rate (try 1e-5)
# - Solution 2: Increase gradient clipping threshold
# - Solution 3: Check for NaN/Inf in input data
# - Solution 4: Add small epsilon to covariance matrices for numerical stability
#
# **Problem: Model performs at chance level**
#
# - Solution 1: Verify data loading and label encoding
# - Solution 2: Check that train/test split is correct
# - Solution 3: Increase model capacity (more filters)
# - Solution 4: Try longer training with early stopping
#
# **Problem: Large gap between train and test accuracy (overfitting)**
#
# - Solution 1: Add regularization (weight decay in optimizer)
# - Solution 2: Use dropout (final_layer_drop_prob for EEGSPDNet)
# - Solution 3: Reduce model complexity
# - Solution 4: Use data augmentation
#
# **Problem: Training is too slow**
#
# - Solution 1: Use GPU (set device="cuda")
# - Solution 2: Reduce batch size
# - Solution 3: Use mixed precision training
# - Solution 4: Reduce model complexity
#
# **Problem: Out of memory errors**
#
# - Solution 1: Reduce batch size
# - Solution 2: Use gradient accumulation
# - Solution 3: Use a simpler model
#

print("\n" + "=" * 60)
print("Troubleshooting Checklist")
print("=" * 60)
print("""
If your model isn't working:

1. Data Quality
   [ ] Check for NaN/Inf values: np.any(np.isnan(X)) should be False
   [ ] Verify shapes: X should be (n_trials, n_channels, n_timepoints)
   [ ] Ensure proper filtering (8-35 Hz for motor imagery)

2. Training Settings
   [ ] Learning rate is low (1e-4 or lower)
   [ ] Gradient clipping is enabled (1.0)
   [ ] Batch size is reasonable (16-64)

3. Model Selection
   [ ] SPDNet: For simple baselines
   [ ] TSMNet: For session transfer (has SPDBatchNormMeanVar)
   [ ] EEGSPDNet: For channel-specific processing
   [ ] TensorCSPNet: For multi-frequency (filter bank)

4. Numerical Stability
   [ ] ReEig threshold > 0 (default 1e-4)
   [ ] SPD dropout disabled if unstable (spd_drop_prob=0.0)
""")

######################################################################
# Summary and Best Practices
# --------------------------
#
# In this tutorial, we covered:
#
# 1. **Data Loading**: Using MOABB for standardized EEG data access
# 2. **Preprocessing**: Default filtering for motor imagery (8-35 Hz)
# 3. **Model Selection**: SPDNet (simple), TSMNet (transfer), EEGSPDNet (spatial)
# 4. **Training**: Low learning rate (1e-4), gradient clipping, Adam optimizer
# 5. **Evaluation**: Cross-session validation for realistic estimates
#
# **Best Practices Summary**:
#
# - Always use gradient clipping with SPD networks
# - Start with low learning rates (1e-4) and increase if needed
# - Monitor both training and validation loss for overfitting
# - Use cross-session/cross-subject evaluation for realistic estimates
# - Consider TSMNet for session transfer scenarios
#
# **Next Steps**:
#
# - Try TensorCSPNet with FilterBankMotorImagery for multi-frequency analysis
# - Explore domain adaptation with TSMNet's SPDBatchNormMeanVar
# - Implement your own preprocessing pipeline for specific needs
#
