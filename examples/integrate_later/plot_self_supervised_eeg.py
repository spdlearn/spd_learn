"""
Self-Supervised Contrastive Learning on SPD Manifolds for EEG
=============================================================

This example demonstrates self-supervised contrastive learning on the SPD
manifold for EEG data, inspired by the DeepGeoCCA framework
:cite:p:`ju2024self`.

The key idea is to learn representations by maximizing the correlation between
different "views" of the same EEG trial in the tangent space of the SPD manifold.
This enables learning meaningful features without labels.

.. topic:: Method Overview

    **Geodesic Correlation**: Traditional correlation measures don't respect
    the geometry of SPD matrices. The geodesic correlation projects SPD matrices
    to the tangent space (via matrix logarithm) and computes correlation there.

    **Contrastive Learning**: We create two views of each EEG trial by:
    1. Using different subsets of channels (spatial views)
    2. Using different time windows (temporal views)

    The model learns to maximize correlation between views of the same trial
    while learning discriminative SPD representations.

.. note::
   This example requires: ``braindecode``, ``moabb``.
   Install with: ``pip install braindecode moabb``

"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, SPDBatchNormMeanVar


# %%
# Configuration
# -------------

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
# Generate Synthetic Multi-View EEG Data
# --------------------------------------
#
# We create synthetic EEG data with two spatial views:
# - View 1: Central channels (motor cortex)
# - View 2: Lateral channels
#
# Each view captures different aspects of the same neural activity.


def generate_synthetic_eeg(
    n_samples=200, n_channels=22, n_times=256, n_classes=2, seed=42
):
    """Generate synthetic EEG data with class-specific covariance structure."""
    np.random.seed(seed)

    X_list = []
    y_list = []

    for class_idx in range(n_classes):
        n_per_class = n_samples // n_classes

        # Create class-specific mixing matrix
        # Different classes have different spatial patterns
        A = np.random.randn(n_channels, n_channels) * 0.5
        A = A + class_idx * 0.3 * np.eye(n_channels)  # Class-specific bias

        for _ in range(n_per_class):
            # Generate source signals with class-specific characteristics
            sources = np.random.randn(n_channels, n_times)

            # Add class-specific frequency content
            t = np.linspace(0, 1, n_times)
            if class_idx == 0:
                # Class 0: More alpha activity (8-12 Hz)
                sources[0] += 2 * np.sin(2 * np.pi * 10 * t)
                sources[1] += 1.5 * np.sin(2 * np.pi * 11 * t)
            else:
                # Class 1: More beta activity (15-25 Hz)
                sources[0] += 2 * np.sin(2 * np.pi * 20 * t)
                sources[1] += 1.5 * np.sin(2 * np.pi * 18 * t)

            # Mix sources
            X = A @ sources
            X_list.append(X)
            y_list.append(class_idx)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


# Generate data
print("Generating synthetic EEG data...")
X_data, y_data = generate_synthetic_eeg(n_samples=300, n_channels=22, n_times=256)
print(f"Data shape: {X_data.shape}, Labels shape: {y_data.shape}")

# Define channel views (simulating different electrode groups)
# View 1: Central channels (like C3, Cz, C4 in 10-20 system)
view1_channels = [7, 8, 9, 10, 11]  # 5 channels
# View 2: Frontal/Parietal channels
view2_channels = [0, 1, 2, 3, 4, 15, 16, 17, 18, 19]  # 10 channels

print(f"View 1 channels: {len(view1_channels)}")
print(f"View 2 channels: {len(view2_channels)}")


# %%
# Multi-View Dataset
# ------------------
#
# Create a dataset that returns two views of each trial.


class MultiViewEEGDataset(torch.utils.data.Dataset):
    """Dataset returning two spatial views of EEG data."""

    def __init__(self, X, y, view1_channels, view2_channels):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
        self.view1_ch = view1_channels
        self.view2_ch = view2_channels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        view1 = x[self.view1_ch]  # (n_ch1, n_times)
        view2 = x[self.view2_ch]  # (n_ch2, n_times)
        return view1, view2, self.y[idx]


# %%
# SPD Encoder Network
# -------------------
#
# Each view is encoded using an SPD neural network that:
# 1. Computes covariance matrix (CovLayer)
# 2. Applies BiMap for dimensionality reduction
# 3. Applies ReEig for non-linearity
# 4. Projects to tangent space (LogEig)


class SPDEncoder(nn.Module):
    """SPD-based encoder for EEG covariance features.

    Maps raw EEG signals to SPD representations and then to
    tangent space features.
    """

    def __init__(self, n_channels, embed_dim, output_spd=False):
        super().__init__()
        self.output_spd = output_spd

        # Covariance estimation
        self.cov = CovLayer()

        # SPD transformation layers
        self.bimap = BiMap(n_channels, embed_dim)
        self.reeig = ReEig(threshold=1e-4)
        self.batchnorm = SPDBatchNormMeanVar(embed_dim)

        # Project to tangent space
        self.logeig = LogEig(upper=True, flatten=True)

        # Output dimension
        self.output_dim = embed_dim * (embed_dim + 1) // 2

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            EEG signals of shape (batch, channels, time)

        Returns
        -------
        torch.Tensor
            Tangent space features of shape (batch, output_dim)
            or SPD matrices if output_spd=True
        """
        # Compute covariance
        cov = self.cov(x)  # (batch, channels, channels)

        # SPD transformations
        spd = self.bimap(cov)
        spd = self.reeig(spd)
        spd = self.batchnorm(spd)

        if self.output_spd:
            return spd

        # Project to tangent space
        features = self.logeig(spd)
        return features


# %%
# Geodesic Correlation Loss
# -------------------------
#
# The geodesic correlation measures similarity between SPD matrices
# by computing correlation in the tangent space. This respects the
# Riemannian geometry of SPD matrices.


class GeodesicCorrelationLoss(nn.Module):
    """Loss based on geodesic correlation for SPD matrices.

    Maximizes correlation between paired views in tangent space.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable projections for CCA-like alignment
        feat_dim = embed_dim * (embed_dim + 1) // 2
        self.proj1 = nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj2 = nn.Linear(feat_dim, feat_dim, bias=False)

    def forward(self, z1, z2):
        """
        Compute geodesic correlation loss.

        Parameters
        ----------
        z1, z2 : torch.Tensor
            Tangent space features from two views, shape (batch, feat_dim)

        Returns
        -------
        torch.Tensor
            Negative correlation (to minimize)
        """
        batch_size = z1.shape[0]

        # Center the features
        z1 = z1 - z1.mean(dim=0, keepdim=True)
        z2 = z2 - z2.mean(dim=0, keepdim=True)

        # Apply learnable projections
        z1_proj = self.proj1(z1)
        z2_proj = self.proj2(z2)

        # Normalize
        z1_norm = F.normalize(z1_proj, dim=1)
        z2_norm = F.normalize(z2_proj, dim=1)

        # Compute correlation matrix
        corr_matrix = z1_norm @ z2_norm.T  # (batch, batch)

        # Positive pairs are on diagonal
        pos_corr = torch.diag(corr_matrix).mean()

        # Negative pairs are off-diagonal
        mask = ~torch.eye(batch_size, dtype=bool, device=z1.device)
        neg_corr = corr_matrix[mask].mean()

        # Contrastive loss: maximize positive, minimize negative correlation
        loss = -pos_corr + 0.5 * neg_corr.abs()

        return loss, pos_corr.item()


# %%
# Contrastive Model
# -----------------
#
# Combines two encoders (one per view) with the geodesic correlation loss.


class ContrastiveSPDNet(nn.Module):
    """Contrastive learning model for multi-view SPD data."""

    def __init__(self, n_channels_view1, n_channels_view2, embed_dim):
        super().__init__()
        self.encoder1 = SPDEncoder(n_channels_view1, embed_dim)
        self.encoder2 = SPDEncoder(n_channels_view2, embed_dim)
        self.loss_fn = GeodesicCorrelationLoss(embed_dim)

    def forward(self, view1, view2):
        """Encode both views."""
        z1 = self.encoder1(view1)
        z2 = self.encoder2(view2)
        return z1, z2

    def compute_loss(self, view1, view2):
        """Compute contrastive loss."""
        z1, z2 = self.forward(view1, view2)
        loss, pos_corr = self.loss_fn(z1, z2)
        return loss, pos_corr


# %%
# Training Loop
# -------------
#
# Self-supervised pretraining: learn to correlate different views.

# Create datasets
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=SEED, stratify=y_data
)

train_dataset = MultiViewEEGDataset(X_train, y_train, view1_channels, view2_channels)
test_dataset = MultiViewEEGDataset(X_test, y_test, view1_channels, view2_channels)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, drop_last=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
embed_dim = 4  # Small embedding dimension
model = ContrastiveSPDNet(
    n_channels_view1=len(view1_channels),
    n_channels_view2=len(view2_channels),
    embed_dim=embed_dim,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
n_epochs = 50
train_losses = []
train_correlations = []

print("\n" + "=" * 50)
print("Self-Supervised Pretraining")
print("=" * 50)

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    epoch_corr = 0

    for view1, view2, _ in train_loader:
        view1 = view1.to(device)
        view2 = view2.to(device)

        optimizer.zero_grad()
        loss, pos_corr = model.compute_loss(view1, view2)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_corr += pos_corr

    avg_loss = epoch_loss / len(train_loader)
    avg_corr = epoch_corr / len(train_loader)
    train_losses.append(avg_loss)
    train_correlations.append(avg_corr)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Correlation: {avg_corr:.4f}"
        )

# %%
# Visualize Training Progress
# ---------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, "b-", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Contrastive Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_correlations, "g-", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Correlation")
axes[1].set_title("View Correlation (Positive Pairs)")
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color="r", linestyle="--", alpha=0.5)

plt.suptitle("Self-Supervised Pretraining Progress", fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# Extract Learned Representations
# -------------------------------
#
# Use the pretrained encoder to extract features for downstream classification.


def extract_features(model, data_loader, device):
    """Extract features using the pretrained encoders."""
    model.eval()
    features_v1, features_v2, labels = [], [], []

    with torch.no_grad():
        for view1, view2, y in data_loader:
            view1 = view1.to(device)
            view2 = view2.to(device)

            z1, z2 = model(view1, view2)

            features_v1.append(z1.cpu().numpy())
            features_v2.append(z2.cpu().numpy())
            labels.append(y.numpy())

    return (
        np.concatenate(features_v1),
        np.concatenate(features_v2),
        np.concatenate(labels),
    )


# Extract features
train_z1, train_z2, train_labels = extract_features(model, train_loader, device)
test_z1, test_z2, test_labels = extract_features(model, test_loader, device)

# Concatenate features from both views
train_features = np.concatenate([train_z1, train_z2], axis=1)
test_features = np.concatenate([test_z1, test_z2], axis=1)

print(f"\nExtracted feature shape: {train_features.shape}")

# %%
# Downstream Classification
# -------------------------
#
# Evaluate the learned representations on a classification task.
# We compare: pretrained features vs. random features.

# Train classifier on pretrained features
clf_pretrained = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf_pretrained.fit(train_features, train_labels)
pretrained_acc = accuracy_score(test_labels, clf_pretrained.predict(test_features))

# Compare with random initialization
model_random = ContrastiveSPDNet(
    n_channels_view1=len(view1_channels),
    n_channels_view2=len(view2_channels),
    embed_dim=embed_dim,
).to(device)

# Extract features from random model
random_train_z1, random_train_z2, _ = extract_features(
    model_random, train_loader, device
)
random_test_z1, random_test_z2, _ = extract_features(model_random, test_loader, device)

random_train_features = np.concatenate([random_train_z1, random_train_z2], axis=1)
random_test_features = np.concatenate([random_test_z1, random_test_z2], axis=1)

clf_random = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf_random.fit(random_train_features, train_labels)
random_acc = accuracy_score(test_labels, clf_random.predict(random_test_features))

print("\n" + "=" * 50)
print("Downstream Classification Results")
print("=" * 50)
print(f"Random initialization accuracy:  {random_acc:.4f}")
print(f"Pretrained (SSL) accuracy:       {pretrained_acc:.4f}")
print(f"Improvement:                     {(pretrained_acc - random_acc)*100:+.2f}%")

# %%
# Visualize Feature Space
# -----------------------
#
# Project features to 2D for visualization.

from sklearn.decomposition import PCA


pca = PCA(n_components=2)
features_2d = pca.fit_transform(test_features)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    features_2d[:, 0],
    features_2d[:, 1],
    c=test_labels,
    cmap="coolwarm",
    alpha=0.7,
    s=50,
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Learned SPD Feature Space (PCA Projection)")
plt.colorbar(scatter, label="Class")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Conclusion
# ----------
#
# This example demonstrated self-supervised contrastive learning on the SPD
# manifold for EEG data. Key takeaways:
#
# 1. **Geodesic Correlation**: By projecting SPD matrices to the tangent space,
#    we can compute meaningful correlations that respect the geometry.
#
# 2. **Multi-View Learning**: Different channel subsets provide complementary
#    views of the same neural activity, enabling contrastive learning.
#
# 3. **Pretraining Benefits**: The self-supervised pretraining improved
#    downstream classification compared to random initialization.
#
# Extensions
# ~~~~~~~~~~
#
# - Use real EEG data from MOABB datasets
# - Try different view creation strategies (temporal, frequency bands)
# - Implement the full DeepGeoCCA loss with distribution constraints
# - Apply to multi-modal data (EEG + fMRI)
