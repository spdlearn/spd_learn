"""
.. _spdim-domain-adaptation-example:

SPDIM: Source-Free Domain Adaptation on SPD Manifolds
=====================================================

This example reproduces the SPDIM pipeline :cite:p:`li2025spdim` for
source-free unsupervised domain adaptation (SFUDA) on SPD manifolds, using
the geometric operations in ``spd_learn.functional``.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# **SPDIM** (SPD Information Maximization) :cite:p:`li2025spdim` is a
# source-free unsupervised domain adaptation method for EEG-based BCIs that
# operates directly on the SPD manifold. It adapts a pre-trained TSMNet model
# to a new session/subject using only unlabeled target data.
#
# The Problem: Label Shift on SPD Manifolds
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In EEG-based BCIs, covariance features :math:`C_i \in \mathcal{S}_{++}^D`
# suffer from **conditional shift** (distribution shifts across domains) and
# **label shift** (the class priors change resulting in class-imbalance).
#
# A natural baseline is the **Recentering Transform (RCT)**
# :cite:p:`zanini2017transfer`, which centers SPD features around the
# identity by applying the congruence transformation:
#
# .. math::
#
#     \tilde{C}_i = \bar{C}_j^{-1/2} \, C_i \, \bar{C}_j^{-1/2}
#
# where :math:`\bar{C}_j` is the Fréchet mean of the target domain.
# However, the paper's **Proposition 2** shows that RCT only compensates
# conditional shift when the label priors are identical across domains.
# Under label shift, :math:`\bar{C}_j` is biased toward the
# over-represented class, causing RCT to **misalign**.
#
# SPDIM Overview
# ~~~~~~~~~~~~~~
#
# SPDIM addresses this by learning the transport parameters via an
# **Information Maximization (IM) loss** that does not require labels.
# SPDIM learns a full SPD reference matrix :math:`\Phi_j`
# that replaces the standard centering (Eq. 19 in the paper):
#
# .. math::
#
#     \tilde{C}_i = \Phi_j^{1/2} \, \bar{C}_j^{-1/2}
#     \, C_i \, \bar{C}_j^{-1/2} \, \Phi_j^{1/2}
#
# It initializes :math:`\Phi_j` with the target Fréchet mean and
# optimizes it as an SPD-constrained parameter via
# ``torch.nn.utils.parametrize`` and ``SymmetricPositiveDefinite``.
#
# SPDIM optimizes the **IM loss** (Eq. 21):
#
# .. math::
#
#     \mathcal{L}_{\mathrm{IM}} = \underbrace{H(Y | X)}_{\text{conditional
#     entropy}} - \underbrace{H(\bar{Y})}_{\text{marginal entropy}}
#
# Here, :math:`H(Y \mid X)` is the conditional entropy of the model
# predictions for each target sample. :math:`H(Y)` is the marginal
# entropy of the predicted labels, estimated here by
# :math:`H(\bar{Y})`, the entropy of the average predictive
# distribution across the target set. This encourages confident
# predictions (low :math:`H(Y \mid X)`) while maintaining class
# diversity (high :math:`H(Y)`).
#

######################################################################
# Setup and Imports
# -----------------
#

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch


warnings.filterwarnings("ignore")

######################################################################
# SPDIM Geometric Operations
# --------------------------
#
# The Fréchet mean and geodesic distances used by SPDIM are available
# directly from ``spd_learn.functional``.
#

######################################################################
# Loading the Dataset
# -------------------
#
# We use **BNCI2015_001** (2-class motor imagery: right hand vs feet),
# the same dataset used in the SPDIM paper. This dataset has 12 subjects,
# each with two sessions.
#
# We demonstrate cross-session transfer on **Subject 7**, which exhibits
# meaningful session-to-session variability.
#
# - **Source domain**: Session A (training with labels)
# - **Target domain**: Session B (adaptation without labels)
#
from braindecode.datasets import create_from_X_y
from moabb.datasets import BNCI2015_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import LabelEncoder

from spd_learn.functional import frechet_mean


dataset = BNCI2015_001()
paradigm = MotorImagery(
    n_classes=2,
    events=["right_hand", "feet"],
    fmin=4,
    fmax=36,
    tmin=1.0,
    tmax=4.0,
    resample=256,
)

subject_id = 7

cache_config = dict(
    save_raw=True,
    save_epochs=True,
    save_array=True,
    use=True,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
)

X, labels, meta = paradigm.get_data(
    dataset=dataset,
    subjects=[subject_id],
    cache_config=cache_config,
)

le = LabelEncoder()
y = le.fit_transform(labels)

sfreq = int(paradigm.resample)

# Split by session
sessions = meta["session"].unique()
source_session, target_session = sessions[0], sessions[1]

source_idx = meta.query(f"session == '{source_session}'").index.to_numpy()
target_idx = meta.query(f"session == '{target_session}'").index.to_numpy()

X_source, y_source = X[source_idx], y[source_idx]
X_target, y_target = X[target_idx], y[target_idx]

# Create braindecode dataset for target domain
target_ds = create_from_X_y(
    X_target,
    y_target,
    drop_last_window=True,
    sfreq=sfreq,
)

print(f"Dataset: {dataset.code}")
print(f"Subject {subject_id}: Session {source_session} -> {target_session}")
print(f"Source domain: {len(X_source)} samples")
print(f"Target domain: {len(target_ds)} samples")
print(f"Classes: {le.classes_}")

######################################################################
# Simulating Label Shift in Target Domain
# ----------------------------------------
#
# A key contribution of SPDIM is handling **label shift** --- when the
# class priors differ between source and target domains.
#
# Following the paper's ``get_label_ratio`` protocol with
# ``ratio_level=0.2``: we keep all samples of the last class and
# subsample the other class(es) to 20%. This creates a 5:1 class
# imbalance, making the Fréchet mean biased toward the majority class.
#
# As shown by the paper's **Proposition 2**, this biased mean causes
# RCT to misalign: the recentered features no longer align with
# the source domain's learned decision boundary.
#

ratio_level = 0.2

# SPDIM protocol: subsample all classes except the last to ratio_level
rng = np.random.RandomState(42)
classes = sorted(np.unique(y_target))
subsample_inds = np.sort(
    np.concatenate(
        [
            rng.choice(
                np.flatnonzero(y_target == c),
                size=math.ceil(
                    np.sum(y_target == c)
                    * (ratio_level if i < len(classes) - 1 else 1.0)
                ),
                replace=False,
            )
            for i, c in enumerate(classes)
        ]
    )
)

target_shifted_ds = target_ds.split(by={"shifted": subsample_inds.tolist()})["shifted"]

# Keep arrays for SPDIM adaptation methods
X_target_shifted = X_target[subsample_inds]
y_target_shifted = y_target[subsample_inds]

print(f"\nAfter label shift (ratio_level={ratio_level}):")
print(f"  Target samples: {len(target_shifted_ds)}")
for c in np.unique(y_target_shifted):
    n = (y_target_shifted == c).sum()
    print(f"  Class {le.classes_[c]}: {n} ({100 * n / len(y_target_shifted):.0f}%)")

######################################################################
# Training TSMNet on Source Domain
# --------------------------------
#
# We train TSMNet using braindecode's ``EEGClassifier`` wrapper with:
#
# - **AdamW optimizer** with ``lr=1e-3`` and ``weight_decay=1e-4``
# - **Gradient clipping** (max norm 1.0) for stable SPD optimization
# - **Validation split** (10%) for early stopping
#

from braindecode import EEGClassifier
from skorch.callbacks import GradientNormClipping
from skorch.dataset import ValidSplit

from spd_learn.models import TSMNet


n_chans = X_source.shape[1]
n_outputs = len(le.classes_)

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

model = TSMNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    n_temp_filters=4,
    temp_kernel_length=25,
    n_spatiotemp_filters=40,
    n_bimap_filters=20,
    reeig_threshold=1e-4,
)

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=1e-4,
    optimizer__weight_decay=1e-4,
    train_split=ValidSplit(0.1, stratified=True, random_state=42),
    batch_size=32,
    max_epochs=30,  # Reduced from 200 for faster documentation build
    callbacks=[
        ("gradient_clip", GradientNormClipping(gradient_clip_value=5.0)),
    ],
    device=device,
    verbose=1,
)

print("\n" + "=" * 50)
print("Training TSMNet on Source Domain")
print("=" * 50)
clf.fit(X_source, y_source)

######################################################################
# Baseline: No Adaptation
# -----------------------
#
# Evaluate the source-trained model on target domain without adaptation.
#

from sklearn.metrics import balanced_accuracy_score


underlying_model = clf.module_

y_pred_source = clf.predict(X_source)
source_bacc = balanced_accuracy_score(y_source, y_pred_source)

y_pred_target_no_adapt = clf.predict(target_shifted_ds)
no_adapt_bacc = balanced_accuracy_score(y_target_shifted, y_pred_target_no_adapt)

print(f"\n{'=' * 50}")
print("Baseline: No Adaptation")
print(f"{'=' * 50}")
print(f"Source Balanced Accuracy: {source_bacc * 100:.2f}%")
print(f"Target Balanced Accuracy: {no_adapt_bacc * 100:.2f}%")
print(f"Performance Drop: {(source_bacc - no_adapt_bacc) * 100:.2f}%")

######################################################################
# Helper Functions
# ----------------
#
# For SPDIM, we intercept TSMNet's forward pass before
# SPDBatchNormMeanVar and replace the centering step with learnable
# geodesic transport, while keeping the variance normalization and
# rebiasing from the trained BN layer.
#

from spd_learn.modules import LogEig


def extract_spd_features(model, X_data, batch_size=32):
    """Extract SPD features before batch normalization."""
    model.eval()
    if not isinstance(X_data, torch.Tensor):
        dtype = next(model.parameters()).dtype
        X_data = torch.tensor(X_data, dtype=dtype)
    dev = next(model.parameters()).device
    X_data = X_data.to(dev)

    spd_list = []
    with torch.no_grad():
        for i in range(0, len(X_data), batch_size):
            batch = X_data[i : i + batch_size]
            x = model.cnn(batch[:, None, ...])
            x = model.covpool(x)
            x = model.spdnet(x)
            spd_list.append(x.cpu())

    return torch.cat(spd_list, dim=0)


def spdim_forward(model, X_spd, adapter=None):
    """SPDIM test-time forward pass.

    Matches the original SPDIM test-time pipeline
    (``transp_geosedic_identity_transp``): geodesic transport + LogEig
    + classifier, without dispersion normalization.

    1. Geodesic transport: A^{-t/2} X A^{-t/2}
    2. LogEig (tangent space mapping)
    3. Classifier
    """
    # Geodesic transport (replaces BN centering)
    if adapter is not None:
        X_transported = adapter(X_spd)

    # LogEig + classifier
    logeig = LogEig(upper=True, flatten=True)
    X_tangent = logeig(X_transported)
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    logits = model.head(X_tangent.to(dev, dtype=dtype))
    return logits


######################################################################
# SFUDA Step 1: Refit BN Statistics (RCT Baseline)
# -------------------------------------------------
#
# The **Recentering Transform (RCT)** :cite:p:`zanini2017transfer`
# baseline recomputes the Fréchet mean and variance on target SPD
# features using the full Karcher flow. This corresponds to setting
# :math:`\varphi = 1` (standard centering) in the geodesic transport.
#
# Under label shift, Proposition 2 predicts that this will degrade
# performance because the biased Fréchet mean shifts features away
# from the source decision boundary.
#

# Save original running stats
orig_running_mean = underlying_model.spdbnorm.running_mean.clone()
orig_running_var = underlying_model.spdbnorm.running_var.clone()


def refit_spdbn_frechet(model, X_data, batch_size=32):
    """Refit SPDBatchNormMeanVar using the Fréchet mean (SPDIM style)."""
    X_spd = extract_spd_features(model, X_data, batch_size=batch_size)
    mean, distances = frechet_mean(X_spd, max_iter=50, return_distances=True)
    variance = distances.square().mean(dim=0, keepdim=True).squeeze()
    with torch.no_grad():
        model.spdbnorm.running_mean.copy_(mean)
        model.spdbnorm.running_var.fill_(variance.item())


print(f"\n{'=' * 50}")
print("SFUDA Step 1: Refit BN Statistics (RCT)")
print(f"{'=' * 50}")

refit_spdbn_frechet(underlying_model, X_target_shifted)

target_frechet_mean = underlying_model.spdbnorm.running_mean.clone()

rct_pred = clf.predict(target_shifted_ds)
rct_bacc = balanced_accuracy_score(y_target_shifted, rct_pred)
print(f"RCT Balanced Accuracy: {rct_bacc * 100:.2f}%")
print(f"Improvement over baseline: {(rct_bacc - no_adapt_bacc) * 100:+.2f}%")

# Restore original stats
underlying_model.spdbnorm.running_mean.copy_(orig_running_mean)
underlying_model.spdbnorm.running_var.copy_(orig_running_var)

######################################################################
# Information Maximization Loss
# -----------------------------
#
# The IM loss encourages confident predictions (low conditional entropy)
# while maintaining class diversity (high marginal entropy).
#

import torch.nn.functional as F


def im_loss(logits, temperature=2.0):
    """Information Maximization loss (matching SPDIM paper)."""
    p = F.softmax(logits / temperature, dim=1)
    ce = -(p * torch.log(p + 1e-5)).sum(dim=1).mean()
    p_bar = p.mean(dim=0)
    me = -(p_bar * torch.log(p_bar + 1e-5)).sum()
    return ce - me


######################################################################
# SPDIM(bias) Strategy
# --------------------
#
# SPDIM(bias) (Eq. 19) learns a full SPD reference mean that replaces
# the (biased) Fréchet mean in the geodesic transport. With
# :math:`D(D+1)/2` degrees of freedom (vs 1 scalar for geodesic), it
# can compensate both conditional and label shift.
#
# We initialize the learnable mean with the target Fréchet mean and
# keep it on the SPD manifold via ``SymmetricPositiveDefinite``.
#
# Learnable SPD Recenter Module
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from torch.nn.utils.parametrize import register_parametrization

from spd_learn.functional import matrix_inv_sqrt
from spd_learn.modules.manifold import SymmetricPositiveDefinite


class SPDLearnableRecenter(torch.nn.Module):
    def __init__(
        self,
        num_features,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_features = num_features

        self.bias = torch.nn.Parameter(
            torch.empty(1, num_features, num_features, device=device, dtype=dtype),
        )

        self.reset_parameters()
        register_parametrization(self, "bias", SymmetricPositiveDefinite())

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.bias.zero_()
        self.bias[0].fill_diagonal_(1.0)

    def forward(self, input):
        bias_inv_sqrt = matrix_inv_sqrt.apply(self.bias)
        output = bias_inv_sqrt @ input @ bias_inv_sqrt
        return output


######################################################################
# SPDIM(bias) Optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~
#

print(f"\n{'=' * 50}")
print("SPDIM(bias): Learnable SPD Mean")
print(f"{'=' * 50}")

X_spd_target = extract_spd_features(underlying_model, X_target_shifted, batch_size=32)

print(f"SPD reference initialized. Shape: {target_frechet_mean.shape}")

adapter = SPDLearnableRecenter(target_frechet_mean.shape[-1])
adapter.bias = target_frechet_mean.clone()

optimizer_bias = torch.optim.Adam(adapter.parameters(), lr=0.05)
n_epochs_bias = 30  # Reduced from 200 for faster documentation build
losses_bias = []
best_loss_bias = float("inf")
best_bias = target_frechet_mean.clone().detach()
for epoch in range(n_epochs_bias):
    optimizer_bias.zero_grad()
    logits = spdim_forward(
        underlying_model,
        X_spd_target,
        adapter,
    )
    loss = im_loss(logits, temperature=2.0)
    loss.backward()
    optimizer_bias.step()

    current_loss = loss.item()
    losses_bias.append(current_loss)
    if current_loss < best_loss_bias:
        best_loss_bias = current_loss
        best_bias = adapter.bias.clone().detach()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:3d}/{n_epochs_bias}: loss={current_loss:.4f}")

# Evaluate with best parameters
with torch.no_grad():
    adapter.bias = best_bias
    logits = spdim_forward(
        underlying_model,
        X_spd_target,
        adapter,
    )
    y_pred_bias = logits.argmax(dim=1).cpu().numpy()

bias_bacc = balanced_accuracy_score(y_target_shifted, y_pred_bias)
print(f"\nSPDIM(bias) Balanced Accuracy: {bias_bacc * 100:.2f}%")
print(f"Improvement over baseline: {(bias_bacc - no_adapt_bacc) * 100:+.2f}%")

######################################################################
# Results Summary
# ---------------
#

results = {
    "No Adaptation": no_adapt_bacc,
    "RCT (Refit BN)": rct_bacc,
    "SPDIM(bias)": bias_bacc,
}

print(f"\n{'=' * 60}")
print(f"Results Summary (Subject {subject_id}, Label Shift ratio={ratio_level})")
print(f"{'=' * 60}")
print(f"{'Method':<25} {'Bal. Accuracy':>14} {'vs Baseline':>14}")
print("-" * 57)
for method, acc in results.items():
    if method == "No Adaptation":
        print(f"{method:<25} {acc * 100:>12.2f}% {'-':>14}")
    else:
        imp = acc - no_adapt_bacc
        print(f"{method:<25} {acc * 100:>12.2f}% {imp * 100:>+12.2f}%")
print("-" * 57)
print("Chance level: 50.00% (2 classes)")

best_method = max(results.keys(), key=lambda k: results[k])
print(f"\nBest method: {best_method} ({results[best_method] * 100:.2f}%)")

######################################################################
# Visualizing Results
# -------------------
#

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Bar chart
ax1 = axes[0]
methods = list(results.keys())
accuracies = [results[m] * 100 for m in methods]
colors = ["#e74c3c", "#3498db", "#2ecc71"]
bars = ax1.bar(
    range(len(methods)),
    accuracies,
    color=colors,
    edgecolor="black",
    linewidth=1.5,
)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
ax1.set_ylabel("Balanced Accuracy (%)", fontsize=12)
ax1.set_title("Domain Adaptation Comparison", fontsize=14)
ax1.set_ylim([0, 100])
ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
ax1.axhline(
    y=source_bacc * 100,
    color="blue",
    linestyle=":",
    alpha=0.5,
    label=f"Source ({source_bacc * 100:.1f}%)",
)
for bar, acc in zip(bars, accuracies):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
ax1.legend(loc="lower right", fontsize=8)

# 2. SPDIM(bias) loss curve
ax2 = axes[1]
ax2.plot(range(1, len(losses_bias) + 1), losses_bias, "r-", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("IM Loss", fontsize=12)
ax2.set_title("SPDIM(bias) Optimization", fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

######################################################################
# Training History
# ~~~~~~~~~~~~~~~~
#

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
history = clf.history
epochs_hist = range(1, len(history) + 1)
ax.plot(epochs_hist, history[:, "train_loss"], "b-", label="Train Loss", linewidth=2)
ax.plot(epochs_hist, history[:, "valid_loss"], "r--", label="Valid Loss", linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("TSMNet Training History", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

######################################################################
# Discussion
# ----------
#
# In this example we reproduced the SPDIM pipeline for source-free
# domain adaptation on SPD manifolds. The results illustrate the
# paper's theoretical predictions.
#
# Why RCT degrades under label shift
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Recentering Transform (RCT) :cite:p:`zanini2017transfer` computes
# the Fréchet mean of the target domain and uses it to center the SPD
# features. Under **label shift**, the Fréchet mean is biased toward
# the over-represented class (here, *feet* at 83% of samples).
#
# As predicted by **Proposition 2** of the paper, this biased mean
# causes the recentered features to no longer align with the source
# domain's learned decision boundary, resulting in degraded accuracy.
# Without label shift, RCT typically *improves* accuracy.
#
# Key implementation details
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The following details match the original SPDIM code:
#
# - **Temperature = 2.0** in the IM loss softmax (T=0.8 for
#   multi-class tasks).
# - **Best-model tracking**: Returns the parameter with lowest IM loss.
# - **Test-time BN**: Only geodesic transport (no dispersion
#   normalization), matching the original SPDIM test-time pipeline.
#
# References
# ----------
#
# .. bibliography::
#    :filter: False
#
#    li2025spdim
#    zanini2017transfer
#    kobler2022spd
#

# Cleanup
plt.close("all")
