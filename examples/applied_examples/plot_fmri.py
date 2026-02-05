"""=================================
Classifying fMRI data with SPDNet
=================================

This example demonstrates how to use **spd_learn**'s
:class:`SPDNet` to distinguish autism spectrum disorder (ASD)
from typical controls (TC) with resting-state fMRI from the **ABIDE
Pre-processed Connectome Project** :cite:p:`nielsen2014abnormal`.

This example compares the performance of the :class:`SPDNet` with a
stronge and robust baseline classifier that uses the tangent space of the
covariance matrices, followed by a linear classifier
(:class:`sklearn.linear_model.LogisticRegression`).

More details about the baseline can be found in the benchmark
:cite:p:`dadi2019benchmarking`.

The results show that the SPDNet have competitive results the baseline, even
if don't reach the state-of-the-art performance.

This tutorial shows how to:

- Fetch the ABIDE preprocessed connectome project (PCP) dataset using
  nilearn :cite:p:`nilearn`.
- The dataset contains resting-state fMRI time-series of 400 subjects.
- Compute the covariance matrices of the time-series.
- Train a :class:`SPDNet` on the training set.
- Evaluate the model on the held-out test set.

You can cut runtime further by lowering ``N_SUBJECTS`` or removing the
variable to use the full dataset.

The code have strong similarities with the braindecode example.

"""

# Imports & configuration
# -----------------------
import matplotlib.pyplot as plt
import numpy as np
import torch

from braindecode import EEGClassifier
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from spd_learn.models import SPDNet


SEED = 42
N_SUBJECTS = 400

# Set random seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


######################################################################
# Fetching the ABIDE Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ABIDE (Autism Brain Imaging Data Exchange) dataset contains
# resting-state fMRI data from multiple sites. We use the CC200 atlas
# parcellation which extracts time-series from 200 brain regions.

abide = datasets.fetch_abide_pcp(
    derivatives=["rois_cc200"],
    n_subjects=N_SUBJECTS,
    verbose=1,
)

ts = abide.rois_cc200
phenotypic = abide.phenotypic

# Create diagnosis: 0 for control, 1 for autism
y = phenotypic["DX_GROUP"].replace({2: 0, 1: 1}).to_numpy()

print(f"Labels: {np.unique(y, return_counts=True)}")

######################################################################
# Visualizing the Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's visualize a sample time-series and compute a sample connectivity
# matrix to understand the data structure.

# Compute sample covariance matrices for visualization
covariance_viz = ConnectivityMeasure(kind="correlation")
sample_corr = covariance_viz.fit_transform([ts[0]])[0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot sample time-series (first 10 ROIs, first 100 time points)
ax1 = axes[0]
time_points = min(100, ts[0].shape[0])
n_rois_show = 10
for i in range(n_rois_show):
    ax1.plot(ts[0][:time_points, i], alpha=0.7, label=f"ROI {i+1}")
ax1.set_xlabel("Time (TRs)")
ax1.set_ylabel("BOLD Signal")
ax1.set_title("Sample fMRI Time-Series\n(10 ROIs)", fontweight="bold")
ax1.grid(True, alpha=0.3)

# Plot correlation matrix
ax2 = axes[1]
im = ax2.imshow(sample_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax2.set_xlabel("ROI")
ax2.set_ylabel("ROI")
ax2.set_title("Sample Correlation Matrix\n(200 x 200)", fontweight="bold")
plt.colorbar(im, ax=ax2, shrink=0.8, label="Correlation")

# Plot class distribution
ax3 = axes[2]
unique, counts = np.unique(y, return_counts=True)
colors = ["#3498db", "#e74c3c"]
bars = ax3.bar(["Control", "Autism"], counts, color=colors, alpha=0.8)
ax3.set_ylabel("Number of Subjects")
ax3.set_title("Class Distribution\n(ABIDE Dataset)", fontweight="bold")
ax3.grid(True, alpha=0.3, axis="y")
for bar, count in zip(bars, counts):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        str(count),
        ha="center",
        fontweight="bold",
    )

plt.suptitle(f"ABIDE Dataset Overview (N={N_SUBJECTS})", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()


######################################################################
# Creating the connectivity matrices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, we compute the covariance matrices of the time-series
# using the :class:`nilearn.connectome.ConnectivityMeasure` class.

covariance = ConnectivityMeasure(kind="covariance")

X = np.array([covariance.fit_transform([t])[0] for t in ts])

######################################################################
# Splitting the data
# ~~~~~~~~~~~~~~~~~~
#
# As we are handle few instance scenary, we will use a stratified split
# to ensure that the training and test sets have the same proportion of
# classes as the original dataset. This is important to ensure that the
# model is trained and evaluated on a representative sample of the data.
#
# We select 10% of the data for testing, and the rest for training.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    shuffle=True,
    stratify=y,
    random_state=SEED,
)

print(
    f"Train set shape: {X_train.shape} with {len(X_train)} instances \
      Test set shape: {X_test.shape} with {len(X_test)} instances"
)


######################################################################
# Training the models
# ~~~~~~~~~~~~~~~~~~~~~~~~
# In this section, we train the :class:`SPDNet` and a baseline
# classifier that uses the tangent space of the covariance matrices
# followed by a linear classifier.
#
# We use the :class:`EEGClassifier` from **braindecode** to
# train the :class:`SPDNet`. The :class:`EEGClassifier` is a
# wrapper around the PyTorch model that provides a convenient interface
# for training and evaluating the model. It handles the training loop,
# validation, and testing of the model, as well as the optimization of
# the model parameters.
#
# The baseline classifier uses the tangent space of the covariance
# matrices, which is a common approach in machine learning for
# classifying covariance matrices. The tangent space is a linear
# approximation of the manifold of covariance matrices, which allows
# us to use linear classifiers on the covariance matrices.

from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit


clf = make_pipeline(
    TangentSpace(),
    LogisticRegression(random_state=SEED),
)


clf_deep = EEGClassifier(
    # The model is a PyTorch module, so we need to pass it as a callable
    # to the EEGClassifier.
    module=SPDNet,
    module__input_type="cov",
    module__n_chans=X_train.shape[1],
    module__n_outputs=len(np.unique(y_train)),
    module__subspacedim=X_train.shape[1],
    # the rest of the parameters are related to the training
    # and validation of the model.
    device="cuda" if torch.cuda.is_available() else "cpu",
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=ValidSplit(0.1, stratified=True, random_state=SEED),
    max_epochs=100,
    batch_size=64,
    lr=0.01,
    callbacks=[
        "accuracy",
        EarlyStopping(
            monitor="valid_loss",
            patience=10,
        ),
    ],
    compile=True,
)


clf.fit(X_train, y_train)
clf_deep.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred_deep = clf_deep.predict(X_test)

######################################################################
# Evaluating the models
# ~~~~~~~~~~~~~~~~~~~~~~

acc_baseline = balanced_accuracy_score(y_test, y_pred)
acc_deep = balanced_accuracy_score(y_test, y_pred_deep)

print(f"Test bal accuracy tang space: {acc_baseline * 100:.1f}%")
print(f"Test bal accuracy SPDNet: {acc_deep * 100:.1f}%")

######################################################################
# Visualizing the results
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# We can also visualize the confusion matrices for both models to get a
# better understanding of their performance. The confusion matrix shows
# the number of correct and incorrect predictions for each class.

# Increase font size for better readability
plt.rcParams.update({"font.size": 12})

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    ax=axes[0],
    display_labels=["Control", "Autism"],
    cmap="Blues",
)
axes[0].set_title("Tangent Space + Logistic Regression")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_deep,
    ax=axes[1],
    display_labels=["Control", "Autism"],
    cmap="Blues",
)
axes[1].set_title("SPDNet")

fig.suptitle("Confusion matrices on the test set", fontsize=16)
plt.show()
