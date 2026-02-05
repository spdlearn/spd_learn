"""
.. _tutorial-numerical-precision:

Numerical Precision and Stability for SPD Networks
===================================================

This tutorial demonstrates how SPD Learn's numerical stability system works
and why it's essential for reliable training. We compare models with different
eigenvalue threshold configurations to show the impact on training stability.

.. contents:: This tutorial covers:
   :local:
   :depth: 2

The tutorial is organized into seven experiments:

1. **ReEig Threshold Impact**: Compare different eigenvalue clamping thresholds
2. **Raw vs Regularized Data**: Effect of data conditioning on stability
3. **NumericalContext**: Dynamic configuration for different scenarios
4. **Performance Benchmarking**: Custom backward vs autograd speed comparison
5. **Stability Landscape**: 2D heatmap of threshold × shrinkage parameter space
6. **Eigenvalue Distribution Analysis**: Deep dive into automatic thresholds
7. **NumericalConfig Summary**: Using the summary() method for quick inspection

**Estimated runtime**: 8-15 minutes (depending on hardware)
"""

######################################################################
# Why Numerical Stability Matters
# -------------------------------
#
# SPD neural networks perform operations like:
#
# - **Matrix logarithm**: Requires strictly positive eigenvalues
# - **Eigendecomposition**: Gradients explode when eigenvalues are nearly equal
# - **Inverse square root**: Small eigenvalues cause numerical overflow
#
# SPD Learn provides a unified numerical configuration system to handle these
# challenges automatically based on data type and matrix conditioning.
#

######################################################################
# Setup and Imports
# -----------------
#

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# MOABB for EEG data
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# SPD Learn imports
from spd_learn.functional import (
    NumericalContext,
    get_epsilon,
    recommend_dtype_for_spd,
)
from spd_learn.functional.numerical import numerical_config
from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, Shrinkage, TraceNorm


warnings.filterwarnings("ignore")

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


######################################################################
# Understanding Dtype-Aware Thresholds
# ------------------------------------
#
# The numerical module provides dtype-aware thresholds that automatically
# adjust based on the precision of your data.
#

print("\n" + "=" * 70)
print("DTYPE-AWARE THRESHOLDS")
print("=" * 70)

print("\nMachine epsilon by dtype:")
for dtype, name in [(torch.float64, "float64"), (torch.float32, "float32")]:
    info = torch.finfo(dtype)
    print(f"  {name}: eps = {info.eps:.2e}")

print("\nDefault eigenvalue clamping thresholds:")
for dtype, name in [(torch.float64, "float64"), (torch.float32, "float32")]:
    eps = get_epsilon(dtype, "eigval_clamp")
    print(f"  {name}: {eps:.2e}")

print("\nWith eigval_clamp_scale=1e6 (100x larger threshold):")
with NumericalContext(eigval_clamp_scale=1e6):
    for dtype, name in [(torch.float64, "float64"), (torch.float32, "float32")]:
        eps = get_epsilon(dtype, "eigval_clamp")
        print(f"  {name}: {eps:.2e}")


######################################################################
# Load Real EEG Data
# ------------------
#

print("\n" + "=" * 70)
print("LOADING EEG DATA (BNCI2014_001)")
print("=" * 70)

dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

cache_config = dict(
    save_raw=True,
    save_epochs=True,
    save_array=True,
    use=True,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
)

subject_id = 1
X_raw, labels, meta = paradigm.get_data(
    dataset=dataset, subjects=[subject_id], cache_config=cache_config
)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

train_idx = meta.query("session == '0train'").index.to_numpy()
test_idx = meta.query("session == '1test'").index.to_numpy()

X_train_raw = torch.from_numpy(X_raw[train_idx]).float()
y_train = torch.from_numpy(y[train_idx]).long()
X_test_raw = torch.from_numpy(X_raw[test_idx]).float()
y_test = torch.from_numpy(y[test_idx]).long()

n_channels = X_train_raw.shape[1]
n_classes = len(label_encoder.classes_)

print(f"Subject {subject_id}: Train={len(y_train)}, Test={len(y_test)}")
print(f"Channels: {n_channels}, Classes: {n_classes}")


######################################################################
# Analyze Covariance Properties
# -----------------------------
#

print("\n" + "=" * 70)
print("COVARIANCE MATRIX ANALYSIS")
print("=" * 70)

cov_layer = CovLayer()

with torch.no_grad():
    X_train_cov = cov_layer(X_train_raw)
    X_test_cov = cov_layer(X_test_raw)

eigvals_train = torch.linalg.eigvalsh(X_train_cov)
cond_numbers = eigvals_train.max(dim=-1).values / eigvals_train.min(
    dim=-1
).values.clamp(min=1e-10)

print("\nRaw Covariance Statistics:")
print(
    f"  Condition number: Median={cond_numbers.median():.0f}, Max={cond_numbers.max():.0f}"
)
print(f"  Min eigenvalue: {eigvals_train.min():.2e}")
print(f"  Max eigenvalue: {eigvals_train.max():.2e}")
print(f"  Recommended dtype: {recommend_dtype_for_spd(cond_numbers.median().item())}")


######################################################################
# Prepare Regularized Data
# ------------------------
#

trace_norm = TraceNorm(epsilon=1e-5)
shrinkage = Shrinkage(n_chans=n_channels, init_shrinkage=0.3, learnable=False)

with torch.no_grad():
    X_train_reg = shrinkage(trace_norm(X_train_cov.clone()))
    X_test_reg = shrinkage(trace_norm(X_test_cov.clone()))

eigvals_reg = torch.linalg.eigvalsh(X_train_reg)
cond_reg = eigvals_reg.max(dim=-1).values / eigvals_reg.min(dim=-1).values

print("\nAfter Regularization (TraceNorm + Shrinkage):")
print(f"  Condition number: Median={cond_reg.median():.1f}, Max={cond_reg.max():.1f}")
print(f"  Min eigenvalue: {eigvals_reg.min():.4f}")


######################################################################
# Custom SPDNet with Configurable Threshold
# -----------------------------------------
#
# We create a custom SPDNet class where we can configure the ReEig threshold.
#


class SPDNetConfigurable(nn.Module):
    """SPDNet with configurable ReEig threshold."""

    def __init__(self, n_chans, n_outputs, threshold=None, use_autograd=False):
        super().__init__()
        self.bimap = BiMap(n_chans, n_chans)
        self.reeig = ReEig(threshold=threshold, autograd=use_autograd)
        self.logeig = LogEig(upper=True)
        self.len_last_layer = n_chans * (n_chans + 1) // 2
        self.classifier = nn.Linear(self.len_last_layer, n_outputs)

    def forward(self, X):
        X = self.bimap(X)
        X = self.reeig(X)
        X = self.logeig(X)
        X = self.classifier(X)
        return X


######################################################################
# Training Function
# -----------------
#


def train_model(
    model, X_train, X_test, y_train, y_test, epochs=80, lr=5e-4, verbose=True
):
    """Train SPDNet with monitoring."""
    model = model.to(DEVICE)
    X_train_d = X_train.to(DEVICE)
    X_test_d = X_test.to(DEVICE)
    y_train_d = y_train.to(DEVICE)
    y_test_d = y_test.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "nan_count": 0,
        "grad_explosions": 0,
    }

    dataset = TensorDataset(X_train_d, y_train_d)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()

            try:
                outputs = model(X_batch)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    history["nan_count"] += 1
                    continue

                loss = criterion(outputs, y_batch)

                if torch.isnan(loss) or torch.isinf(loss):
                    history["nan_count"] += 1
                    continue

                loss.backward()

                # Check gradients
                bad_grad = False
                max_grad = 0
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            bad_grad = True
                            break
                        max_grad = max(max_grad, p.grad.abs().max().item())

                if bad_grad:
                    history["nan_count"] += 1
                    continue

                if max_grad > 1e6:
                    history["grad_explosions"] += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item() * len(y_batch)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += len(y_batch)

            except RuntimeError:
                history["nan_count"] += 1
                continue

        scheduler.step()

        if total > 0:
            history["train_loss"].append(epoch_loss / total)
            history["train_acc"].append(100.0 * correct / total)
        else:
            history["train_loss"].append(float("nan"))
            history["train_acc"].append(0.0)

        model.eval()
        with torch.no_grad():
            try:
                test_out = model(X_test_d)
                if not (torch.isnan(test_out).any() or torch.isinf(test_out).any()):
                    _, pred = test_out.max(1)
                    test_acc = 100.0 * pred.eq(y_test_d).sum().item() / len(y_test_d)
                else:
                    test_acc = 0.0
            except RuntimeError:
                test_acc = 0.0
        history["test_acc"].append(test_acc)

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"  Epoch {epoch+1}: Loss={history['train_loss'][-1]:.4f}, "
                f"Train={history['train_acc'][-1]:.1f}%, Test={test_acc:.1f}%"
            )

    history["best_test_acc"] = max(history["test_acc"]) if history["test_acc"] else 0.0
    return history


######################################################################
# EXPERIMENT 1: Impact of ReEig Threshold
# ---------------------------------------
#
# Compare different eigenvalue clamping thresholds on regularized data.
#

print("\n" + "=" * 70)
print("EXPERIMENT 1: IMPACT OF REEIG THRESHOLD")
print("=" * 70)
print("\nComparing different eigenvalue thresholds on regularized data...")

threshold_results = {}

thresholds = [
    ("threshold=1e-10", 1e-10),
    ("threshold=1e-6", 1e-6),
    ("threshold=1e-4", 1e-4),
    ("threshold=1e-2", 1e-2),
    ("threshold=None (auto)", None),  # Uses numerical config based on dtype
]

for name, threshold in thresholds:
    print(f"\n--- {name} ---")
    model = SPDNetConfigurable(n_channels, n_classes, threshold=threshold)
    history = train_model(model, X_train_reg, X_test_reg, y_train, y_test, epochs=60)
    threshold_results[name] = history
    status = (
        "STABLE"
        if history["nan_count"] == 0
        else f"UNSTABLE ({history['nan_count']} NaN)"
    )
    print(
        f"Best: {history['best_test_acc']:.1f}%, Status: {status}, Grad explosions: {history['grad_explosions']}"
    )

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

colors = {
    "threshold=1e-10": "#c0392b",
    "threshold=1e-6": "#e74c3c",
    "threshold=1e-4": "#3498db",
    "threshold=1e-2": "#f39c12",
    "threshold=None (auto)": "#27ae60",
}

# Training loss
ax1 = axes[0]
for name, hist in threshold_results.items():
    # Extract short label: "threshold=1e-10" -> "1e-10", "threshold=None (auto)" -> "auto"
    short_label = name.replace("threshold=", "").replace(" (auto)", "")
    if short_label == "None":
        short_label = "auto"
    ax1.plot(hist["train_loss"], label=short_label, color=colors[name], linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Training Loss", fontsize=12)
ax1.set_title("Training Loss", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Test accuracy
ax2 = axes[1]
for name, hist in threshold_results.items():
    short_label = name.replace("threshold=", "").replace(" (auto)", "")
    if short_label == "None":
        short_label = "auto"
    ax2.plot(hist["test_acc"], label=short_label, color=colors[name], linewidth=2)
ax2.axhline(y=25, color="gray", linestyle="--", alpha=0.5)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Test Accuracy (%)", fontsize=12)
ax2.set_title("Test Accuracy", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 80])

# Summary bars
ax3 = axes[2]
names = list(threshold_results.keys())
accs = [threshold_results[n]["best_test_acc"] for n in names]
nans = [threshold_results[n]["nan_count"] for n in names]
color_list = [colors[n] for n in names]

bars = ax3.barh(range(len(names)), accs, color=color_list, edgecolor="black")
ax3.set_yticks(range(len(names)))
# Create readable y-tick labels
y_labels = []
for n in names:
    label = n.replace("threshold=", "")
    y_labels.append(label)
ax3.set_yticklabels(y_labels, fontsize=9)
ax3.set_xlabel("Best Test Accuracy (%)", fontsize=12)
ax3.set_title("Best Performance", fontsize=13, fontweight="bold")

for i, (acc, nan) in enumerate(zip(accs, nans)):
    label = f"{acc:.1f}%" if nan == 0 else f"{acc:.1f}% (NaN)"
    color = "green" if nan == 0 else "red"
    ax3.text(acc + 1, i, label, va="center", fontsize=9, fontweight="bold", color=color)

ax3.set_xlim([0, 80])
ax3.grid(True, alpha=0.3, axis="x")

plt.suptitle(
    "Experiment 1: Impact of ReEig Threshold on Training",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.show()


######################################################################
# EXPERIMENT 2: Raw vs Regularized Data
# -------------------------------------
#
# Compare training on raw (ill-conditioned) vs regularized (well-conditioned) data.
#

print("\n" + "=" * 70)
print("EXPERIMENT 2: RAW VS REGULARIZED DATA")
print("=" * 70)

data_results = {}

# Raw data
print("\n--- Raw data (κ ≈ 12000) ---")
model_raw = SPDNetConfigurable(n_channels, n_classes, threshold=1e-4)
history_raw = train_model(
    model_raw, X_train_cov, X_test_cov, y_train, y_test, epochs=80
)
data_results["Raw (κ≈12k)"] = history_raw

# Regularized data
print("\n--- Regularized data (κ ≈ 13) ---")
model_reg = SPDNetConfigurable(n_channels, n_classes, threshold=1e-4)
history_reg = train_model(
    model_reg, X_train_reg, X_test_reg, y_train, y_test, epochs=80
)
data_results["Regularized (κ≈13)"] = history_reg

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

data_colors = {"Raw (κ≈12k)": "#e74c3c", "Regularized (κ≈13)": "#27ae60"}

ax1 = axes[0]
for name, hist in data_results.items():
    ax1.plot(hist["train_loss"], label=name, color=data_colors[name], linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Training Loss", fontsize=12)
ax1.set_title("Training Loss", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for name, hist in data_results.items():
    ax2.plot(
        hist["test_acc"],
        label=f"{name} (best: {hist['best_test_acc']:.1f}%)",
        color=data_colors[name],
        linewidth=2,
    )
ax2.axhline(y=25, color="gray", linestyle="--", alpha=0.5, label="Chance")
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Test Accuracy (%)", fontsize=12)
ax2.set_title("Test Accuracy", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

plt.suptitle(
    "Experiment 2: Raw vs Regularized Data", fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.show()


######################################################################
# EXPERIMENT 3: NumericalContext for Dynamic Configuration
# --------------------------------------------------------
#

print("\n" + "=" * 70)
print("EXPERIMENT 3: NumericalContext DEMONSTRATION")
print("=" * 70)

print("\n1. Default thresholds:")
print(f"   eigval_clamp: {get_epsilon(torch.float32, 'eigval_clamp'):.2e}")
print(f"   eigval_log:   {get_epsilon(torch.float32, 'eigval_log'):.2e}")

print("\n2. With eigval_clamp_scale=1e6, eigval_log_scale=1e4:")
with NumericalContext(eigval_clamp_scale=1e6, eigval_log_scale=1e4):
    print(f"   eigval_clamp: {get_epsilon(torch.float32, 'eigval_clamp'):.2e}")
    print(f"   eigval_log:   {get_epsilon(torch.float32, 'eigval_log'):.2e}")

print("\n3. After context (restored):")
print(f"   eigval_clamp: {get_epsilon(torch.float32, 'eigval_clamp'):.2e}")
print(f"   eigval_log:   {get_epsilon(torch.float32, 'eigval_log'):.2e}")

# Compare dtype-aware model with different contexts
print("\n4. Training with different NumericalContext configurations:")

context_results = {}

# Default context
print("\n--- eigval_clamp_scale=1e4 (default) ---")
model_default = SPDNetConfigurable(n_channels, n_classes, threshold=None)
history_default = train_model(
    model_default, X_train_reg, X_test_reg, y_train, y_test, epochs=60
)
context_results["eigval_clamp_scale=1e4"] = history_default

# Larger scale context (more aggressive clamping)
print("\n--- eigval_clamp_scale=1e6 (larger threshold) ---")
with NumericalContext(eigval_clamp_scale=1e6, eigval_log_scale=1e4):
    model_large_scale = SPDNetConfigurable(n_channels, n_classes, threshold=None)
    history_large_scale = train_model(
        model_large_scale, X_train_reg, X_test_reg, y_train, y_test, epochs=60
    )
context_results["eigval_clamp_scale=1e6"] = history_large_scale

# Visualize
fig, ax = plt.subplots(figsize=(10, 4))

ctx_colors = {"eigval_clamp_scale=1e4": "#3498db", "eigval_clamp_scale=1e6": "#27ae60"}

for name, hist in context_results.items():
    ax.plot(
        hist["test_acc"],
        label=f"{name} (best: {hist['best_test_acc']:.1f}%)",
        color=ctx_colors[name],
        linewidth=2,
    )

ax.axhline(y=25, color="gray", linestyle="--", alpha=0.5, label="Chance")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title(
    "Test Accuracy with Different NumericalContext Configurations",
    fontsize=13,
    fontweight="bold",
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 80])

plt.tight_layout()
plt.show()


######################################################################
# EXPERIMENT 4: Performance Benchmarking
# --------------------------------------
#
# Compare performance between custom backward pass and autograd,
# across different dtypes. SPD Learn uses custom backward passes for
# eigendecomposition operations which can be faster than autograd.
# Larger matrix sizes are more numerically demanding (eigenvalues are
# more densely packed), so 128 and 256 highlight stability-sensitive regimes.
#

print("\n" + "=" * 70)
print("EXPERIMENT 4: PERFORMANCE BENCHMARKING")
print("=" * 70)
print("Note: Larger matrices (128/256) are more sensitive to numerical issues.")
print("      If you see NaNs or unstable gradients, increase eigval_clamp_scale.")


def benchmark(func, *args, n_runs=50):
    """Benchmark function returning mean time in milliseconds."""
    # Warmup runs
    for _ in range(3):
        func(*args)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        func(*args)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # Convert to milliseconds


# Matrix sizes to test (include larger sizes for stability stress)
matrix_sizes = [8, 16, 22, 32, 64, 128, 256]

# Store results
benchmark_results = {
    "sizes": matrix_sizes,
    "custom_f32": [],
    "autograd_f32": [],
    "custom_f64": [],
    "autograd_f64": [],
}

print("\nBenchmarking forward+backward pass times...")
print("(This may take a minute)")

for size in matrix_sizes:
    print(f"\n  Matrix size: {size}x{size}")

    for dtype, dtype_name in [(torch.float32, "f32"), (torch.float64, "f64")]:
        # Create test data
        X_bench = torch.randn(32, size, size, dtype=dtype, device=DEVICE)
        X_bench = X_bench @ X_bench.transpose(-1, -2) + 0.1 * torch.eye(
            size, dtype=dtype, device=DEVICE
        )
        X_bench.requires_grad_(True)

        # Custom backward
        reeig_custom = ReEig(threshold=1e-4, autograd=False).to(DEVICE)

        def forward_custom():
            if X_bench.grad is not None:
                X_bench.grad.zero_()
            out = reeig_custom(X_bench)
            loss = out.sum()
            loss.backward()

        time_custom = benchmark(forward_custom, n_runs=30)
        benchmark_results[f"custom_{dtype_name}"].append(time_custom)
        print(f"    Custom ({dtype_name}): {time_custom:.3f} ms")

        # Autograd
        reeig_auto = ReEig(threshold=1e-4, autograd=True).to(DEVICE)

        def forward_autograd():
            if X_bench.grad is not None:
                X_bench.grad.zero_()
            out = reeig_auto(X_bench)
            loss = out.sum()
            loss.backward()

        time_autograd = benchmark(forward_autograd, n_runs=30)
        benchmark_results[f"autograd_{dtype_name}"].append(time_autograd)
        print(f"    Autograd ({dtype_name}): {time_autograd:.3f} ms")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Log-scale timing comparison
ax1 = axes[0]
ax1.semilogy(
    matrix_sizes,
    benchmark_results["custom_f32"],
    "o-",
    color="#3498db",
    linewidth=2,
    markersize=8,
    label="Custom (float32)",
)
ax1.semilogy(
    matrix_sizes,
    benchmark_results["autograd_f32"],
    "s--",
    color="#e74c3c",
    linewidth=2,
    markersize=8,
    label="Autograd (float32)",
)
ax1.semilogy(
    matrix_sizes,
    benchmark_results["custom_f64"],
    "o-",
    color="#2980b9",
    linewidth=2,
    markersize=8,
    alpha=0.6,
    label="Custom (float64)",
)
ax1.semilogy(
    matrix_sizes,
    benchmark_results["autograd_f64"],
    "s--",
    color="#c0392b",
    linewidth=2,
    markersize=8,
    alpha=0.6,
    label="Autograd (float64)",
)
ax1.set_xlabel("Matrix Size", fontsize=12)
ax1.set_ylabel("Time (ms, log scale)", fontsize=12)
ax1.set_title("Forward+Backward Pass Timing", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which="both")
ax1.set_xticks(matrix_sizes)

# Right panel: Speedup ratio bar chart
ax2 = axes[1]
speedup_f32 = [
    a / c
    for a, c in zip(benchmark_results["autograd_f32"], benchmark_results["custom_f32"])
]
speedup_f64 = [
    a / c
    for a, c in zip(benchmark_results["autograd_f64"], benchmark_results["custom_f64"])
]

x = np.arange(len(matrix_sizes))
width = 0.35

bars1 = ax2.bar(
    x - width / 2,
    speedup_f32,
    width,
    label="float32",
    color="#3498db",
    edgecolor="black",
)
bars2 = ax2.bar(
    x + width / 2,
    speedup_f64,
    width,
    label="float64",
    color="#2980b9",
    edgecolor="black",
)
ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Break-even")
ax2.set_xlabel("Matrix Size", fontsize=12)
ax2.set_ylabel("Speedup (Autograd / Custom)", fontsize=12)
ax2.set_title("Custom Backward Speedup", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(matrix_sizes)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.05,
        f"{height:.1f}x",
        ha="center",
        fontsize=9,
    )
for bar in bars2:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.05,
        f"{height:.1f}x",
        ha="center",
        fontsize=9,
    )

plt.suptitle(
    "Experiment 4: Custom vs Autograd Performance",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.show()


######################################################################
# EXPERIMENT 5: Stability Landscape
# ---------------------------------
#
# Visualize the stability/accuracy landscape across different
# threshold and shrinkage parameter combinations using a 2D heatmap.
#

print("\n" + "=" * 70)
print("EXPERIMENT 5: STABILITY LANDSCAPE")
print("=" * 70)
print("\nMapping stability across threshold × shrinkage parameter space...")
print("(This will take several minutes - 64 configurations × 20 epochs each)")

# Parameter grids (8x8 = 64 combinations)
threshold_values = np.logspace(-10, -1, 8)  # 1e-10 to 1e-1
shrinkage_values = np.linspace(0.0, 0.8, 8)  # 0% to 80%

# Results storage
accuracy_grid = np.zeros((len(shrinkage_values), len(threshold_values)))
stability_grid = np.zeros((len(shrinkage_values), len(threshold_values)))
nan_grid = np.zeros((len(shrinkage_values), len(threshold_values)))


def quick_train(model, X_tr, X_te, y_tr, y_te, epochs=20):
    """Quick training for landscape exploration."""
    model = model.to(DEVICE)
    X_tr_d, X_te_d = X_tr.to(DEVICE), X_te.to(DEVICE)
    y_tr_d, y_te_d = y_tr.to(DEVICE), y_te.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    nan_count = 0
    best_acc = 0.0

    dataset = TensorDataset(X_tr_d, y_tr_d)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            try:
                outputs = model(X_batch)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    nan_count += 1
                    continue
                loss = criterion(outputs, y_batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    continue
                loss.backward()
                # Check gradients
                bad_grad = False
                for p in model.parameters():
                    if p.grad is not None and (
                        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    ):
                        bad_grad = True
                        break
                if bad_grad:
                    nan_count += 1
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            except RuntimeError:
                nan_count += 1
                continue

        # Evaluate
        model.eval()
        with torch.no_grad():
            try:
                test_out = model(X_te_d)
                if not (torch.isnan(test_out).any() or torch.isinf(test_out).any()):
                    _, pred = test_out.max(1)
                    acc = 100.0 * pred.eq(y_te_d).sum().item() / len(y_te_d)
                    best_acc = max(best_acc, acc)
            except RuntimeError:
                pass

    is_stable = nan_count == 0
    return best_acc, is_stable, nan_count


total_configs = len(shrinkage_values) * len(threshold_values)
config_num = 0

for i, shrink in enumerate(shrinkage_values):
    for j, thresh in enumerate(threshold_values):
        config_num += 1
        if config_num % 16 == 0 or config_num == 1:
            print(f"  Progress: {config_num}/{total_configs} configurations...")

        # Apply shrinkage to raw covariance
        if shrink > 0:
            shrink_layer = Shrinkage(
                n_chans=n_channels, init_shrinkage=shrink, learnable=False
            )
            with torch.no_grad():
                X_tr_shrink = shrink_layer(trace_norm(X_train_cov.clone()))
                X_te_shrink = shrink_layer(trace_norm(X_test_cov.clone()))
        else:
            X_tr_shrink = trace_norm(X_train_cov.clone())
            X_te_shrink = trace_norm(X_test_cov.clone())

        # Create model with specific threshold
        model = SPDNetConfigurable(n_channels, n_classes, threshold=thresh)
        acc, stable, nans = quick_train(
            model, X_tr_shrink, X_te_shrink, y_train, y_test, epochs=20
        )

        accuracy_grid[i, j] = acc
        stability_grid[i, j] = 1.0 if stable else 0.0
        nan_grid[i, j] = nans

print("  Landscape mapping complete!")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Common axis labels
log_thresh = np.log10(threshold_values)
extent = [
    log_thresh.min(),
    log_thresh.max(),
    shrinkage_values.min(),
    shrinkage_values.max(),
]

# Panel 1: Accuracy heatmap
ax1 = axes[0]
im1 = ax1.pcolormesh(
    log_thresh,
    shrinkage_values,
    accuracy_grid,
    cmap="RdYlGn",
    shading="auto",
    vmin=25,
    vmax=70,
)
cs1 = ax1.contour(
    log_thresh,
    shrinkage_values,
    accuracy_grid,
    levels=[30, 40, 50, 60],
    colors="black",
    linewidths=0.5,
    alpha=0.5,
)
ax1.clabel(cs1, inline=True, fontsize=8, fmt="%.0f%%")
ax1.set_xlabel("log₁₀(Threshold)", fontsize=12)
ax1.set_ylabel("Shrinkage", fontsize=12)
ax1.set_title("Test Accuracy (%)", fontsize=13, fontweight="bold")
plt.colorbar(im1, ax=ax1, label="Accuracy (%)")

# Panel 2: Stability map
ax2 = axes[1]
im2 = ax2.pcolormesh(
    log_thresh,
    shrinkage_values,
    stability_grid,
    cmap="RdYlGn",
    shading="auto",
    vmin=0,
    vmax=1,
)
ax2.set_xlabel("log₁₀(Threshold)", fontsize=12)
ax2.set_ylabel("Shrinkage", fontsize=12)
ax2.set_title("Stability (Green=Stable)", fontsize=13, fontweight="bold")
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_ticks([0, 1])
cbar2.set_ticklabels(["Unstable", "Stable"])

# Panel 3: NaN event count
ax3 = axes[2]
im3 = ax3.pcolormesh(
    log_thresh,
    shrinkage_values,
    nan_grid,
    cmap="Reds",
    shading="auto",
)
cs3 = ax3.contour(
    log_thresh,
    shrinkage_values,
    nan_grid,
    levels=[5, 10, 20],
    colors="black",
    linewidths=0.5,
    alpha=0.7,
)
ax3.clabel(cs3, inline=True, fontsize=8, fmt="%.0f")
ax3.set_xlabel("log₁₀(Threshold)", fontsize=12)
ax3.set_ylabel("Shrinkage", fontsize=12)
ax3.set_title("NaN Event Count", fontsize=13, fontweight="bold")
plt.colorbar(im3, ax=ax3, label="NaN Events")

plt.suptitle(
    "Experiment 5: Stability Landscape (Threshold × Shrinkage)",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.show()

# Print key findings
print("\nKey observations from stability landscape:")
best_idx = np.unravel_index(np.argmax(accuracy_grid), accuracy_grid.shape)
print(
    f"  Best accuracy: {accuracy_grid[best_idx]:.1f}% at "
    f"shrinkage={shrinkage_values[best_idx[0]]:.2f}, "
    f"threshold={threshold_values[best_idx[1]]:.1e}"
)
stable_mask = stability_grid == 1.0
stable_accs = accuracy_grid[stable_mask]
if len(stable_accs) > 0:
    print(f"  Stable configurations: {stable_mask.sum()}/{total_configs}")
    print(f"  Best stable accuracy: {stable_accs.max():.1f}%")


######################################################################
# EXPERIMENT 6: Eigenvalue Distribution Analysis
# ----------------------------------------------
#
# Deep dive into the automatic threshold system by visualizing
# eigenvalue distributions with threshold overlays showing "danger zones."
#

print("\n" + "=" * 70)
print("EXPERIMENT 6: EIGENVALUE DISTRIBUTION ANALYSIS")
print("=" * 70)

# Collect eigenvalues
eigvals_raw = torch.linalg.eigvalsh(X_train_cov).flatten().numpy()
eigvals_reg = torch.linalg.eigvalsh(X_train_reg).flatten().numpy()

# Get thresholds for different dtypes
thresholds_by_dtype = {}
for dtype, name in [
    (torch.float16, "float16"),
    (torch.float32, "float32"),
    (torch.float64, "float64"),
]:
    thresholds_by_dtype[name] = get_epsilon(dtype, "eigval_clamp")

print("\nAutomatic thresholds by dtype:")
for name, thresh in thresholds_by_dtype.items():
    print(f"  {name}: {thresh:.2e}")

# Count at-risk eigenvalues
print("\nEigenvalues at risk (below threshold):")
for dtype_name, thresh in thresholds_by_dtype.items():
    raw_at_risk = (eigvals_raw < thresh).sum()
    reg_at_risk = (eigvals_reg < thresh).sum()
    print(
        f"  {dtype_name}: Raw={raw_at_risk}/{len(eigvals_raw)}, "
        f"Regularized={reg_at_risk}/{len(eigvals_reg)}"
    )

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Raw eigenvalue histogram with thresholds
ax1 = axes[0, 0]
ax1.hist(
    eigvals_raw,
    bins=50,
    color="#3498db",
    alpha=0.7,
    edgecolor="black",
    label="Raw eigenvalues",
)
colors_thresh = {"float16": "#e74c3c", "float32": "#f39c12", "float64": "#27ae60"}
for dtype_name, thresh in thresholds_by_dtype.items():
    ax1.axvline(
        x=thresh,
        color=colors_thresh[dtype_name],
        linestyle="--",
        linewidth=2,
        label=f"{dtype_name} threshold",
    )
# Shade danger zone
ax1.axvspan(
    eigvals_raw.min(),
    thresholds_by_dtype["float32"],
    alpha=0.2,
    color="red",
    label="Danger zone (float32)",
)
ax1.set_xlabel("Eigenvalue", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_title("Raw Covariance Eigenvalues", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.set_xscale("log")
ax1.grid(True, alpha=0.3)

# Top-right: Regularized eigenvalue histogram
ax2 = axes[0, 1]
ax2.hist(
    eigvals_reg,
    bins=50,
    color="#27ae60",
    alpha=0.7,
    edgecolor="black",
    label="Regularized eigenvalues",
)
for dtype_name, thresh in thresholds_by_dtype.items():
    ax2.axvline(
        x=thresh,
        color=colors_thresh[dtype_name],
        linestyle="--",
        linewidth=2,
        label=f"{dtype_name} threshold",
    )
ax2.set_xlabel("Eigenvalue", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("Regularized Covariance Eigenvalues", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xscale("log")
ax2.grid(True, alpha=0.3)

# Bottom-left: CDF plot
ax3 = axes[1, 0]
sorted_raw = np.sort(eigvals_raw)
sorted_reg = np.sort(eigvals_reg)
cdf_raw = np.arange(1, len(sorted_raw) + 1) / len(sorted_raw)
cdf_reg = np.arange(1, len(sorted_reg) + 1) / len(sorted_reg)

ax3.plot(sorted_raw, cdf_raw, color="#3498db", linewidth=2, label="Raw")
ax3.plot(sorted_reg, cdf_reg, color="#27ae60", linewidth=2, label="Regularized")
for dtype_name, thresh in thresholds_by_dtype.items():
    ax3.axvline(x=thresh, color=colors_thresh[dtype_name], linestyle="--", linewidth=2)
    # Add marker at CDF value
    cdf_val_raw = (sorted_raw < thresh).sum() / len(sorted_raw)
    ax3.plot(thresh, cdf_val_raw, "o", color=colors_thresh[dtype_name], markersize=8)
ax3.set_xlabel("Eigenvalue", fontsize=12)
ax3.set_ylabel("CDF", fontsize=12)
ax3.set_title("Cumulative Distribution Function", fontsize=13, fontweight="bold")
ax3.legend(fontsize=10)
ax3.set_xscale("log")
ax3.grid(True, alpha=0.3)

# Bottom-right: At-risk eigenvalue counts
ax4 = axes[1, 1]
dtypes = list(thresholds_by_dtype.keys())
raw_counts = [(eigvals_raw < thresholds_by_dtype[d]).sum() for d in dtypes]
reg_counts = [(eigvals_reg < thresholds_by_dtype[d]).sum() for d in dtypes]

x = np.arange(len(dtypes))
width = 0.35
bars1 = ax4.bar(
    x - width / 2, raw_counts, width, label="Raw", color="#e74c3c", edgecolor="black"
)
bars2 = ax4.bar(
    x + width / 2,
    reg_counts,
    width,
    label="Regularized",
    color="#27ae60",
    edgecolor="black",
)
ax4.set_xlabel("Data Type", fontsize=12)
ax4.set_ylabel("At-Risk Eigenvalues", fontsize=12)
ax4.set_title("Eigenvalues Below Threshold", fontsize=13, fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(dtypes)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis="y")

# Add count labels
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{int(height)}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{int(height)}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

plt.suptitle(
    "Experiment 6: Eigenvalue Distribution & Threshold Analysis",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.show()


######################################################################
# EXPERIMENT 7: NumericalConfig Summary
# -------------------------------------
#
# Demonstrate the new summary() method for inspecting the complete
# numerical configuration at a glance.
#

print("\n" + "=" * 70)
print("EXPERIMENT 7: NUMERICAL CONFIG SUMMARY")
print("=" * 70)

print("\n--- Default scales, dtype=torch.float32 ---")
print(numerical_config.summary(torch.float32))

print("\n--- Default scales, dtype=torch.float64 ---")
print(numerical_config.summary(torch.float64))

print(
    "\n--- Modified: eigval_clamp_scale=1e6, eigval_log_scale=1e4, dtype=torch.float32 ---"
)
with NumericalContext(eigval_clamp_scale=1e6, eigval_log_scale=1e4):
    from spd_learn.functional.numerical import numerical_config as nc

    print(nc.summary(torch.float32))


######################################################################
# Summary and Recommendations
# ---------------------------
#

print("\n" + "=" * 70)
print("SUMMARY: NUMERICAL STABILITY RECOMMENDATIONS")
print("=" * 70)

print("""
+------------------------------------------------------------------------+
|                   NUMERICAL CONFIGURATION GUIDE                         |
+------------------------------------------------------------------------+
| SCENARIO                  | RECOMMENDED SETTINGS                        |
+------------------------------------------------------------------------+
| Well-conditioned (κ<100)  | - eigval_clamp_scale=1e4 (default)         |
|                           | - Regularization optional                   |
+------------------------------------------------------------------------+
| Typical EEG (κ~1000)      | - Light regularization (shrinkage~0.1)     |
|                           | - Consider eigval_clamp_scale=1e5          |
+------------------------------------------------------------------------+
| Ill-conditioned (κ>10k)   | - ReEig threshold protects automatically   |
|                           | - Consider eigval_clamp_scale=1e6          |
|                           | - Light regularization may help            |
+------------------------------------------------------------------------+
| Mixed precision (fp16)    | - Requires eigval_clamp_scale=1e6-1e8      |
|                           | - Strong regularization required           |
+------------------------------------------------------------------------+

IMPORTANT INSIGHT:
The built-in ReEig layer with its eigenvalue threshold provides automatic
protection against numerical instabilities. Raw EEG covariances with high
condition numbers (κ≈12000) can still be processed stably AND often retain
more discriminative information than heavily regularized data.

Over-regularization can hurt performance! The key is finding the right
balance for your specific task.

KEY FINDINGS:
""")

print("1. THRESHOLD IMPACT:")
for name, hist in threshold_results.items():
    status = (
        "STABLE" if hist["nan_count"] == 0 else f"UNSTABLE ({hist['nan_count']} NaN)"
    )
    print(f"   {name}: {hist['best_test_acc']:.1f}% ({status})")

print("\n2. DATA CONDITIONING:")
print(f"   Raw data (κ≈12k): {history_raw['best_test_acc']:.1f}%")
print(f"   Regularized (κ≈13): {history_reg['best_test_acc']:.1f}%")

print("\n3. NUMERICAL CONTEXT (dtype-aware threshold):")
for name, hist in context_results.items():
    print(f"   {name}: {hist['best_test_acc']:.1f}%")

print("\n4. PERFORMANCE (Custom backward vs Autograd):")
avg_speedup = np.mean(speedup_f32)
print(f"   Average speedup (float32): {avg_speedup:.1f}x")

print("\n5. STABILITY LANDSCAPE:")
print(f"   Stable configurations: {stable_mask.sum()}/{total_configs}")
if len(stable_accs) > 0:
    print(f"   Best stable accuracy: {stable_accs.max():.1f}%")

print("""
RECOMMENDED WORKFLOW:

1. Check data conditioning:
   >>> eigvals = torch.linalg.eigvalsh(covariances)
   >>> condition = eigvals.max() / eigvals.min()
   >>> print(f"Condition number: {condition.median():.0f}")

2. Inspect your numerical configuration:
   >>> from spd_learn.functional.numerical import numerical_config
   >>> print(numerical_config.summary(torch.float32))

3. Try training without regularization first:
   - The ReEig layer provides automatic protection
   - Raw data often contains more discriminative information

4. If training is unstable, apply light regularization:
   >>> from spd_learn.modules import TraceNorm, Shrinkage
   >>> trace_norm = TraceNorm(epsilon=1e-5)
   >>> shrinkage = Shrinkage(n_chans=22, init_shrinkage=0.1)  # Light shrinkage
   >>> regularized = shrinkage(trace_norm(covariances))

5. Use dtype-aware thresholds for flexibility:
   >>> layer = ReEig(threshold=None)  # Auto-adjusts based on dtype

6. Use NumericalContext for temporary configuration:
   >>> from spd_learn.functional import NumericalContext
   >>> with NumericalContext(eigval_clamp_scale=1e6):
   ...     outputs = model(challenging_data)

7. Consider performance trade-offs:
   - Custom backward passes are typically faster than autograd
   - Use autograd=True only when debugging gradient issues
""")


######################################################################
# Final Summary Plot
# ------------------
#

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Threshold comparison
ax1 = axes[0, 0]
names = list(threshold_results.keys())
accs = [threshold_results[n]["best_test_acc"] for n in names]
color_list = [colors[n] for n in names]
# Create short labels for x-axis
short_labels = []
for n in names:
    short = n.replace("threshold=", "").replace(" (auto)", "")
    if short == "None":
        short = "auto"
    short_labels.append(short)
bars = ax1.bar(range(len(names)), accs, color=color_list, edgecolor="black")
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(short_labels, fontsize=9)
ax1.set_ylabel("Best Accuracy (%)", fontsize=12)
ax1.set_title("ReEig Threshold Impact", fontsize=13, fontweight="bold")
ax1.set_ylim([0, 80])
ax1.axhline(y=25, color="gray", linestyle="--", alpha=0.5)
ax1.grid(True, alpha=0.3, axis="y")

# Plot 2: Data conditioning
ax2 = axes[0, 1]
data_names = list(data_results.keys())
data_accs = [data_results[n]["best_test_acc"] for n in data_names]
data_color_list = [data_colors[n] for n in data_names]
bars = ax2.bar(data_names, data_accs, color=data_color_list, edgecolor="black")
ax2.set_ylabel("Best Accuracy (%)", fontsize=12)
ax2.set_title("Data Conditioning Impact", fontsize=13, fontweight="bold")
ax2.set_ylim([0, 100])
ax2.axhline(y=25, color="gray", linestyle="--", alpha=0.5)
for bar, acc in zip(bars, data_accs):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{acc:.1f}%",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: Performance comparison
ax3 = axes[1, 0]
x = np.arange(len(matrix_sizes))
width = 0.35
bars1 = ax3.bar(
    x - width / 2,
    benchmark_results["custom_f32"],
    width,
    label="Custom",
    color="#3498db",
    edgecolor="black",
)
bars2 = ax3.bar(
    x + width / 2,
    benchmark_results["autograd_f32"],
    width,
    label="Autograd",
    color="#e74c3c",
    edgecolor="black",
)
ax3.set_xlabel("Matrix Size", fontsize=12)
ax3.set_ylabel("Time (ms)", fontsize=12)
ax3.set_title("Forward+Backward Performance (float32)", fontsize=13, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(matrix_sizes)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: Stability landscape summary (accuracy heatmap)
ax4 = axes[1, 1]
im4 = ax4.pcolormesh(
    np.log10(threshold_values),
    shrinkage_values,
    accuracy_grid,
    cmap="RdYlGn",
    shading="auto",
    vmin=25,
    vmax=70,
)
# Mark best configuration
best_idx = np.unravel_index(np.argmax(accuracy_grid), accuracy_grid.shape)
ax4.plot(
    np.log10(threshold_values[best_idx[1]]),
    shrinkage_values[best_idx[0]],
    "k*",
    markersize=15,
    label=f"Best: {accuracy_grid[best_idx]:.1f}%",
)
ax4.set_xlabel("log₁₀(Threshold)", fontsize=12)
ax4.set_ylabel("Shrinkage", fontsize=12)
ax4.set_title("Stability Landscape (Accuracy)", fontsize=13, fontweight="bold")
ax4.legend(loc="upper right", fontsize=10)
plt.colorbar(im4, ax=ax4, label="Accuracy (%)")

plt.suptitle(
    "Summary: Numerical Stability in SPD Networks",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)
