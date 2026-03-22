"""
.. _liebn-batch-normalization:

Reproducing LieBN Paper Results (Table 4)
==========================================

This example reproduces the SPDNet experiments from Table 4 of
Chen et al., "A Lie Group Approach to Riemannian Batch
Normalization", ICLR 2024 :cite:p:`chen2024liebn`.

We compare batch normalization strategies on HDM05 (7 configs), Radar
(6 configs), and AFEW (7 configs) datasets. HDM05 and Radar follow the
paper's evaluation protocol (10 independent random-split runs with
batch-mean accuracy). AFEW uses a fixed train/val split (10 runs varying
only model initialization).

**Configurations benchmarked:**

- **SPDNet**: No batch normalization
- **SPDNetBN**: Riemannian BN (Brooks et al. + variance normalization)
- **LieBN-AIM**: LieBN under the Affine-Invariant Metric (theta=1, 1.5)
- **LieBN-LEM**: LieBN under the Log-Euclidean Metric
- **LieBN-LCM**: LieBN under the Log-Cholesky Metric (theta=1, 0.5, -0.5)

.. note::

   New to SPD batch normalization? Start with the
   :ref:`tutorial-batch-normalization` tutorial for an introduction, or
   see :ref:`howto-add-batchnorm` for a quick integration guide.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Setup and Imports
# -----------------
#

import json
import os
import random
import tarfile
import tempfile
import time
import urllib.request
import warnings
import zipfile

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset

from spd_learn.functional import ensure_sym
from spd_learn.modules import (
    BiMap,
    LogEig,
    ReEig,
    SPDBatchNormLie,
    SPDBatchNormMeanVar,
)


# Suppress noisy warnings from matplotlib and torch internals;
# keep UserWarning and RuntimeWarning visible for diagnostic signals.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="matplotlib")


def set_reproducibility(seed=1024):
    """Set random seeds and enable deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


GLOBAL_SEED = 1024
set_reproducibility(GLOBAL_SEED)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

######################################################################
# SPDNet Architecture & Training Setup
# -------------------------------------
#
# We build a multi-layer SPDNet following the reference architecture:
#
# - Intermediate layers: ``BiMap -> [BN] -> ReEig``
# - Final layer: ``BiMap -> [BN]`` (no ReEig)
# - Classifier: ``LogEig -> flatten -> Linear``
#


def make_bn(n, bn_type, bn_kwargs):
    """Create a batch normalization layer."""
    if bn_type == "SPDBN":
        return SPDBatchNormMeanVar(n, momentum=bn_kwargs.get("momentum", 0.1))
    elif bn_type == "LieBN":
        return SPDBatchNormLie(n, **bn_kwargs)
    else:
        raise ValueError(f"Unknown bn_type: {bn_type}")


class SPDNetModel(nn.Module):
    """Multi-layer SPDNet with optional batch normalization.

    Parameters
    ----------
    dims : list of int
        Sequence of SPD matrix dimensions, e.g. [93, 30].
    n_classes : int
        Number of output classes.
    bn_type : str or None
        None (no BN), 'SPDBN', or 'LieBN'.
    bn_kwargs : dict or None
        Keyword arguments for the BN layer.
    """

    def __init__(self, dims, n_classes, bn_type=None, bn_kwargs=None):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(BiMap(dims[i], dims[i + 1]))
            if bn_type is not None:
                layers.append(make_bn(dims[i + 1], bn_type, bn_kwargs or {}))
            if i < len(dims) - 2:
                layers.append(ReEig())
        self.features = nn.Sequential(*layers)
        n_out = dims[-1]
        # upper=False uses full n^2 features (matches reference's dims[-1]**2).
        self.logeig = LogEig(upper=False, flatten=True)
        self.classifier = nn.Linear(n_out**2, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.logeig(x)
        return self.classifier(x)


######################################################################
# Training and Evaluation Utilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We define a single atomic training function that is dispatched via
# ``joblib`` with the loky (process-based) backend for true parallelism.
#
# **Checkpointing**: Results are saved per-run to a checkpoint file
# so that a crashed/interrupted experiment can be resumed without
# re-running completed work.
#
# **Performance tuning**:
#
# - Each worker uses 1 BLAS thread (``torch.set_num_threads(1)``)
#   since Apple Silicon Accelerate's ``eigh`` is fastest single-threaded.
#   With 14 workers this fully utilizes all CPU cores.
# - ``optimizer.zero_grad(set_to_none=True)`` avoids memset overhead.
# - Data arrives as numpy arrays for joblib memmapping (no pickle).
#

N_RUNS = 10
EPOCHS = 200
BATCH_SIZE = 30
LR = 5e-3

# Checkpoint file: per-run results saved as they complete.
try:
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "liebn_checkpoint.json")
except NameError:
    CHECKPOINT_PATH = os.path.join(tempfile.gettempdir(), "liebn_checkpoint.json")


def _load_checkpoint():
    """Load existing checkpoint, return dict of completed runs."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


def _save_checkpoint(checkpoint):
    """Atomically save checkpoint (write tmp + rename)."""
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(tmp, CHECKPOINT_PATH)


def _train_single_run(
    run_seed,
    X_train,
    y_train,
    X_test,
    y_test,
    dims,
    n_classes,
    bn_type,
    bn_kwargs,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
):
    """Train and evaluate a single run (atomic unit of work).

    Data arrives as numpy arrays (for joblib memmapping) and is
    converted to tensors inside the worker process.

    Returns (accuracy, fit_time) or (NaN, fit_time) on failure.
    """
    # One BLAS thread per worker; eigh on Apple Silicon Accelerate is
    # fastest single-threaded. With 14 workers we use all 14 cores.
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    # Convert numpy → tensors inside each worker process.
    # Use torch.tensor() (not as_tensor) to copy from read-only memmaps.
    X_train = torch.tensor(X_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_gen = torch.Generator()
    train_gen.manual_seed(run_seed)

    model = SPDNetModel(dims, n_classes, bn_type=bn_type, bn_kwargs=bn_kwargs)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    epoch_times = []
    try:
        for epoch in range(epochs):
            t0 = time.time()
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            epoch_times.append(time.time() - t0)

        # Batch-mean accuracy (matches paper's training_script.py)
        model.eval()
        batch_accs = []
        with torch.no_grad():
            for xb, yb in test_loader:
                acc = (model(xb).argmax(1) == yb).sum().item() / yb.shape[0]
                batch_accs.append(acc)

        return np.mean(batch_accs) * 100.0, np.mean(epoch_times[-10:])
    except torch._C._LinAlgError as e:
        warnings.warn(f"Run {run_seed} failed (LinAlgError): {e}")
        fit_time = np.mean(epoch_times[-10:]) if epoch_times else 0.0
        return float("nan"), fit_time
    except RuntimeError as e:
        if "linalg" in str(e).lower() or "cholesky" in str(e).lower():
            warnings.warn(f"Run {run_seed} failed (linalg RuntimeError): {e}")
            fit_time = np.mean(epoch_times[-10:]) if epoch_times else 0.0
            return float("nan"), fit_time
        raise


######################################################################
# Dataset Loading: HDM05
# ----------------------
#
# HDM05 contains 2086 pre-computed 93x93 SPD covariance matrices
# representing 117 motion capture classes.
#
# - Architecture: ``[93, 30]``
# - Source: `HDM05 Motion Capture Database <https://resources.mpi-inf.mpg.de/HDM05/>`_
#


def download_and_extract(url, dest_dir, zip_name, extract_tgz=None):
    """Download a zip file and extract it."""
    zip_path = dest_dir / zip_name
    if not zip_path.exists():
        print(f"Downloading {zip_name}...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    if extract_tgz:
        tgz_path = dest_dir / extract_tgz
        if tgz_path.exists():
            with tarfile.open(tgz_path, "r:gz") as tf:
                tf.extractall(dest_dir)


def load_hdm05(data_dir):
    """Load HDM05 dataset: pre-computed 93x93 SPD covariance matrices."""
    hdm_path = data_dir / "HDM05"
    if not hdm_path.exists():
        download_and_extract(
            "https://www.dropbox.com/scl/fi/x2ouxjwqj3zrb1idgkg2g/"
            "HDM05.zip?rlkey=4f90ktgzfz28x3i2i4ylu6dvu&dl=1",
            data_dir,
            "HDM05.zip",
        )
    names = sorted(f for f in os.listdir(hdm_path) if f.endswith(".npy"))
    X_list, y_list = [], []
    for name in names:
        x = np.load(hdm_path / name).real
        label = int(name.split(".")[0].split("_")[-1])
        X_list.append(x)
        y_list.append(label)
    X = torch.from_numpy(np.stack(X_list)).double()
    y = torch.from_numpy(np.array(y_list)).long()
    print(
        f"HDM05: {X.shape[0]} samples, {len(set(y_list))} classes, "
        f"matrix size {X.shape[1]}x{X.shape[2]}"
    )
    return X, y


X_hdm, y_hdm = load_hdm05(DATA_DIR)
eigvals_hdm = torch.linalg.eigvalsh(X_hdm)
print(
    f"HDM05 SPD check: min eigenvalue = {eigvals_hdm.min():.2e}, "
    f"max = {eigvals_hdm.max():.2e}"
)


######################################################################
# HDM05 Experiments
# -----------------
#
# We run 10 independent random-split experiments with 7 configurations
# matching Table 4b of the paper: SPDNet, SPDNetBN, AIM-(1), LEM-(1),
# LCM-(1), AIM-(1.5), and LCM-(0.5).
#
# Split: 50/50 train/test (random shuffle).
# Training: 200 epochs, batch_size=30, lr=5e-3, Adam with amsgrad.
#

HDM05_DIMS = [93, 30]
HDM05_CLASSES = len(torch.unique(y_hdm))
print(f"HDM05: dims={HDM05_DIMS}, n_classes={HDM05_CLASSES}")

hdm05_configs = {
    "SPDNet": {"bn_type": None, "bn_kwargs": None},
    "SPDNetBN": {"bn_type": "SPDBN", "bn_kwargs": {"momentum": 0.1}},
    "AIM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "AIM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LEM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LEM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LCM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LCM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "AIM-(1.5)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "AIM",
            "theta": 1.5,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LCM-(0.5)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LCM",
            "theta": 0.5,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
}

######################################################################
# Dataset Loading: Radar
# ----------------------
#
# The Radar dataset contains 3000 complex time-frequency signals
# (3 gesture classes), converted to 20x20 SPD matrices via
# covariance pooling.
#
# Signal processing pipeline:
#
# 1. Split complex signals into overlapping windows (size=20, hop=10)
# 2. Compute real covariance from complex windowed signal
# 3. Apply ReEig to ensure well-conditioned SPD output
#
# - Architecture: ``[20, 16, 12]``
#


def _split_signal_cplx(x, window_size=20, hop_length=10):
    """Window complex signals into overlapping segments.

    Input: (batch, 2, T) where dim=1 is [real, imag]
    Output: (batch, 2, window_size, T')
    """
    x_re = x[:, 0:1, :]
    x_im = x[:, 1:2, :]
    x_re_w = x_re.unfold(2, window_size, hop_length)
    x_im_w = x_im.unfold(2, window_size, hop_length)
    x_re_out = x_re_w.squeeze(1).permute(0, 2, 1)
    x_im_out = x_im_w.squeeze(1).permute(0, 2, 1)
    return torch.stack([x_re_out, x_im_out], dim=1)


def _cov_pool_cplx(f):
    """Compute real covariance matrix from complex windowed signal.

    Input: (batch, 2, n, T)
    Output: (batch, n, n) real SPD covariance matrix
    """
    f_re = f[:, 0, :, :].double()
    f_im = f[:, 1, :, :].double()
    f_re = f_re - f_re.mean(-1, keepdim=True)
    f_im = f_im - f_im.mean(-1, keepdim=True)
    T = f.shape[-1]
    X_Re = (f_re @ f_re.mT + f_im @ f_im.mT) / (T - 1)
    return ensure_sym(X_Re)


def load_radar(data_dir):
    """Load Radar dataset: complex signals -> 20x20 SPD covariance matrices."""
    radar_path = data_dir / "radar"
    if not radar_path.exists():
        download_and_extract(
            "https://www.dropbox.com/s/dfnlx2bnyh3kjwy/data.zip?e=1&dl=1",
            data_dir,
            "data.zip",
            extract_tgz="data/radar.tgz",
        )
    names = sorted(f for f in os.listdir(radar_path) if f.endswith(".npy"))
    signals, labels = [], []
    for name in names:
        x = np.load(radar_path / name)
        x_ri = np.stack([x.real, x.imag], axis=0)
        signals.append(x_ri)
        labels.append(int(name.split(".")[0].split("_")[-1]))
    signals_t = torch.from_numpy(np.stack(signals)).float()
    y = torch.from_numpy(np.array(labels)).long()
    with torch.no_grad():
        windowed = _split_signal_cplx(signals_t, window_size=20, hop_length=10)
        X_cov = _cov_pool_cplx(windowed)
        reeig = ReEig()
        X = reeig(X_cov)
    print(
        f"Radar: {X.shape[0]} samples, {len(set(labels))} classes, "
        f"matrix size {X.shape[1]}x{X.shape[2]}"
    )
    return X, y


X_radar, y_radar = load_radar(DATA_DIR)
eigvals_radar = torch.linalg.eigvalsh(X_radar)
print(
    f"Radar SPD check: min eigenvalue = {eigvals_radar.min():.2e}, "
    f"max = {eigvals_radar.max():.2e}"
)


######################################################################
# Dataset Loading: AFEW
# ---------------------
#
# AFEW (Acted Facial Expressions in the Wild) contains pre-computed
# 400x400 SPD covariance matrices for facial expression recognition
# (7 emotion classes). The dataset comes pre-split into train/val.
#
# - Architecture: ``[400, 200, 100, 50]``
# - Source: `AFEW Dataset <https://cs.anu.edu.au/few/AFEW.html>`_
#


def load_afew(data_dir):
    """Load AFEW dataset: pre-computed 400x400 SPD covariance matrices."""
    afew_path = data_dir / "afew"
    if not afew_path.exists():
        # Try extracting from data/afew.tgz
        tgz_candidates = [
            data_dir / "data" / "afew.tgz",
            data_dir / "afew.tgz",
        ]
        for tgz in tgz_candidates:
            if tgz.exists():
                with tarfile.open(tgz, "r:gz") as tf:
                    tf.extractall(data_dir)
                break
        else:
            raise FileNotFoundError(
                "AFEW data not found. Place afew.tgz in the data/ directory."
            )

    train_path = afew_path / "train"
    val_path = afew_path / "val"

    def _load_split(split_path):
        names = sorted(f for f in os.listdir(split_path) if f.endswith(".npy"))
        X_list, y_list = [], []
        for name in names:
            x = np.load(split_path / name).real
            label = int(name.split(".")[0].split("_")[-1])
            X_list.append(x)
            y_list.append(label)
        return (
            torch.from_numpy(np.stack(X_list)).double(),
            torch.from_numpy(np.array(y_list)).long(),
        )

    X_train, y_train = _load_split(train_path)
    X_val, y_val = _load_split(val_path)
    print(
        f"AFEW: {X_train.shape[0]} train + {X_val.shape[0]} val samples, "
        f"{len(set(y_train.tolist()) | set(y_val.tolist()))} classes, "
        f"matrix size {X_train.shape[1]}x{X_train.shape[2]}"
    )
    return X_train, y_train, X_val, y_val


X_afew_train, y_afew_train, X_afew_val, y_afew_val = load_afew(DATA_DIR)
eigvals_afew = torch.linalg.eigvalsh(X_afew_train)
print(
    f"AFEW SPD check: min eigenvalue = {eigvals_afew.min():.2e}, "
    f"max = {eigvals_afew.max():.2e}"
)


######################################################################
# Radar Experiments
# -----------------
#
# We run 10 independent random-split experiments with 6 configurations
# matching Table 4a: SPDNet, SPDNetBN, AIM-(1), LEM-(1), LCM-(1),
# and LCM-(-0.5).
#
# Split: 50/25/25 train/val/test (random shuffle; val discarded).
# We use ``test_ratio=0.25, val_ratio=0.25`` to match the paper's split.
#

RADAR_DIMS = [20, 16, 12]
RADAR_CLASSES = len(torch.unique(y_radar))
print(f"Radar: dims={RADAR_DIMS}, n_classes={RADAR_CLASSES}")

radar_configs = {
    "SPDNet": {"bn_type": None, "bn_kwargs": None},
    "SPDNetBN": {"bn_type": "SPDBN", "bn_kwargs": {"momentum": 0.1}},
    "AIM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "AIM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LEM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LEM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LCM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LCM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LCM-(-0.5)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LCM",
            "theta": -0.5,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
}

######################################################################
# AFEW Experiments
# ----------------
#
# AFEW (Acted Facial Expressions in the Wild) has a fixed train/val split,
# so we run 10 experiments varying only model initialization.
#
# Architecture: ``[400, 200, 100, 50]`` (from the original SPDNet paper,
# Huang & Van Gool, 2017).
#
# .. note::
#
#    The LieBN paper (Table 4) uses the **FPHA** dataset (63x63, 45 classes)
#    rather than AFEW (400x400, 7 classes). We include AFEW as an additional
#    benchmark; no paper comparison numbers are available.
#

AFEW_DIMS = [400, 200, 100, 50]
AFEW_CLASSES = len(set(y_afew_train.tolist()) | set(y_afew_val.tolist()))
print(f"AFEW: dims={AFEW_DIMS}, n_classes={AFEW_CLASSES}")

afew_configs = {
    "SPDNet": {"bn_type": None, "bn_kwargs": None},
    "SPDNetBN": {"bn_type": "SPDBN", "bn_kwargs": {"momentum": 0.1}},
    "AIM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "AIM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LEM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LEM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LCM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LCM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "AIM-(1.5)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "AIM",
            "theta": 1.5,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LCM-(0.5)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LCM",
            "theta": 0.5,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
}

######################################################################
# Run All Experiments (Parallelized with Checkpointing)
# -----------------------------------------------------
#
# Experiments are dispatched per-dataset via ``joblib`` (loky backend)
# for true multi-process parallelism. Each worker uses 1 BLAS thread
# since Apple Silicon Accelerate's ``eigh`` is fastest single-threaded,
# with 14 workers fully utilizing all CPU cores.
#
# **Checkpointing**: After each dataset completes, all finished
# per-run results are saved to ``liebn_checkpoint.json``. On restart,
# already-completed (dataset, method, run) tuples are skipped.
#
# Total jobs: HDM05 (7 x 10) + Radar (6 x 10) + AFEW (7 x 10) = 200
#

checkpoint = _load_checkpoint()
n_cached = len(checkpoint)
if n_cached > 0:
    print(f"Loaded checkpoint with {n_cached} completed runs.")


def _run_dataset_jobs(dataset_name, configs, jobs_and_keys, checkpoint):
    """Dispatch jobs for one dataset, skipping already-checkpointed runs.

    Returns dict of {method: [(acc, fit_time), ...]} for this dataset.
    Mutates ``checkpoint`` in-place with newly completed runs.
    """

    filtered_jobs = []
    filtered_keys = []
    cached_results = defaultdict(list)

    for (ds, method, run_idx), job in jobs_and_keys:
        key = f"{ds}|{method}|{run_idx}"
        if key in checkpoint:
            acc = checkpoint[key]["acc"]
            # Restore NaN for failed runs (stored as null in JSON).
            if acc is None:
                acc = float("nan")
            cached_results[method].append((acc, checkpoint[key]["fit_time"]))
        else:
            filtered_jobs.append(job)
            filtered_keys.append((ds, method, run_idx))

    n_skip = len(jobs_and_keys) - len(filtered_jobs)
    n_todo = len(filtered_jobs)
    if n_skip > 0:
        print(f"  {dataset_name}: {n_skip} runs cached, {n_todo} remaining.")
    if n_todo == 0:
        print(f"  {dataset_name}: all runs cached, nothing to do.")
    else:
        t0 = time.time()
        # n_jobs=-1: 14 workers x 1 BLAS thread = 14 cores fully used.
        raw = Parallel(n_jobs=-1, verbose=10)(filtered_jobs)
        elapsed = time.time() - t0
        print(f"  {dataset_name}: {n_todo} runs finished in {elapsed:.1f}s")

        # Save to checkpoint immediately.
        for (ds, method, run_idx), result in zip(filtered_keys, raw):
            key = f"{ds}|{method}|{run_idx}"
            acc, ft = result
            checkpoint[key] = {
                "acc": None if np.isnan(acc) else acc,
                "fit_time": ft,
                "status": "failed" if np.isnan(acc) else "ok",
            }
            cached_results[method].append(result)
        _save_checkpoint(checkpoint)
        print(f"  Checkpoint saved ({len(checkpoint)} total runs).")

    return dict(cached_results)


def _aggregate(runs):
    """Aggregate per-run (acc, fit_time) tuples."""
    accs = [r[0] for r in runs if not np.isnan(r[0])]
    fts = [r[1] for r in runs if not np.isnan(r[0])]
    n_failed = sum(1 for r in runs if np.isnan(r[0]))
    if n_failed > 0:
        warnings.warn(f"{n_failed}/{len(runs)} runs failed (NaN)")
    if not accs:
        return {
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "folds": [],
            "fit_time": 0.0,
        }
    return {
        "mean": np.mean(accs),
        "std": np.std(accs),
        "max": np.max(accs),
        "folds": accs,
        "fit_time": np.mean(fts),
    }


# ---- Prepare HDM05 jobs (fixed 50/50 split) ----
n_hdm = len(X_hdm)
rng_hdm = np.random.RandomState(GLOBAL_SEED)
perm_hdm = rng_hdm.permutation(n_hdm)
n_test_hdm = int(0.5 * n_hdm)
Xtr_hdm = X_hdm[perm_hdm[n_test_hdm:]].numpy()
ytr_hdm = y_hdm[perm_hdm[n_test_hdm:]].numpy()
Xte_hdm = X_hdm[perm_hdm[:n_test_hdm]].numpy()
yte_hdm = y_hdm[perm_hdm[:n_test_hdm]].numpy()

hdm05_jobs = []
for name, cfg in hdm05_configs.items():
    for i in range(N_RUNS):
        job = delayed(_train_single_run)(
            GLOBAL_SEED + i,
            Xtr_hdm,
            ytr_hdm,
            Xte_hdm,
            yte_hdm,
            HDM05_DIMS,
            HDM05_CLASSES,
            cfg["bn_type"],
            cfg["bn_kwargs"],
        )
        hdm05_jobs.append((("HDM05", name, i), job))

# ---- Prepare Radar jobs (per-run 50/25/25 split) ----
n_radar = len(X_radar)
n_test_radar = int(0.25 * n_radar)
n_val_radar = int(0.25 * n_radar)
X_radar_np = X_radar.numpy()
y_radar_np = y_radar.numpy()

radar_jobs = []
for name, cfg in radar_configs.items():
    for i in range(N_RUNS):
        run_seed = GLOBAL_SEED + i
        rng = np.random.RandomState(run_seed)
        perm = rng.permutation(n_radar)
        Xte_r = X_radar_np[perm[:n_test_radar]]
        yte_r = y_radar_np[perm[:n_test_radar]]
        Xtr_r = X_radar_np[perm[n_test_radar + n_val_radar :]]
        ytr_r = y_radar_np[perm[n_test_radar + n_val_radar :]]
        job = delayed(_train_single_run)(
            run_seed,
            Xtr_r,
            ytr_r,
            Xte_r,
            yte_r,
            RADAR_DIMS,
            RADAR_CLASSES,
            cfg["bn_type"],
            cfg["bn_kwargs"],
        )
        radar_jobs.append((("Radar", name, i), job))

# ---- Prepare AFEW jobs (fixed train/val split) ----
X_afew_train_np = X_afew_train.numpy()
y_afew_train_np = y_afew_train.numpy()
X_afew_val_np = X_afew_val.numpy()
y_afew_val_np = y_afew_val.numpy()

afew_jobs = []
for name, cfg in afew_configs.items():
    for i in range(N_RUNS):
        job = delayed(_train_single_run)(
            GLOBAL_SEED + i,
            X_afew_train_np,
            y_afew_train_np,
            X_afew_val_np,
            y_afew_val_np,
            AFEW_DIMS,
            AFEW_CLASSES,
            cfg["bn_type"],
            cfg["bn_kwargs"],
        )
        afew_jobs.append((("AFEW", name, i), job))

# ---- Run datasets sequentially, saving after each ----
# This way Radar+HDM05 results are safe even if AFEW crashes.

total_jobs = len(hdm05_jobs) + len(radar_jobs) + len(afew_jobs)
print(
    f"\nTotal: {total_jobs} training runs "
    f"({len(hdm05_configs)} + {len(radar_configs)} + "
    f"{len(afew_configs)} methods x {N_RUNS} runs)."
)

t_wall_start = time.time()

print("\n--- HDM05 ---")
hdm05_raw = _run_dataset_jobs("HDM05", hdm05_configs, hdm05_jobs, checkpoint)
print("\n--- Radar ---")
radar_raw = _run_dataset_jobs("Radar", radar_configs, radar_jobs, checkpoint)
print("\n--- AFEW ---")
afew_raw = _run_dataset_jobs("AFEW", afew_configs, afew_jobs, checkpoint)

t_wall = time.time() - t_wall_start
print(f"\nAll experiments finished in {t_wall:.1f}s")

# ---- Aggregate per-method results ----
hdm05_results = {m: _aggregate(r) for m, r in hdm05_raw.items()}
radar_results = {m: _aggregate(r) for m, r in radar_raw.items()}
afew_results = {m: _aggregate(r) for m, r in afew_raw.items()}

for ds, results in [
    ("HDM05", hdm05_results),
    ("Radar", radar_results),
    ("AFEW", afew_results),
]:
    print(f"\n{ds}:")
    for m, r in results.items():
        print(
            f"  {m}: {r['mean']:.2f} +/- {r['std']:.2f} "
            f"(max={r['max']:.2f}, fit_time={r['fit_time']:.2f}s)"
        )


######################################################################
# Results Comparison & Visualization
# -----------------------------------
#
# We compare our reproduction results against the paper's Table 4 numbers
# for HDM05 and Radar. AFEW results are shown without paper comparison
# (the paper uses FPHA, a different dataset).
#

# Paper Table 4 numbers: (mean, std, max, fit_time)
paper_results = {
    "Radar": {
        "SPDNet": (93.25, 1.10, 94.4, 0.98),
        "SPDNetBN": (94.85, 0.99, 96.13, 1.56),
        "AIM-(1)": (95.47, 0.90, 96.27, 1.62),
        "LEM-(1)": (94.89, 1.04, 96.8, 1.28),
        "LCM-(1)": (93.52, 1.07, 95.2, 1.11),
        "LCM-(-0.5)": (94.80, 0.71, 95.73, 1.43),
    },
    "HDM05": {
        "SPDNet": (59.13, 0.67, 60.34, 0.57),
        "SPDNetBN": (66.72, 0.52, 67.66, 0.97),
        "AIM-(1)": (67.79, 0.65, 68.75, 1.14),
        "LEM-(1)": (65.05, 0.63, 66.05, 0.87),
        "LCM-(1)": (66.68, 0.71, 68.52, 0.66),
        "AIM-(1.5)": (68.16, 0.68, 69.25, 1.46),
        "LCM-(0.5)": (70.84, 0.92, 72.27, 1.01),
    },
    "AFEW": {},
}

radar_methods = list(radar_configs.keys())
hdm05_methods = list(hdm05_configs.keys())
afew_methods = list(afew_configs.keys())

######################################################################
# Results Tables
# ~~~~~~~~~~~~~~
#


def _print_table(dataset, methods, our_results, paper):
    """Print comparison table for one dataset."""
    hdr = (
        f"{'Method':<14} | {'Fit Time':>8} | "
        f"{'Mean+-STD (Ours)':>18} {'Max (Ours)':>10} | "
        f"{'Mean+-STD (Paper)':>18} {'Max (Paper)':>11}"
    )
    sep = "=" * len(hdr)
    print(f"\n{dataset}")
    print(sep)
    print(hdr)
    print(sep)
    for m in methods:
        ours = our_results.get(m, {})
        p = paper.get(m)
        ft = f"{ours.get('fit_time', 0):.2f}" if ours else "---"
        o_str = f"{ours['mean']:.2f}+-{ours['std']:.2f}" if ours else "---"
        o_max = f"{ours['max']:.2f}" if ours else "---"
        if p:
            p_str = f"{p[0]:.2f}+-{p[1]:.2f}"
            p_max = f"{p[2]:.2f}"
        else:
            p_str, p_max = "---", "---"
        print(f"{m:<14} | {ft:>8} | {o_str:>18} {o_max:>10} | {p_str:>18} {p_max:>11}")
    print(sep)


_print_table("Radar", radar_methods, radar_results, paper_results["Radar"])
_print_table("HDM05", hdm05_methods, hdm05_results, paper_results["HDM05"])
_print_table("AFEW", afew_methods, afew_results, paper_results["AFEW"])

######################################################################
# Save results to JSON for reproducibility.
#

try:
    results_path = os.path.join(os.path.dirname(__file__), "liebn_table4_results.json")
except NameError:
    results_path = os.path.join(tempfile.gettempdir(), "liebn_table4_results.json")
results_to_save = {
    "radar": {k: dict(v) for k, v in radar_results.items()},
    "hdm05": {k: dict(v) for k, v in hdm05_results.items()},
    "afew": {k: dict(v) for k, v in afew_results.items()},
    "paper_results": {
        ds: {
            m: {"mean": v[0], "std": v[1], "max": v[2], "fit_time": v[3]}
            for m, v in mv.items()
        }
        for ds, mv in paper_results.items()
        if mv  # skip empty (AFEW has no paper numbers)
    },
}
with open(results_path, "w") as f:
    json.dump(results_to_save, f, indent=2)
print(f"\nResults saved to {results_path}")

######################################################################
# Comparison Bar Chart
# ~~~~~~~~~~~~~~~~~~~~
#

fig, axes = plt.subplots(1, 3, figsize=(22, 5))

dataset_info = [
    ("Radar", radar_results, radar_methods),
    ("HDM05", hdm05_results, hdm05_methods),
    ("AFEW", afew_results, afew_methods),
]

for ax, (dataset, our_results, methods) in zip(axes, dataset_info):
    x_pos = np.arange(len(methods))
    has_paper = bool(paper_results.get(dataset))

    ours_means = [our_results[m]["mean"] for m in methods]
    ours_stds = [our_results[m]["std"] for m in methods]

    if has_paper:
        width = 0.35
        paper_means = [paper_results[dataset][m][0] for m in methods]
        paper_stds = [paper_results[dataset][m][1] for m in methods]

        ax.bar(
            x_pos - width / 2,
            ours_means,
            width,
            yerr=ours_stds,
            label="Ours",
            capsize=3,
            color="#3498db",
            alpha=0.85,
        )
        ax.bar(
            x_pos + width / 2,
            paper_means,
            width,
            yerr=paper_stds,
            label="Paper",
            capsize=3,
            color="#e74c3c",
            alpha=0.85,
        )
        all_vals = ours_means + paper_means
    else:
        width = 0.5
        ax.bar(
            x_pos,
            ours_means,
            width,
            yerr=ours_stds,
            label="Ours",
            capsize=3,
            color="#3498db",
            alpha=0.85,
        )
        all_vals = ours_means

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(dataset)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ymin = min(all_vals) - 5
    ymax = max(all_vals) + 3
    ax.set_ylim(ymin, ymax)

plt.suptitle("LieBN Batch Normalization: SPDNet Results", fontweight="bold")
plt.tight_layout()
plt.show()

######################################################################
# References
# ----------
#
# .. bibliography::
#    :filter: docname in docnames
#
