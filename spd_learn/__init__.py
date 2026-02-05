# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""SPD Learn: Deep Learning on Riemannian Manifolds.

A pure PyTorch library for Symmetric Positive Definite (SPD) matrix learning.
"""

# Initialization functions (like torch.nn.init)
from . import init

# Models
from .models import EEGSPDNet, Green, MAtt, PhaseSPDNet, SPDNet, TensorCSPNet, TSMNet

# Modules (layers)
from .modules import (
    BatchReNorm,
    BiMap,
    BiMapIncreaseDim,
    CovLayer,
    ExpEig,
    LogEig,
    PatchEmbeddingLayer,
    ReEig,
    Shrinkage,
    SPDBatchNormMean,
    SPDBatchNormMeanVar,
    SPDDropout,
    TraceNorm,
    Vec,
    Vech,
    WaveletConv,
)
from .version import __version__


__all__ = [
    # Version
    "__version__",
    # Initialization (like torch.nn.init)
    "init",
    # Models
    "EEGSPDNet",
    "Green",
    "MAtt",
    "PhaseSPDNet",
    "SPDNet",
    "TensorCSPNet",
    "TSMNet",
    # Modules
    "BatchReNorm",
    "BiMap",
    "BiMapIncreaseDim",
    "SPDBatchNormMean",
    "CovLayer",
    "ExpEig",
    "LogEig",
    "PatchEmbeddingLayer",
    "ReEig",
    "Shrinkage",
    "SPDBatchNormMeanVar",
    "SPDDropout",
    "TraceNorm",
    "Vec",
    "Vech",
    "WaveletConv",
]
