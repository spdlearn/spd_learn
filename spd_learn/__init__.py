# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""SPD Learn: Deep Learning on Riemannian Manifolds.

A pure PyTorch library for Symmetric Positive Definite (SPD) matrix learning.
"""

# Initialization functions (like torch.nn.init)
from . import init

# Models
from .models import (
    SPNN,
    CorAtt,
    CorNet,
    EEGSPDNet,
    GBWAtt,
    Green,
    GyroAtt,
    MAtt,
    PhaseSPDNet,
    SPDMetaFormer,
    SPDNet,
    SPDTransNet,
    TensorCSPNet,
    TSMNet,
)

# Modules (layers)
from .modules import (
    RHOP,
    SPDRMLR,
    BatchReNorm,
    BiMap,
    BiMapIncreaseDim,
    CovLayer,
    ExpEig,
    GaussianSPD,
    LogEig,
    PatchEmbeddingLayer,
    ReEig,
    Shrinkage,
    SPDBatchNormBW,
    SPDBatchNormCholesky,
    SPDBatchNormHarmonic,
    SPDBatchNormLie,
    SPDBatchNormMean,
    SPDBatchNormMeanVar,
    SPDDropout,
    SPDLinear,
    SPMultiheadAttention,
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
    "CovLayer",
    "ExpEig",
    "LogEig",
    "PatchEmbeddingLayer",
    "ReEig",
    "Shrinkage",
    "SPDBatchNormLie",
    "SPDBatchNormMean",
    "SPDBatchNormMeanVar",
    "SPDDropout",
    "TraceNorm",
    "Vec",
    "Vech",
    "WaveletConv",
    "GBWAtt",
    "SPNN",
    "GyroAtt",
    "SPDMetaFormer",
    "SPDTransNet",
    "GaussianSPD",
    "RHOP",
    "SPDLinear",
    "SPDRMLR",
    "SPMultiheadAttention",
    "CorAtt",
    "CorNet",
    "SPDBatchNormBW",
    "SPDBatchNormCholesky",
    "SPDBatchNormHarmonic",
]
