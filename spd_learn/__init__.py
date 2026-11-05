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
    DeepGeoCCA,
    EEGSPDNet,
    GBWAtt,
    GraphCSPNet,
    Green,
    GyroSPD,
    GyroSPDAtt,
    MAtt,
    PhaseSPDNet,
    RCEEGNet,
    SPDMetaFormer,
    SPDNet,
    SPDNetAE,
    SPDResNet,
    SPDTransNet,
    SSGCNet,
    STaRNet,
    TensorCSPNet,
    TSMNet,
)

# Modules (layers)
from .modules import (
    BatchReNorm,
    BiMap,
    BiMapIncreaseDim,
    CovLayer,
    ExpEig,
    GaussianSPDDescriptor,
    GraphBiMap,
    LogEig,
    PatchEmbeddingLayer,
    ReEig,
    RiemannianHighOrderPooling,
    Shrinkage,
    SPDBatchNormBW,
    SPDBatchNormCholesky,
    SPDBatchNormHarmonicArithmetic,
    SPDBatchNormLie,
    SPDBatchNormMean,
    SPDBatchNormMeanVar,
    SPDDropout,
    SPDLinear,
    SPDRMLR,
    SPDTransformerEncoderLayer,
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
    "CorAtt",
    "CorNet",
    "SPNN",
    "GyroSPDAtt",
    "GraphCSPNet",
    "GBWAtt",
    "SPDTransNet",
    "SPDMetaFormer",
    "DeepGeoCCA",
    "RCEEGNet",
    "STaRNet",
    "SPDResNet",
    "GyroSPD",
    "SPDNetAE",
    "SSGCNet",
    # Modules
    "BatchReNorm",
    "BiMap",
    "BiMapIncreaseDim",
    "GraphBiMap",
    "CovLayer",
    "ExpEig",
    "LogEig",
    "PatchEmbeddingLayer",
    "ReEig",
    "Shrinkage",
    "SPDBatchNormLie",
    "SPDBatchNormMean",
    "SPDBatchNormMeanVar",
    "SPDBatchNormHarmonicArithmetic",
    "SPDBatchNormBW",
    "SPDBatchNormCholesky",
    "SPMultiheadAttention",
    "SPDTransformerEncoderLayer",
    "GaussianSPDDescriptor",
    "RiemannianHighOrderPooling",
    "SPDRMLR",
    "SPDLinear",
    "SPDDropout",
    "TraceNorm",
    "Vec",
    "Vech",
    "WaveletConv",
]
