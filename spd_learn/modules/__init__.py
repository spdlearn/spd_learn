# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from .batchnorm import (
    BatchReNorm,
    SPDBatchNormBW,
    SPDBatchNormCholesky,
    SPDBatchNormHarmonicArithmetic,
    SPDBatchNormMean,
    SPDBatchNormMeanVar,
)
from .bilinear import BiMap, BiMapIncreaseDim
from .covariance import CovLayer
from .dropout import SPDDropout
from .gbw import GBWAttention, GBWGeometry, GBWPatchCovariance  # noqa: F401
from .graph import (  # noqa: F401
    GraphBiMap,
    TimeFrequencyCovariances,
    TimeFrequencyGraph,
)
from .liebn import SPDBatchNormLie
from .manifold import (
    PositiveDefiniteScalar,
    SymmetricPositiveDefinite,
    step_dynamic_trivialization,
)
from .modeig import ExpEig, LogEig, ReEig
from .pooling import GaussianSPDDescriptor, RiemannianHighOrderPooling
from .regularize import Shrinkage, TraceNorm
from .residual import LogEuclideanResidual
from .rietrans import SPDLinear, SPDRMLR
from .spdtransnet import (  # noqa: F401
    LearnedSinusoidalPosition,
    SPDInterEpochEncoder,
    SPDIntraEpochEncoder,
    SPDLogTokenizer,
    SPDTemporalEncoder,
    SPDTransformerEncoderLayer,
    SPMultiheadAttention,
    StructurePreservingMultiheadAttention,
)
from .utils import PatchEmbeddingLayer, Vec, Vech
from .wavelet import WaveletConv


__all__ = [
    # covariance
    "CovLayer",
    # manifold parametrization
    "SymmetricPositiveDefinite",
    "PositiveDefiniteScalar",
    "step_dynamic_trivialization",
    # modeig
    "LogEig",
    "ReEig",
    "ExpEig",
    # bilinear
    "BiMap",
    "BiMapIncreaseDim",
    # batchnorm
    "SPDBatchNormMean",
    "BatchReNorm",
    "SPDBatchNormMeanVar",
    "SPDBatchNormLie",
    "SPDBatchNormHarmonicArithmetic",
    "SPDBatchNormBW",
    "SPDBatchNormCholesky",
    # pooling
    "GaussianSPDDescriptor",
    "RiemannianHighOrderPooling",
    # rietrans
    "SPDRMLR",
    "SPDLinear",
    # dropout
    "SPDDropout",
    # residual
    "LogEuclideanResidual",
    # utils
    "PatchEmbeddingLayer",
    "Vec",
    "Vech",
    # regularization
    "TraceNorm",
    "Shrinkage",
    # wavelet
    "WaveletConv",
]

# Sequence modules operate on explicit token axes, not generic covariance inputs.
__sequence_modules__ = [
    "SPDLogTokenizer",
    "StructurePreservingMultiheadAttention",
    "SPDTransformerEncoderLayer",
    "LearnedSinusoidalPosition",
    "SPDTemporalEncoder",
    "SPDIntraEpochEncoder",
    "SPDInterEpochEncoder",
]
