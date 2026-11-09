# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from .batchnorm import (
    BatchReNorm,
    SPDBatchNormBW,
    SPDBatchNormCholesky,
    SPDBatchNormHarmonic,
    SPDBatchNormMean,
    SPDBatchNormMeanVar,
)
from .bilinear import BiMap, BiMapIncreaseDim
from .covariance import CovLayer
from .dropout import SPDDropout
from .gbw import GBWAttention, GBWGeometry, GBWPatchCovariance  # noqa: F401
from .liebn import SPDBatchNormLie
from .manifold import (
    PositiveDefiniteScalar,
    SymmetricPositiveDefinite,
    step_dynamic_trivialization,
)
from .modeig import ExpEig, LogEig, ReEig
from .pooling import RHOP, GaussianSPD
from .regularize import Shrinkage, TraceNorm
from .residual import LogEuclideanResidual
from .rietrans import SPDRMLR, SPDLinear
from .spdtransnet import (  # noqa: F401
    LearnedSinusoidalPosition,
    SPDInterEpochEncoder,
    SPDIntraEpochEncoder,
    SPDLogTokenizer,
    SPDTemporalEncoder,
    SPDTransformerEncoderLayer,
    SPMultiheadAttention,
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
    "GaussianSPD",
    "RHOP",
    "SPDRMLR",
    "SPDLinear",
    "SPDBatchNormHarmonic",
    "SPDBatchNormBW",
    "SPDBatchNormCholesky",
]

__sequence_modules__ = [
    "SPDLogTokenizer",
    "SPMultiheadAttention",
    "SPDTransformerEncoderLayer",
    "LearnedSinusoidalPosition",
    "SPDTemporalEncoder",
    "SPDIntraEpochEncoder",
    "SPDInterEpochEncoder",
]
