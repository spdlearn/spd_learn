# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from .attention import SPDTransformerEncoderLayer, SPMultiheadAttention
from .batchnorm import (
    BatchReNorm,
    SPDBatchNormBW,
    SPDBatchNormCholesky,
    SPDBatchNormHarmonicArithmetic,
    SPDBatchNormMean,
    SPDBatchNormMeanVar,
)
from .bilinear import BiMap, BiMapIncreaseDim, GraphBiMap
from .covariance import CovLayer
from .dropout import SPDDropout
from .liebn import SPDBatchNormLie
from .manifold import PositiveDefiniteScalar, SymmetricPositiveDefinite
from .modeig import ExpEig, LogEig, ReEig
from .pooling import GaussianSPDDescriptor, RiemannianHighOrderPooling
from .regularize import Shrinkage, TraceNorm
from .residual import LogEuclideanResidual
from .rietrans import SPDLinear, SPDRMLR
from .utils import PatchEmbeddingLayer, Vec, Vech
from .wavelet import WaveletConv


__all__ = [
    # covariance
    "CovLayer",
    # manifold parametrization
    "SymmetricPositiveDefinite",
    "PositiveDefiniteScalar",
    # modeig
    "LogEig",
    "ReEig",
    "ExpEig",
    # bilinear
    "BiMap",
    "BiMapIncreaseDim",
    "GraphBiMap",
    # batchnorm
    "SPDBatchNormMean",
    "BatchReNorm",
    "SPDBatchNormMeanVar",
    "SPDBatchNormLie",
    "SPDBatchNormHarmonicArithmetic",
    "SPDBatchNormBW",
    "SPDBatchNormCholesky",
    # attention
    "SPMultiheadAttention",
    "SPDTransformerEncoderLayer",
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
