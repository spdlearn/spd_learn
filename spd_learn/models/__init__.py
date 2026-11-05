# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from .coratt import CorAtt, CorrAttention  # noqa: F401
from .cornet import CorNet
from .deepgeocca import DeepGeoCCA, GeodesicCCALoss  # noqa: F401
from .eegspdnet import EEGSPDNet
from .gbwatt import GBWAtt, GBWAttention  # noqa: F401
from .graph_cspnet import GraphCSPNet
from .green import Green
from .gyroatt import GyroSPDAtt, GyroSPDAttention, SymSPDAttention  # noqa: F401
from .gyrospd import GyroSPD, GyroSPDLayer  # noqa: F401
from .matt import AttentionManifold, MAtt  # noqa: F401  (AttentionManifold re-exported)
from .phase_spdnet import (  # noqa: F401  (PhaseDelay re-exported)
    PhaseDelay,
    PhaseSPDNet,
)
from .rceegnet import RCEEGNet
from .spd_metaformer import SPDMetaFormer, SPDMetaFormerBlock  # noqa: F401
from .spdnet import SPDNet
from .spdnet_ae import SPDDecoderBlock, SPDNetAE  # noqa: F401
from .spdresnet import AdaptiveLogEuclidean, SPDResBlock, SPDResNet  # noqa: F401
from .spdtransnet import SPDTransNet
from .spnn import SPNN
from .ssgcnet import SSGCNet, TangentGraphSPDAggregation  # noqa: F401
from .starnet import SqueezeExcitation, STaRNet  # noqa: F401
from .tensorcsp import TensorCSPNet
from .tsmnet import TSMNet


# ``AttentionManifold`` and ``PhaseDelay`` (and similar building blocks) are
# importable from ``spd_learn.models`` and documented in api.rst, but
# intentionally kept out of ``__all__`` -- they are sub-modules of the models
# above (different constructors, not standalone ``n_chans``/``n_outputs``
# models), while ``__all__`` is the contract the model test harness iterates to
# instantiate full models.
__all__ = [
    "TensorCSPNet",
    "SPDNet",
    "TSMNet",
    "MAtt",
    "EEGSPDNet",
    "PhaseSPDNet",
    "Green",
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
]

__filter_bank_models__ = ["TensorCSPNet", "GraphCSPNet"]
