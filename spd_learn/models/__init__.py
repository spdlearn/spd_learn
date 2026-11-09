# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from .coratt import CorAtt, CorrAttention  # noqa: F401
from .cornet import CorNet
from .eegspdnet import EEGSPDNet
from .gbwatt import GBWAtt
from .green import Green
from .gyroatt import GyroAtt, GyroSPDAttention, SymSPDAttention  # noqa: F401
from .matt import AttentionManifold, MAtt  # noqa: F401  (AttentionManifold re-exported)
from .phase_spdnet import (  # noqa: F401  (PhaseDelay re-exported)
    PhaseDelay,
    PhaseSPDNet,
)
from .spdmetaformer import SPDMetaFormer, SPDMetaFormerBlock  # noqa: F401
from .spdnet import SPDNet
from .spdtransnet import SPDTransNet  # noqa: F401
from .spnn import SPNN
from .tensorcsp import TensorCSPNet
from .tsmnet import TSMNet


# ``AttentionManifold`` and ``PhaseDelay`` are model building blocks: importable
# from ``spd_learn.models`` and documented in api.rst, but intentionally kept out
# of ``__all__`` -- they are sub-modules of the models above (different
# constructors, not standalone ``n_chans``/``n_outputs`` models), while
# ``__all__`` is the contract the model test harness iterates to instantiate
# full models.
__all__ = [
    "TensorCSPNet",
    "SPDNet",
    "TSMNet",
    "MAtt",
    "EEGSPDNet",
    "PhaseSPDNet",
    "Green",
    "GBWAtt",
    "SPNN",
    "GyroAtt",
    "SPDMetaFormer",
    "CorAtt",
    "CorNet",
]

__filter_bank_models__ = ["TensorCSPNet"]

__sequence_models__ = ["SPDTransNet"]
