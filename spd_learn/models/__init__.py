# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from .eegspdnet import EEGSPDNet
from .green import Green
from .matt import MAtt
from .phase_spdnet import PhaseSPDNet
from .spdnet import SPDNet
from .tensorcsp import TensorCSPNet
from .tsmnet import TSMNet


__all__ = [
    "TensorCSPNet",
    "SPDNet",
    "TSMNet",
    "MAtt",
    "EEGSPDNet",
    "PhaseSPDNet",
    "Green",
]

__filter_bank_models__ = ["TensorCSPNet"]
