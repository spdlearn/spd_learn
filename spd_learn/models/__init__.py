# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from .eegspdnet import EEGSPDNet
from .green import Green
from .matt import AttentionManifold, MAtt  # noqa: F401  (AttentionManifold re-exported)
from .phase_spdnet import (  # noqa: F401  (PhaseDelay re-exported)
    PhaseDelay,
    PhaseSPDNet,
)
from .spdnet import SPDNet
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
]

__filter_bank_models__ = ["TensorCSPNet"]
