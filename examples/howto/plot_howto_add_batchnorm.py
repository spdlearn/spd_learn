"""
.. _howto-add-batchnorm:

How to Add Batch Normalization to an SPDNet
===========================================

Insert Riemannian batch normalization into an existing SPDNet pipeline
to stabilize training and improve convergence.

**Prerequisites**: Familiarity with SPDNet building blocks
(see :ref:`tutorial-building-blocks`).

"""

######################################################################
# The Problem
# -----------
#
# You have a working SPDNet but training is unstable or converges slowly.
# Adding batch normalization after each ``BiMap`` layer can help.
#

import torch
import torch.nn as nn

from spd_learn.modules import BiMap, LogEig, ReEig, SPDBatchNormLie

######################################################################
# Step 1: Choose Your Normalization Layer
# ----------------------------------------
#
# spd_learn provides three batch normalization modules:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 30 70
#
#    * - Module
#      - When to Use
#    * - :class:`~spd_learn.modules.SPDBatchNormMeanVar`
#      - Standard choice. AIRM Frechet mean + variance scaling.
#    * - :class:`~spd_learn.modules.SPDBatchNormLie`
#      - Multiple metrics (AIM, LEM, LCM). Based on Lie group
#        structure :cite:p:`chen2024liebn`.
#    * - :class:`~spd_learn.modules.SPDBatchNormMean`
#      - Mean-only centering (no variance scaling). Simplest option.
#

######################################################################
# Step 2: Insert After BiMap, Before ReEig
# -----------------------------------------
#
# The standard placement is ``BiMap -> BN -> ReEig``. The final BiMap
# uses BN but skips ReEig:
#

dims = [64, 32, 16]  # your SPD matrix dimensions
layers = []
for i in range(len(dims) - 1):
    layers.append(BiMap(dims[i], dims[i + 1]))
    layers.append(SPDBatchNormLie(dims[i + 1], metric="LEM"))
    if i < len(dims) - 2:  # no ReEig after last BiMap
        layers.append(ReEig())

features = nn.Sequential(*layers)
print(features)

######################################################################
# Step 3: Complete Network
# -------------------------
#
# Wrap the features with ``LogEig`` and a linear classifier:
#


class SPDNetWithBN(nn.Module):
    """SPDNet with configurable batch normalization."""

    def __init__(self, dims, n_classes, metric="LEM"):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(BiMap(dims[i], dims[i + 1]))
            layers.append(SPDBatchNormLie(dims[i + 1], metric=metric))
            if i < len(dims) - 2:
                layers.append(ReEig())
        self.features = nn.Sequential(*layers)
        self.logeig = LogEig(upper=False, flatten=True)
        self.classifier = nn.Linear(dims[-1] ** 2, n_classes)

    def forward(self, x):
        return self.classifier(self.logeig(self.features(x)))


model = SPDNetWithBN([64, 32, 16], n_classes=4, metric="LEM")

######################################################################
# Verify it works with a dummy forward pass:

X = torch.randn(8, 64, 64)
X = X @ X.mT + 0.01 * torch.eye(64)
out = model(X)
print(f"Input: {X.shape} -> Output: {out.shape}")  # [8, 64, 64] -> [8, 4]

######################################################################
# Key Points
# ----------
#
# - Place BN **after** ``BiMap`` and **before** ``ReEig``
# - Use ``model.train()`` / ``model.eval()`` -- BN uses running stats
#   at inference time
# - ``momentum=0.1`` (default) works well in most cases
# - Consider ``float64`` for numerical stability with Riemannian operations
#
# .. seealso::
#
#    - :ref:`tutorial-batch-normalization` -- Learn how BN works on SPD manifolds
#    - :ref:`howto-choose-metric` -- Choosing between AIM, LEM, and LCM
#    - :class:`~spd_learn.modules.SPDBatchNormLie` -- API reference
#    - :class:`~spd_learn.modules.SPDBatchNormMeanVar` -- API reference
#
# References
# ----------
#
# .. bibliography::
#    :filter: docname in docnames
#
