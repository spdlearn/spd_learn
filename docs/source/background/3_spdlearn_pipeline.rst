.. _background_spdlearn_pipeline:

=========================================
SPD Learn Pipeline and Trivialization
=========================================

This page connects the theory to what the package actually implements, aligned
with the library paper. For a longer walkthrough, see :doc:`/user_guide`.


SPDNet Building Blocks
======================

SPD Learn is centered on the SPDNet pipeline:

.. code-block:: text

   Input SPD -> BiMap -> ReEig -> BiMap -> ReEig -> LogEig -> Linear -> Output

Key layers:

Following the original paper, SPD Learn defines key architectural components as neural
network layers specifically designed to handle and operate on Riemannian geometries.
These layers ensure that the geometric structure of the data is preserved or
appropriately transformed throughout the network:

* **BiMap**: Stiefel-constrained bilinear mapping for SPD-preserving
  dimension change
* **ReEig**: eigenvalue rectification (nonlinearity)
* **LogEig**: maps SPD matrices to the tangent space for Euclidean classifiers

These layers are exposed as :class:`~spd_learn.modules.BiMap`,
:class:`~spd_learn.modules.ReEig`, and :class:`~spd_learn.modules.LogEig`.

The model zoo builds on these blocks and includes SPDNet-based architectures
for neural decoding, such as :class:`~spd_learn.models.TensorCSPNet`,
:class:`~spd_learn.models.TSMNet`, :class:`~spd_learn.models.MAtt`,
:class:`~spd_learn.models.Green`, :class:`~spd_learn.models.EEGSPDNet`,
and :class:`~spd_learn.models.PhaseSPDNet`.


Trivialization in SPD Learn
===========================

The paper and the code use **trivialization-based parametrizations**
:cite:p:`lezcano2019trivializations`, to keep
manifold-valued parameters valid during training.

**Trivialization** maps points on a manifold to vectors in a Euclidean space
via a diffeomorphism (smooth, invertible map):

.. math::

    \phi: \manifold \to \reals^d \quad \text{and} \quad \phi^{-1}: \reals^d \to \manifold

This allows optimizing on the manifold using standard gradient descent in the
Euclidean parameterization:

.. math::

    \mathbf{v}_{t+1} = \mathbf{v}_t - \eta \nabla_{\mathbf{v}} (f \circ \phi^{-1})(\mathbf{v}_t)

In SPD Learn:

* **Stiefel parameters** (BiMap weights) are represented in Euclidean space and
  mapped to the Stiefel manifold through polar decomposition:
  :math:`\phi^{-1}(\mathbf{X}) = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1/2}`
* **SPD parameters** (e.g., batch norm bias) are represented in Euclidean space
  and mapped to SPD using matrix exponential trivialization.

This allows standard optimizers (SGD, Adam) to train models without explicit
Riemannian solvers, while preserving manifold constraints by construction.

**Correspondence to CNNs**:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - CNN Component
     - SPDNet Analog
     - Function
   * - Conv layer
     - BiMap
     - Feature extraction/dimension change
   * - ReLU
     - ReEig
     - Nonlinearity
   * - Flatten
     - LogEig + Vech
     - Trivialization to vector
   * - FC layer
     - Linear
     - Classification


Batch Normalization
===================

SPD Learn provides :class:`~spd_learn.modules.SPDBatchNormMean` and
:class:`~spd_learn.modules.SPDBatchNormMeanVar`. These layers normalize SPD-valued
features while preserving geometric structure and are central to domain
adaptation models such as TSMNet.


Example: Building an SPDNet
===========================

.. code-block:: python

    import torch
    import torch.nn as nn
    from spd_learn.modules import BiMap, ReEig, LogEig, SPDBatchNormMean


    class ManualSPDNet(nn.Module):
        """SPDNet built from individual layers."""

        def __init__(self, n_channels=32, n_classes=4):
            super().__init__()

            # SPD dimension reduction pipeline
            self.bimap1 = BiMap(n_channels, n_channels // 2)
            self.reeig1 = ReEig()
            self.bn1 = SPDBatchNormMean(n_channels // 2)

            self.bimap2 = BiMap(n_channels // 2, n_channels // 4)
            self.reeig2 = ReEig()

            # Trivialization to tangent space
            self.logeig = LogEig(upper=True, flatten=True)

            # Euclidean classifier
            out_dim = (n_channels // 4) * (n_channels // 4 + 1) // 2
            self.classifier = nn.Linear(out_dim, n_classes)

        def forward(self, x):
            # x: (batch, n_channels, n_channels) SPD matrices
            x = self.reeig1(self.bimap1(x))
            x = self.bn1(x)
            x = self.reeig2(self.bimap2(x))
            x = self.logeig(x)  # Trivialization
            return self.classifier(x)


    # Usage
    model = ManualSPDNet(n_channels=32, n_classes=4)
    spd_batch = torch.randn(16, 32, 32)
    spd_batch = spd_batch @ spd_batch.mT + 0.1 * torch.eye(32)
    output = model(spd_batch)
    print(f"Output shape: {output.shape}")  # (16, 4)

Or use the pre-built :class:`~spd_learn.models.SPDNet` model directly.


Where to Go Next
================

* :doc:`/user_guide` for a step-by-step model walkthrough
* :doc:`/api` for full layer and function references
