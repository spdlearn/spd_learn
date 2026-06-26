=============
API Reference
=============


:py:mod:`spd_learn`:

.. automodule:: spd_learn
   :no-members:
   :no-inherited-members:

SPD Learn follows a **functional-first** design philosophy: low-level operations
are implemented as pure functions in :py:mod:`spd_learn.functional`, which are
then wrapped into stateful layers in :py:mod:`spd_learn.modules`, and finally
composed into complete architectures in :py:mod:`spd_learn.models`.

.. contents:: API Sections
   :local:
   :depth: 2


Functional
==========

:py:mod:`spd_learn.functional`:

.. automodule:: spd_learn.functional
   :no-members:
   :no-inherited-members:

This module provides low-level functions for operations on tensors, particularly
those representing SPD matrices or elements in related manifolds. These functions
form the core computational backend for the layers in :py:mod:`spd_learn.modules`
and models in :py:mod:`spd_learn.models`.

.. currentmodule:: spd_learn.functional


Core Matrix Operations
----------------------
Basic matrix operations commonly used in Riemannian geometry, such as matrix
logarithm, exponential, power functions, and utilities for ensuring symmetry
or clamping eigenvalues.

.. autosummary::
   :toctree: generated/functional

   matrix_log
   matrix_exp
   matrix_softplus
   matrix_inv_softplus
   softplus
   inv_softplus
   matrix_power
   matrix_sqrt
   matrix_inv_sqrt
   matrix_sqrt_inv
   clamp_eigvals
   abs_eigvals
   ensure_sym
   orthogonal_polar_factor


Covariance Estimation
---------------------
Functions for computing various types of covariance matrices from input data.

.. autosummary::
   :toctree: generated/covariance_functional

   covariance
   sample_covariance
   real_covariance
   cross_covariance


Regularization
--------------
Functional regularization utilities for covariance matrices.

.. autosummary::
   :toctree: generated/regularization_functional

   trace_normalization
   ledoit_wolf
   shrinkage_covariance


.. _api-riemannian-metrics:

Riemannian Metrics
------------------

SPD Learn implements four Riemannian metrics for SPD manifolds. Each metric
provides distance computation, geodesic interpolation, and exponential/logarithmic
maps. Choose based on your application's needs:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Properties
     - Best For
   * - **AIRM**
     - Affine-invariant, curvature-aware
     - Theoretical correctness, domain adaptation
   * - **Log-Euclidean**
     - Bi-invariant, closed-form mean
     - Fast computation, deep learning
   * - **Log-Cholesky**
     - Avoids eigendecomposition
     - Numerical stability, large matrices
   * - **Bures-Wasserstein**
     - Optimal transport connection
     - Covariance interpolation, statistics


AIRM (Affine-Invariant Riemannian Metric)
'''''''''''''''''''''''''''''''''''''''''
The AIRM is the canonical Riemannian metric on SPD manifolds with affine invariance
properties. It provides geodesic distances and interpolations that are invariant
under congruence transformations.

.. autosummary::
   :toctree: generated/airm

   airm_distance
   airm_geodesic
   exp_map_airm
   log_map_airm
   spd_egrad2rgrad


Log-Euclidean Metric
''''''''''''''''''''
The Log-Euclidean Metric (LEM) maps SPD matrices to a flat (Euclidean) space
via matrix logarithm, enabling efficient closed-form computations.

.. autosummary::
   :toctree: generated/log_euclidean

   log_euclidean_distance
   log_euclidean_geodesic
   log_euclidean_mean
   log_euclidean_multiply
   log_euclidean_scalar_multiply
   exp_map_lem
   log_map_lem


Log-Cholesky Metric
'''''''''''''''''''
The Log-Cholesky metric uses the Cholesky decomposition to parameterize SPD
matrices, avoiding expensive eigendecompositions while maintaining numerical
stability.

.. autosummary::
   :toctree: generated/log_cholesky

   cholesky_log
   cholesky_exp
   log_cholesky_distance
   log_cholesky_mean
   log_cholesky_geodesic


Bures-Wasserstein Metric
''''''''''''''''''''''''
The Bures-Wasserstein (or Procrustes) metric has connections to optimal transport
theory and is particularly useful for covariance interpolation.

.. autosummary::
   :toctree: generated/bures_wasserstein

   bures_wasserstein_distance
   bures_wasserstein_geodesic
   bures_wasserstein_mean
   bures_wasserstein_transport


Parallel Transport
------------------
Functions for parallel transport of tangent vectors along geodesics on the SPD
manifold, essential for operations like domain adaptation.

.. autosummary::
   :toctree: generated/parallel_transport

   parallel_transport_airm
   parallel_transport_lem
   parallel_transport_log_cholesky
   schild_ladder
   pole_ladder
   transport_tangent_vector


Fréchet Derivatives
-------------------
Directional (Fréchet) derivatives of the matrix logarithm and exponential,
used for parallel transport under the Log-Euclidean metric.

.. autosummary::
   :toctree: generated/frechet_functional

   frechet_derivative_log
   frechet_derivative_exp


Adversarial Robustness
----------------------
Riemannian projected gradient descent (PGD) attack for SPD-matrix inputs.

.. autosummary::
   :toctree: generated/attacks_functional

   spd_rpgd_attack


Vectorization Utilities
-----------------------
Vectorization helpers for batching, (un)vectorizing matrices, and symmetric matrix
encodings.

.. autosummary::
   :toctree: generated/vectorization

   vec_batch
   unvec_batch
   sym_to_upper
   vec_to_sym


Dropout
-------
Functional implementation of dropout specifically designed for SPD tensors.

.. autosummary::
   :toctree: generated/dropout_functional

   dropout_spd


Autograd Helpers
----------------
Custom forward and backward functions for operations like matrix eigen-decomposition,
enabling gradient computation through these potentially complex steps.

.. autosummary::
   :toctree: generated/autograd

   modeig_backward
   modeig_forward


Batch Normalization Operations
------------------------------
Functions for Riemannian batch normalization computations on SPD manifolds.

.. autosummary::
   :toctree: generated/batchnorm_functional

   frechet_mean
   karcher_mean_iteration
   spd_centering
   spd_rebiasing
   spd_cholesky_congruence
   tangent_space_variance
   lie_group_variance


Bilinear Operations
-------------------
Bilinear transformations that preserve SPD properties.

.. autosummary::
   :toctree: generated/bilinear_functional

   bimap_transform
   bimap_increase_dim


Wavelet Operations
------------------
Time-frequency analysis using Gabor wavelets.

.. autosummary::
   :toctree: generated/wavelet_functional

   compute_gabor_wavelet


Numerical Stability
-------------------
Utilities for ensuring numerical stability when working with SPD matrices,
including epsilon handling and eigenvalue clamping. See :doc:`/numerical_stability`
for detailed guidance.

.. autosummary::
   :toctree: generated/numerical

   numerical_config
   NumericalConfig
   NumericalContext
   get_epsilon
   get_epsilon_tensor
   get_loewner_threshold
   safe_clamp_eigenvalues
   check_spd_eigenvalues
   is_half_precision
   recommend_dtype_for_spd


pyRiemann Geometry (re-exported)
--------------------------------
Since pyRiemann (>= 0.12) is a core dependency and runs on PyTorch tensors via
the Python Array API, its broader geometry toolkit is re-exported from
:py:mod:`spd_learn.functional`, so the full SPD/Riemannian API is reachable from
a single namespace. These run on torch tensors (with autograd) like the native
functions above.

.. note::

   The low-level matrix-function primitives (``logm``/``expm``/``sqrtm``/
   ``invsqrtm``/``powm``/``ddlogm``/``ddexpm``) are intentionally **not**
   re-exported. Use spd_learn's numerically-stable :func:`matrix_log`,
   :func:`matrix_exp`, :func:`matrix_sqrt`, ... and :func:`frechet_derivative_log` /
   :func:`frechet_derivative_exp`, which keep gradients stable near degenerate
   eigenvalues. The re-exported distances/means/maps inherit pyRiemann's generic
   ``eigh`` backward (see :doc:`/numerical_stability`).

Additional distances:

.. autosummary::
   :toctree: generated/pyriemann_distance

   distance
   distance_chol
   distance_euclid
   distance_harmonic
   distance_kullback
   distance_kullback_right
   distance_kullback_sym
   distance_logchol
   distance_logdet
   distance_logeuclid
   distance_mahalanobis
   distance_poweuclid
   distance_riemann
   distance_thompson
   distance_wasserstein
   pairwise_distance

Additional geodesics:

.. autosummary::
   :toctree: generated/pyriemann_geodesic

   geodesic
   geodesic_chol
   geodesic_euclid
   geodesic_logchol
   geodesic_logeuclid
   geodesic_riemann
   geodesic_thompson
   geodesic_wasserstein

Additional means and barycenters:

.. autosummary::
   :toctree: generated/pyriemann_mean

   gmean
   maskedmean_riemann
   mean_ale
   mean_alm
   mean_bmp
   mean_cheap
   mean_chol
   mean_euclid
   mean_harmonic
   mean_kullback_sym
   mean_logchol
   mean_logdet
   mean_logeuclid
   mean_power
   mean_poweuclid
   mean_riemann
   mean_thompson
   mean_wasserstein
   nanmean_riemann

Tangent space, inner products, and transport:

.. autosummary::
   :toctree: generated/pyriemann_tangent

   exp_map
   exp_map_euclid
   exp_map_logchol
   exp_map_logeuclid
   exp_map_riemann
   exp_map_wasserstein
   innerproduct
   innerproduct_euclid
   innerproduct_logchol
   innerproduct_logeuclid
   innerproduct_riemann
   log_map
   log_map_euclid
   log_map_logchol
   log_map_logeuclid
   log_map_riemann
   log_map_wasserstein
   norm
   tangent_space
   transport
   transport_euclid
   transport_logchol
   transport_logeuclid
   transport_riemann
   untangent_space
   unupper
   upper

Kernels:

.. autosummary::
   :toctree: generated/pyriemann_kernel

   kernel
   kernel_euclid
   kernel_logeuclid
   kernel_riemann

Approximate joint diagonalization:

.. autosummary::
   :toctree: generated/pyriemann_ajd

   ajd
   ajd_pham
   rjd
   uwedge

Matrix utilities:

.. autosummary::
   :toctree: generated/pyriemann_base

   nearest_sym_pos_def


Modules
=======

:py:mod:`spd_learn.modules`:

.. automodule:: spd_learn.modules
   :no-members:
   :no-inherited-members:

This module provides neural network layers specifically designed for deep learning
on Riemannian manifolds, particularly SPD matrices. These layers wrap the
functional operations from :py:mod:`spd_learn.functional` into stateful
``torch.nn.Module`` components.

.. currentmodule:: spd_learn.modules


Covariance Layer
----------------
Modules for computing covariance matrices, often used as the first step in
SPD-based pipelines.

.. autosummary::
   :toctree: generated/covariance

   CovLayer


Matrix Eigen-Operations
-----------------------
Modules performing operations based on matrix eigenvalue decomposition (LogEig,
ReEig, ExpEig), essential for mapping between the SPD manifold and Euclidean/tangent
spaces or applying non-linearities.

.. autosummary::
   :toctree: generated/modeig

   LogEig
   ReEig
   ExpEig


Manifold Parametrization
------------------------
Modules for parametrizing learnable SPD matrices, ensuring parameters remain
on the SPD manifold during optimization. Supports both matrix exponential and
softplus mappings from symmetric matrices to SPD matrices.

.. autosummary::
   :toctree: generated/manifold

   SymmetricPositiveDefinite
   PositiveDefiniteScalar


Bilinear Mappings
-----------------
Layers implementing learnable bilinear transformations suitable for SPD matrices,
acting as analogous operations to linear layers in Euclidean space.

.. autosummary::
   :toctree: generated/bilinear

   BiMap
   BiMapIncreaseDim


Batch Normalization
-------------------
Batch normalization layers specifically adapted for data on the SPD manifold
or related representations.

.. autosummary::
   :toctree: generated/batchnorm

   SPDBatchNormMean
   SPDBatchNormMeanVar
   SPDBatchNormLie
   BatchReNorm


Regularization
--------------
Modules implementing regularization covariance methods, such as Ledoit-Wolf
shrinkage for covariance estimation or trace normalization.

.. autosummary::
   :toctree: generated/regularization

   TraceNorm
   Shrinkage


Dropout
-------
Dropout mechanisms designed for SPD matrix inputs or features derived from them.

.. autosummary::
   :toctree: generated/dropout

   SPDDropout


Residual Connections
--------------------
Modules for residual/skip connections on Riemannian manifolds, enabling deeper
SPD networks with improved gradient flow.

.. autosummary::
   :toctree: generated/residual

   LogEuclideanResidual


Signal Processing
-----------------
Layers for signal processing operations, including learnable wavelet convolutions
for time-frequency feature extraction.

.. autosummary::
   :toctree: generated/signal

   WaveletConv


Utilities
---------
Utility layers for preprocessing, feature extraction, or other auxiliary tasks
within Riemannian deep learning models.

.. autosummary::
   :toctree: generated/utils

   PatchEmbeddingLayer
   Vec
   Vech


Models
======

:py:mod:`spd_learn.models`:

.. automodule:: spd_learn.models
   :no-members:
   :no-inherited-members:

This module offers pre-built models for working with SPD matrices, using the
building blocks from :py:mod:`spd_learn.modules`.

.. currentmodule:: spd_learn.models

.. autosummary::
   :toctree: generated/

    EEGSPDNet
    Green
    MAtt
    PhaseSPDNet
    SPDNet
    TensorCSPNet
    TSMNet


Model Building Blocks
---------------------
Sub-modules used inside the models above, exposed for reuse and customization.

.. autosummary::
   :toctree: generated/

   AttentionManifold
   PhaseDelay


Model Selection Guide
---------------------

Use this table to choose the right model for your application:

.. list-table::
   :header-rows: 1
   :widths: 15 20 25 20 20

   * - Model
     - Best For
     - Input Type
     - Key Feature
     - Complexity
   * - :class:`SPDNet`
     - General SPD learning
     - Covariance matrices
     - Foundational architecture
     - Low
   * - :class:`TensorCSPNet`
     - Multi-band EEG
     - Filter bank data
     - Temporal-spectral-spatial
     - Medium
   * - :class:`TSMNet`
     - Domain adaptation, Interpretation
     - Raw EEG
     - SPDBatchNorm
     - Medium
   * - :class:`EEGSPDNet`
     - Channel-specific EEG
     - Raw EEG
     - Grouped convolution
     - Medium
   * - :class:`MAtt`
     - Attention-based
     - Raw EEG
     - Manifold attention
     - High
   * - :class:`Green`
     - Interpretable features
     - Raw EEG
     - Learnable wavelets
     - Medium
   * - :class:`PhaseSPDNet`
     - Nonlinear dynamics
     - Raw EEG
     - Phase-space embedding
     - Low


Decision Flowchart
''''''''''''''''''

.. code-block:: text

   START
     │
     ▼
   ┌─────────────────────────────────┐
   │ What is your input data type?  │
   └─────────────────────────────────┘
     │
     ├─── Pre-computed covariance matrices ──► SPDNet
     │
     ├─── Filter bank EEG (multiple bands) ──► TensorCSPNet
     │
     └─── Raw time series
           │
           ▼
         ┌─────────────────────────────────┐
         │ Do you need domain adaptation? │
         └─────────────────────────────────┘
           │
           ├─── Yes ──► TSMNet (with domain adaptation)
           │
           └─── No
                 │
                 ▼
               ┌─────────────────────────────────┐
               │ What is your priority?         │
               └─────────────────────────────────┘
                 │
                 ├─── Interpretability ──► TSMNet, Green
                 │
                 ├─── Attention mechanism ──► MAtt
                 │
                 ├─── Conv feature extraction ──► TSMNet, EEGSPDNet
                 │
                 └─── Nonlinear dynamics ──► PhaseSPDNet


Model Architectures
'''''''''''''''''''

**SPDNet** - Foundational architecture for SPD learning:

.. code-block:: text

   [CovLayer] → BiMap → ReEig → LogEig → Linear

**TensorCSPNet** - Multi-frequency EEG with temporal-spectral-spatial features:

.. code-block:: text

   Tensor Stacking → BiMap blocks → SPDBatchNormMean → LogEig → TCN → Linear

**TSMNet** - Domain adaptation with SPD batch normalization:

.. code-block:: text

   Conv2d → CovLayer → BiMap → ReEig → SPDBatchNormMeanVar → LogEig → Linear

**EEGSPDNet** - Channel-specific temporal filtering:

.. code-block:: text

   GroupedConv1d → CovLayer → BiMap/SPDDropout/ReEig blocks → LogEig → Linear

**MAtt** - Attention-based temporal weighting:

.. code-block:: text

   Conv2d → PatchCov → AttentionManifold → ReEig → LogEig → Linear

**Green** - Interpretable wavelet features:

.. code-block:: text

   WaveletConv → CovLayer → Shrinkage → BiMap → LogEig → BatchReNorm → MLP

**PhaseSPDNet** - Phase-space embedding for nonlinear dynamics:

.. code-block:: text

   PhaseDelay → SPDNet


Performance Comparison
''''''''''''''''''''''

Based on motor imagery classification benchmarks (BNCI2014-001):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Model
     - Accuracy (%)
     - Parameters
     - Training Time
     - GPU Memory
   * - SPDNet
     - 70-75
     - ~10K
     - Fast
     - Low
   * - TensorCSPNet
     - 75-82
     - ~50K
     - Medium
     - Medium
   * - TSMNet
     - 72-78
     - ~30K
     - Medium
     - Medium
   * - EEGSPDNet
     - 73-79
     - ~40K
     - Medium
     - Medium
   * - MAtt
     - 74-80
     - ~60K
     - Slow
     - High
   * - Green
     - 72-78
     - ~20K
     - Medium
     - Low

*Note: Performance varies significantly across subjects and datasets.*

To reproduce these results and run your own benchmarks, please refer to the :ref:`sphx_glr_generated_auto_examples_applied_examples_moabb_hydra_benchmark.py` example.


Initialization
==============

:py:mod:`spd_learn.init`:

.. automodule:: spd_learn.init
   :no-members:
   :no-inherited-members:

This module provides functions to initialize tensors with SPD-specific methods,
following PyTorch's ``torch.nn.init`` pattern. All functions operate in-place
and return the modified tensor for convenience.

.. currentmodule:: spd_learn.init

.. autosummary::
   :toctree: generated/init

   stiefel_
   spd_identity_
