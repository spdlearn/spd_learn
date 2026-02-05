.. _numerical_stability:

====================
Numerical Stability
====================

SPD operations involve eigendecomposition, matrix logarithms, and other
operations that can be numerically sensitive. SPD Learn provides a
comprehensive configuration system for managing numerical stability.

This document describes both the theoretical foundations and practical
techniques for working with Symmetric Positive Definite (SPD) matrices
in Riemannian geometry-based learning frameworks. The theoretical approaches
are based on established methods in the field, including those described
in the MENDR framework :cite:p:`chen2025mendr`.

.. contents:: Contents
   :local:
   :depth: 2


Overview
========

Working with SPD matrices presents numerical challenges:

1. **Small eigenvalues**: Operations like :math:`\log(\lambda)` become undefined or
   unstable when eigenvalues approach zero.

2. **Condition number**: Ill-conditioned matrices (large ratio of max/min eigenvalues)
   cause precision loss in matrix operations.

3. **Gradient computation**: The Loewner matrix formulation requires careful handling
   of equal or nearly-equal eigenvalues.

4. **Mixed precision**: Half-precision (float16/bfloat16) training requires larger
   stability margins.

SPD Learn addresses these challenges with **dtype-aware numerical thresholds**
that automatically adjust based on the precision of your computations.

.. seealso::

   :doc:`geometric_concepts` for the mathematical foundations of SPD matrices,
   eigendecomposition, Riemannian metrics, and geometric operations.


Covariance Matrix Rank Deficiency
=================================

The sample covariance matrix (SCM) from a data matrix :math:`\mathbf{X} \in \reals^{C \times T}`
(C channels, T time samples) is computed as:

.. math::

    \mathbf{SCM} = \frac{1}{T-1} \mathbf{X} \mathbf{X}^\top

This matrix is guaranteed to be symmetric positive semi-definite, but may have
zero eigenvalues if :math:`T < C`. This rank deficiency is a common source of
numerical instability when applying SPD operations that require strictly positive
eigenvalues.


Numerical Stability Techniques
==============================

Trace Normalization with Epsilon Regularization
-----------------------------------------------

To ensure numerical stability during forward and backward passes, a two-stage
regularization is applied:

**Stage 1 - Pre-normalization regularization:**

.. math::

    \mathbf{SCM}_{\text{reg}} = \mathbf{SCM} + \epsilon \mathbf{I}

**Stage 2 - Trace normalization with post-regularization:**

.. math::

    \mathbf{SCM}_{\text{norm}} = \frac{\mathbf{SCM}_{\text{reg}}}{\text{tr}(\mathbf{SCM}_{\text{reg}})} + \epsilon \mathbf{I}

where :math:`\epsilon = 10^{-5}` is a typical choice and :math:`\I` is the identity matrix.

**Rationale:**

- Pre-normalization :math:`\epsilon` prevents division by near-zero traces
- Trace normalization ensures bounded eigenvalues
- Post-normalization :math:`\epsilon` guarantees minimum eigenvalue :math:`\lambda_{\min} \geq \epsilon`

Symmetrization
--------------

Floating-point operations can introduce small asymmetries. Explicit symmetrization
ensures the SPD property:

.. math::

    \mathbf{X}_{\text{sym}} = \frac{\mathbf{X} + \mathbf{X}^\top}{2}

This should be applied after any operation that might introduce asymmetry.

Cholesky Decomposition for Gradient Flow
----------------------------------------

Instead of directly optimizing over the SPD manifold, a numerically stable
approach is to parameterize SPD matrices via Cholesky decomposition:

.. math::

    \mathbf{M} = \mathbf{L} \mathbf{L}^\top

where :math:`\mathbf{L} \in \reals^{n \times n}` is a learnable lower triangular
matrix with positive diagonal elements.

**Gradient computation via the product rule:**

.. math::

    d\mathbf{M} = (d\mathbf{L}) \mathbf{L}^\top + \mathbf{L} (d\mathbf{L})^\top

This parameterization:

- Guarantees SPD output without explicit manifold projections
- Enables use of standard Euclidean optimizers (Adam, SGD)
- Provides stable gradient flow through the decomposition

SVD-Based Stable Differentiation
--------------------------------

For operations requiring eigendecomposition (log, exp, power), using SVD provides
numerical stability:

.. math::

    \mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^\top

For symmetric matrices, :math:`\mathbf{U} = \mathbf{V}`, and the singular values
equal the absolute eigenvalues.

**Gradient decomposition:**

The gradient flow through SVD can be decomposed into:

1. **Diagonal component**: Gradients with respect to singular values
2. **Off-diagonal component**: Gradients with respect to singular vectors

The orthonormality constraints on :math:`\mathbf{U}` and :math:`\mathbf{V}` provide
natural regularization and prevent gradient explosion.

Logarithmic Loss Functions
--------------------------

Operating on log-eigenvalues rather than raw eigenvalues prevents underflow/overflow:

**Masked Autoencoding Loss:**

.. math::

    \mathcal{L}_{\text{MAE}} = \| \log(\boldsymbol{\lambda}_{\text{masked}}) - \log(\hat{\boldsymbol{\lambda}}_{\text{masked}}) \|^2

where :math:`\boldsymbol{\lambda}` denotes the vector of eigenvalues.

**Benefits:**

- Logarithmic scaling compresses the dynamic range
- Equal relative errors contribute equally to the loss
- Prevents gradient explosion from large eigenvalue differences


Condition Number Analysis
=========================

Definition and Significance
---------------------------

The condition number of an SPD matrix is:

.. math::

    \kappa(\mathbf{A}) = \frac{\lambda_{\max}}{\lambda_{\min}}

A high condition number indicates ill-conditioning, where:

- Small perturbations in input lead to large perturbations in output
- Numerical errors are amplified during matrix operations
- Gradient-based optimization becomes unstable

Condition Number Bounds
-----------------------

After trace normalization with :math:`\epsilon`-regularization:

.. math::

    \kappa(\mathbf{A}_{\text{norm}}) \leq \frac{1}{\epsilon}

For :math:`\epsilon = 10^{-5}`, this bounds the condition number at :math:`10^5`.

**Practical recommendation:** Choose :math:`\epsilon` to balance:

- Larger :math:`\epsilon`: Better numerical stability, but information loss
- Smaller :math:`\epsilon`: Preserves information, but risk of instability

Eigenvalue Perturbation Bounds
------------------------------

For a symmetric matrix :math:`\mathbf{A}` with perturbation :math:`\mathbf{E}`:

**Weyl's Theorem:**

.. math::

    |\lambda_i(\mathbf{A} + \mathbf{E}) - \lambda_i(\mathbf{A})| \leq \|\mathbf{E}\|_2

This bounds how much eigenvalues can change due to numerical errors bounded by
:math:`\|\mathbf{E}\|_2`.


Configuration System
====================

The Global Configuration
------------------------

SPD Learn provides a global ``numerical_config`` object that controls all
numerical stability thresholds:

.. code-block:: python

   from spd_learn.functional import numerical_config

   # View current settings
   print(numerical_config)

   # Modify a threshold
   numerical_config.eigval_clamp_scale = 1e5  # More conservative clamping

   # Disable warnings
   numerical_config.warn_on_clamp = False


Configuration Parameters
------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - :attr:`~spd_learn.functional.NumericalConfig.eigval_clamp_scale`
     - 1e4
     - Scale for ReEig layer eigenvalue clamping
   * - :attr:`~spd_learn.functional.NumericalConfig.eigval_log_scale`
     - 1e2
     - Scale for matrix logarithm stability
   * - :attr:`~spd_learn.functional.NumericalConfig.eigval_sqrt_scale`
     - 1e2
     - Scale for matrix square root
   * - :attr:`~spd_learn.functional.NumericalConfig.eigval_inv_sqrt_scale`
     - 1e3
     - Scale for inverse square root
   * - :attr:`~spd_learn.functional.NumericalConfig.eigval_power_scale`
     - 1e3
     - Scale for matrix power operations
   * - :attr:`~spd_learn.functional.NumericalConfig.loewner_equal_scale`
     - 1e2
     - Scale for detecting equal eigenvalues
   * - :attr:`~spd_learn.functional.NumericalConfig.batchnorm_var_eps`
     - 1e-5
     - Epsilon for batch normalization scalar dispersion (absolute)
   * - :attr:`~spd_learn.functional.NumericalConfig.dropout_eps`
     - 1e-5
     - Epsilon for dropout diagonal entries (absolute)
   * - :attr:`~spd_learn.functional.NumericalConfig.warn_on_clamp`
     - True
     - Emit warnings when eigenvalues are clamped

**How thresholds are computed:**

.. code-block:: python

   threshold = scale * torch.finfo(dtype).eps

For example, with ``eigval_clamp_scale=1e4`` and ``dtype=torch.float32``:

.. code-block:: python

   threshold = 1e4 * 1.19e-7  # ≈ 1.19e-3


Getting Epsilon Values
----------------------

Use ``get_epsilon()`` to retrieve the appropriate threshold for an operation:

.. code-block:: python

   from spd_learn.functional import get_epsilon
   import torch

   # Get epsilon for eigenvalue clamping in float32
   eps32 = get_epsilon(torch.float32, "eigval_clamp")
   print(f"float32 clamp threshold: {eps32:.2e}")  # ~1.19e-3

   # Get epsilon for float64 (tighter threshold)
   eps64 = get_epsilon(torch.float64, "eigval_clamp")
   print(f"float64 clamp threshold: {eps64:.2e}")  # ~2.22e-12

   # Get epsilon for float16 (much larger threshold)
   eps16 = get_epsilon(torch.float16, "eigval_clamp")
   print(f"float16 clamp threshold: {eps16:.2e}")  # ~9.77e0


Temporary Configuration
-----------------------

Use ``NumericalContext`` to temporarily modify settings:

.. code-block:: python

   from spd_learn.functional import NumericalContext, get_epsilon
   import torch

   # Default threshold
   print(f"Default: {get_epsilon(torch.float32, 'eigval_clamp'):.2e}")

   # Temporarily use more conservative threshold
   with NumericalContext(eigval_clamp_scale=1e6):
       print(f"Conservative: {get_epsilon(torch.float32, 'eigval_clamp'):.2e}")

   # Back to default
   print(f"Restored: {get_epsilon(torch.float32, 'eigval_clamp'):.2e}")


Checking SPD Validity
=====================

Use ``check_spd_eigenvalues()`` to validate matrices:

.. code-block:: python

   from spd_learn.functional import check_spd_eigenvalues
   import torch

   # Create a matrix and check its eigenvalues
   A = torch.randn(3, 3, dtype=torch.float32)
   A = A @ A.T + 0.1 * torch.eye(3)
   eigvals = torch.linalg.eigvalsh(A)

   is_valid, min_val, num_bad = check_spd_eigenvalues(eigvals)
   print(f"Valid: {is_valid}, Min eigenvalue: {min_val:.2e}")

   # Optionally raise an error
   check_spd_eigenvalues(eigvals, raise_on_failure=True)


Safe Eigenvalue Clamping
------------------------

Use ``safe_clamp_eigenvalues()`` for consistent clamping, important for
activation functions like :class:`~spd_learn.modules.ReEig`:

.. code-block:: python

   from spd_learn.functional import safe_clamp_eigenvalues
   import torch

   eigvals = torch.tensor([1e-10, 1e-5, 1e-3, 1.0])

   # Clamp with dtype-aware threshold
   clamped = safe_clamp_eigenvalues(eigvals, "eigval_log")
   print(clamped)  # Small values will be clamped

   # Also get mask of which values were clamped
   clamped, mask = safe_clamp_eigenvalues(eigvals, "eigval_log", return_mask=True)
   print(f"Clamped values at indices: {mask.nonzero().squeeze()}")


Implementation Guidelines
=========================

Recommended Numerical Thresholds
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Typical Value
     - Purpose
   * - :math:`\epsilon_{\text{reg}}`
     - :math:`10^{-5}`
     - Diagonal regularization
   * - :math:`\epsilon_{\text{log}}`
     - :math:`10^{-7}`
     - Logarithm argument floor
   * - :math:`\epsilon_{\text{div}}`
     - :math:`10^{-8}`
     - Division stability
   * - Max condition number
     - :math:`10^{6}`
     - Ill-conditioning threshold

Stability Checklist
-------------------

When implementing SPD matrix operations:

1. **Always regularize** before computing logarithms
2. **Symmetrize** after any matrix operation that might introduce asymmetry
3. **Check eigenvalues** in debug mode to detect near-singular matrices
4. **Use double precision** (float64) when possible for intermediate computations
5. **Clip eigenvalues** to :math:`[\epsilon, \infty)` before taking logarithms
6. **Monitor condition numbers** during training

Code Example
------------

Pseudocode for stable SPD operations::

    def stable_log_euclidean_mean(matrices, eps=1e-5):
        """Compute Log-Euclidean mean with numerical stability."""
        log_sum = 0
        for A in matrices:
            # Regularize
            A_reg = A + eps * eye(n)
            # Symmetrize
            A_sym = (A_reg + A_reg.T) / 2
            # Compute stable log
            eigvals, eigvecs = eigh(A_sym)
            eigvals = maximum(eigvals, eps)  # Clip eigenvalues
            log_A = eigvecs @ diag(log(eigvals)) @ eigvecs.T
            log_sum += log_A

        # Compute mean in tangent space
        log_mean = log_sum / len(matrices)

        # Map back to manifold
        eigvals, eigvecs = eigh(log_mean)
        mean = eigvecs @ diag(exp(eigvals)) @ eigvecs.T

        return (mean + mean.T) / 2  # Final symmetrization


Recommendations for Different Scenarios
=======================================

Standard Training (float32)
---------------------------

The default settings are well-tuned for float32 training:

.. code-block:: python

   # No changes needed for most cases
   from spd_learn.models import SPDNet

   model = SPDNet(n_chans=64, n_outputs=4)


Ill-Conditioned Matrices
------------------------

For matrices with high condition numbers (common in EEG/fMRI):

.. code-block:: python

   from spd_learn.functional import numerical_config

   # Use more conservative clamping
   numerical_config.eigval_clamp_scale = 1e5
   numerical_config.eigval_log_scale = 1e3

   # Or consider using float64
   model = model.double()


Mixed Precision Training
------------------------

For float16/bfloat16 training, be more conservative:

.. code-block:: python

   from spd_learn.functional import numerical_config, recommend_dtype_for_spd

   # Check if float16 is appropriate
   condition_number = 1e6  # Estimated from your data
   recommended = recommend_dtype_for_spd(condition_number)
   print(f"Recommended dtype: {recommended}")

   # If using float16, increase scales
   numerical_config.eigval_clamp_scale = 1e6
   numerical_config.eigval_log_scale = 1e4


High Precision Requirements
---------------------------

For research or when maximum precision is needed:

.. code-block:: python

   import torch
   from spd_learn.functional import numerical_config

   # Use float64
   model = model.double()

   # Use tighter thresholds
   numerical_config.eigval_clamp_scale = 1e2
   numerical_config.eigval_log_scale = 1e1


Common Issues and Solutions
===========================

NaN Values During Training
--------------------------

**Symptom**: Loss becomes NaN after some epochs.

**Cause**: Usually due to eigenvalues becoming too small or negative.

**Solution**:

.. code-block:: python

   from spd_learn.functional import numerical_config

   # 1. Enable warnings to see when clamping occurs
   numerical_config.warn_on_clamp = True

   # 2. Use more conservative thresholds
   numerical_config.eigval_clamp_scale = 1e5

   # 3. Add regularization to your covariance matrices
   from spd_learn.modules import TraceNorm

   trace_norm = TraceNorm(eps=1e-4)


Slow Convergence
----------------

**Symptom**: Model trains but converges slowly or gets stuck.

**Cause**: Overly conservative thresholds may clip important information.

**Solution**:

.. code-block:: python

   # Try tighter thresholds if your data is well-conditioned
   numerical_config.eigval_clamp_scale = 1e3

   # Check condition numbers of your data
   import torch

   cond_numbers = []
   for batch in dataloader:
       cov = compute_covariance(batch)
       eigvals = torch.linalg.eigvalsh(cov)
       cond = eigvals.max() / eigvals.min()
       cond_numbers.append(cond)
   print(f"Median condition number: {torch.median(torch.stack(cond_numbers))}")


Warnings About Eigenvalue Clamping
----------------------------------

**Symptom**: Many warnings about eigenvalue clamping.

**Cause**: Your data has small eigenvalues being modified.

**Options**:

.. code-block:: python

   # Option 1: Disable warnings if this is expected
   numerical_config.warn_on_clamp = False

   # Option 2: Preprocess data with regularization
   from spd_learn.modules import Shrinkage

   shrinkage = Shrinkage(alpha=0.1)  # Ledoit-Wolf shrinkage

   # Option 3: Use higher precision
   model = model.double()


API Reference
=============

.. autofunction:: spd_learn.functional.get_epsilon
   :no-index:

.. autofunction:: spd_learn.functional.get_epsilon_tensor
   :no-index:

.. autofunction:: spd_learn.functional.safe_clamp_eigenvalues
   :no-index:

.. autofunction:: spd_learn.functional.check_spd_eigenvalues
   :no-index:

.. autofunction:: spd_learn.functional.get_loewner_threshold
   :no-index:

.. autofunction:: spd_learn.functional.is_half_precision
   :no-index:

.. autofunction:: spd_learn.functional.recommend_dtype_for_spd
   :no-index:

.. autoclass:: spd_learn.functional.NumericalConfig
   :members:
   :no-index:

.. autoclass:: spd_learn.functional.NumericalContext
   :members:
   :no-index:


References
==========

.. bibliography::
   :filter: docname in docnames


.. seealso::

   - :doc:`faq` — Troubleshooting common issues
   - :doc:`user_guide` — Getting started with SPD Learn
   - :doc:`theory` — Theory section overview
   - :doc:`geometric_concepts` — Understanding SPD geometry
