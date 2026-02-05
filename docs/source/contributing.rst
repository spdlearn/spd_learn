.. _contributing:

============
Contributing
============

Thank you for your interest in contributing to SPD Learn! This guide will help
you get started with contributing to the project.

.. contents:: Contents
   :local:
   :depth: 2


Getting Started
===============

Setting Up Development Environment
----------------------------------

1. **Fork and clone the repository**:

   .. code-block:: bash

      git clone https://github.com/spdlearn/spd_learn.git
      cd spd_learn

2. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install in development mode with all dependencies**:

   .. code-block:: bash

      pip install -e ".[all]"

4. **Install pre-commit hooks** (required for consistent formatting):

   .. code-block:: bash

      pip install pre-commit
      pre-commit install
      pre-commit run --all-files


Development Workflow
====================

Creating a Branch
-----------------

Create a new branch for your feature or bug fix:

.. code-block:: bash

   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix

Running Tests
-------------

Run the test suite to ensure your changes don't break existing functionality:

.. code-block:: bash

   pytest

Run tests with coverage:

.. code-block:: bash

   pytest --cov=spd_learn --cov-report=html

Code Style
----------

We use ``ruff`` for linting and ``black`` for formatting. Run before committing:

.. code-block:: bash

   ruff check spd_learn/
   black spd_learn/


Types of Contributions
======================

Bug Reports
-----------

If you find a bug, please open an issue on GitHub with:

1. A clear, descriptive title
2. Steps to reproduce the bug
3. Expected behavior vs. actual behavior
4. Your environment (Python version, PyTorch version, OS)
5. Minimal code example that reproduces the issue

Feature Requests
----------------

We welcome feature requests! Please open an issue describing:

1. The problem you're trying to solve
2. Your proposed solution
3. Any alternatives you've considered

Code Contributions
------------------

Pull Requests
^^^^^^^^^^^^^

1. **Create an issue first** for significant changes
2. **Write tests** for new functionality
3. **Update documentation** if needed
4. **Follow the code style** guidelines
5. **Keep PRs focused** - one feature/fix per PR

PR Checklist
^^^^^^^^^^^^

Before submitting a PR, ensure:

- [ ] All tests pass (``pytest``)
- [ ] Code is formatted (``black``, ``ruff``)
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] ``pre-commit run --all-files`` passes


Adding New Features
===================

Adding a New Layer
------------------

To add a new neural network layer:

1. Create the layer in the appropriate file under ``spd_learn/modules/``
2. Add comprehensive docstrings following NumPy format
3. Export in ``spd_learn/modules/__init__.py``
4. Add to ``docs/source/api.rst``
5. Write unit tests in ``tests/``

Example layer structure:

.. code-block:: python

   import torch
   import torch.nn as nn


   class MyNewLayer(nn.Module):
       """Short description of the layer.

       Longer description explaining what the layer does,
       its mathematical formulation, and when to use it.

       Parameters
       ----------
       in_features : int
           Input dimension.
       out_features : int
           Output dimension.

       References
       ----------
       .. [1] Author, A. (Year). Paper Title. Journal.

       Examples
       --------
       >>> layer = MyNewLayer(64, 32)
       >>> x = torch.randn(16, 64, 64)
       >>> output = layer(x)
       >>> output.shape
       torch.Size([16, 32, 32])
       """

       def __init__(self, in_features: int, out_features: int):
           super().__init__()
           self.in_features = in_features
           self.out_features = out_features
           # Initialize parameters...

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Forward pass.

           Parameters
           ----------
           x : torch.Tensor
               Input tensor of shape ``(batch, in_features, in_features)``.

           Returns
           -------
           torch.Tensor
               Output tensor of shape ``(batch, out_features, out_features)``.
           """
           # Implementation...
           return x

Adding a New Model
------------------

To add a new model architecture:

1. Create the model in ``spd_learn/models/``
2. Include a docstring with:

   - Architecture description
   - Figure reference (if available)
   - All parameters documented
   - Original paper reference

3. Export in ``spd_learn/models/__init__.py``
4. Add to ``docs/source/api.rst``
5. Consider adding an example script

Adding a Functional Operation
-----------------------------

For low-level operations:

1. Add to ``spd_learn/functional/``
2. Implement both forward and backward passes if using custom autograd
3. Export in ``spd_learn/functional/__init__.py``
4. Document in ``docs/source/api.rst``


Documentation
=============

Building Documentation
----------------------

Build the documentation locally:

.. code-block:: bash

   cd docs
   make html

View the built documentation:

.. code-block:: bash

   open build/html/index.html  # macOS
   xdg-open build/html/index.html  # Linux

Writing Documentation
---------------------

- Use **NumPy-style docstrings** for all public functions and classes
- Include **examples** in docstrings where helpful
- Add **references** to papers when implementing published methods
- Update the **User Guide** for conceptual changes


Code Organization
=================

Architecture Overview
---------------------

SPD Learn follows PyTorch's design philosophy with a clear separation between
**functional operations** and **stateful modules**. This architecture enables
flexibility, composability, and ease of extension.

.. code-block:: text

                    ┌─────────────────────────────────────────┐
                    │              models/                     │
                    │  Pre-built architectures (SPDNet, etc.) │
                    │  Compose modules into complete models   │
                    └──────────────────┬──────────────────────┘
                                       │ uses
                    ┌──────────────────▼──────────────────────┐
                    │              modules/                    │
                    │  Stateful nn.Module layers (BiMap, etc.)│
                    │  Learnable parameters + forward()       │
                    └──────────────────┬──────────────────────┘
                                       │ calls
                    ┌──────────────────▼──────────────────────┐
                    │             functional/                  │
                    │  Pure functions (matrix_log, etc.)       │
                    │  Custom autograd for SPD operations      │
                    └─────────────────────────────────────────┘

Design Principles
^^^^^^^^^^^^^^^^^

SPD Learn follows PyTorch's design philosophy :cite:p:`paszke2019pytorch` with a
clear separation between stateless operations and stateful modules. This pattern,
also adopted by JAX :cite:p:`frostig2018compiling`, reflects classical systems
design principles :cite:p:`saltzer1984end`, :cite:p:`waldo1994note`.

1. **Functional First**: Mathematical operations are implemented as pure functions
   in ``spd_learn.functional``. No internal state, usable outside neural networks.

2. **Modules Wrap Functional**: ``nn.Module`` classes in ``spd_learn.modules``
   wrap functional operations, adding learnable parameters and state management.

3. **Models Compose Modules**: Complete architectures in ``spd_learn.models``
   compose modules into end-to-end trainable networks.

4. **Trivialization for Constraints**: Manifold-valued parameters (Stiefel, SPD)
   use trivialization :cite:p:`lezcano2019trivializations`—mappings from
   unconstrained space to the manifold—enabling standard optimizers.

Package Structure
-----------------

.. code-block:: text

   spd_learn/
   ├── __init__.py          # Package initialization and public API
   ├── version.py           # Version string
   ├── logging.py           # Logging utilities
   │
   ├── functional/          # Pure functions (no learnable parameters)
   │   ├── __init__.py      # Public functional API
   │   ├── autograd.py      # Custom autograd: modeig_forward/backward
   │   ├── core.py          # Core spectral operations (matrix_log, matrix_exp, etc.)
   │   ├── covariance.py    # Covariance estimators
   │   ├── metrics/         # Riemannian metrics subpackage
   │   │   ├── __init__.py  # Metrics API aggregation
   │   │   ├── airm.py      # Affine-Invariant Riemannian Metric
   │   │   ├── log_euclidean.py  # Log-Euclidean Metric
   │   │   ├── bures_wasserstein.py  # Bures-Wasserstein Metric
   │   │   └── log_cholesky.py  # Log-Cholesky Metric
   │   ├── transport.py     # Parallel transport operations
   │   ├── numerical.py     # Numerical stability configuration
   │   ├── regularize.py    # Shrinkage estimators
   │   ├── dropout.py       # SPD-aware dropout
   │   └── utils.py         # Helper functions (ensure_sym, etc.)
   │
   ├── modules/             # nn.Module layers (stateful, learnable)
   │   ├── __init__.py      # Public modules API
   │   ├── bilinear.py      # BiMap, BiMapIncreaseDim (Stiefel-constrained)
   │   ├── modeig.py        # LogEig, ReEig, ExpEig (spectral layers)
   │   ├── covariance.py    # CovLayer (covariance pooling)
   │   ├── batchnorm.py     # SPDBatchNormMeanVar, SPDBatchNormMean
   │   ├── dropout.py       # SPDDropout
   │   ├── regularize.py    # Shrinkage, TraceNorm
   │   ├── wavelet.py       # WaveletConv (Gabor wavelets)
   │   ├── manifold.py      # SPD parametrization helpers
   │   ├── residual.py      # Residual connections on SPD
   │   └── utils.py         # PatchEmbeddingLayer, Vec, Vech
   │
   └── models/              # Pre-built architectures
       ├── __init__.py      # Public models API
       ├── spdnet.py        # SPDNet (Huang et al., 2017)
       ├── tsmnet.py        # TSMNet (Kobler et al., 2022)
       ├── tensorcsp.py     # TensorCSPNet (Ju et al., 2022)
       ├── eegspdnet.py     # EEGSPDNet (Wilson et al., 2025)
       ├── green.py         # GREEN (Paillard et al., 2025)
       ├── matt.py          # MAtt (Pan et al., 2022)
       └── phase_spdnet.py  # PhaseSPDNet (Carrara et al., 2025)

Functional Layer (``spd_learn.functional``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The functional layer provides **pure functions** for SPD matrix operations.
These functions:

- Have **no internal state** or learnable parameters
- Are **differentiable** via custom autograd (when needed)
- Can be used **standalone** outside neural networks
- Follow the pattern: ``output = function(input, *args)``

**Key components:**

- **Spectral operations** (``functional.py``): ``matrix_log``, ``matrix_exp``,
  ``matrix_sqrt``, ``matrix_power`` — implemented via eigendecomposition with
  custom backward passes using the Loewner matrix formulation.

- **Autograd** (``autograd.py``): ``modeig_forward`` and ``modeig_backward``
  implement the forward and backward passes for spectral matrix functions,
  caching eigendecompositions for efficient gradient computation.

- **Riemannian operations** (``metrics/``): ``airm_distance``,
  ``airm_geodesic``, ``log_euclidean_distance``, ``log_euclidean_mean`` for
  operations under various Riemannian metrics (AIRM, Log-Euclidean, etc.).

Example usage:

.. code-block:: python

   import torch
   from spd_learn.functional import matrix_log, matrix_exp, log_euclidean_mean

   # Create SPD matrices
   X = torch.randn(32, 16, 16)
   X = X @ X.mT + 0.1 * torch.eye(16)

   # Apply matrix logarithm (differentiable)
   log_X = matrix_log.apply(X)

   # Compute Log-Euclidean mean (unweighted)
   mean = matrix_exp.apply(log_X.mean(dim=0))

Modules Layer (``spd_learn.modules``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The modules layer provides **stateful ``nn.Module`` classes** that wrap
functional operations. These modules:

- **Inherit from** ``torch.nn.Module``
- **Manage learnable parameters** (weights, biases)
- **Handle manifold constraints** via trivialization
- Follow the pattern: ``output = module(input)``

**Key components:**

- **BiMap** (``bilinear.py``): Bilinear mapping ``Y = W^T X W`` where ``W`` is
  constrained to the Stiefel manifold via ``torch.nn.utils.parametrizations.orthogonal``.

- **LogEig/ReEig/ExpEig** (``modeig.py``): Spectral layers that apply functions
  to eigenvalues. ``ReEig`` rectifies eigenvalues (non-linearity), ``LogEig``
  maps to tangent space, ``ExpEig`` maps back to manifold.

- **SPDBatchNormMeanVar** (``batchnorm.py``): Batch normalization on the SPD manifold
  using the Fréchet mean and variance, with learnable SPD bias via trivialization.

- **CovLayer** (``covariance.py``): Computes covariance matrices from time series,
  supporting various estimators (sample, shrinkage, real-valued).

Example of trivialization in BiMap:

.. code-block:: python

   from torch.nn.utils import parametrizations


   class BiMap(nn.Module):
       def __init__(self, in_features, out_features):
           super().__init__()
           # Raw unconstrained parameter
           self.weight = nn.Parameter(torch.empty(in_features, out_features))
           # Apply orthogonal parametrization (Stiefel constraint)
           parametrizations.orthogonal(self, "weight")

       def forward(self, X):
           # W is automatically orthogonalized
           return self.weight.T @ X @ self.weight

Models Layer (``spd_learn.models``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The models layer provides **complete architectures** that compose modules into
end-to-end trainable networks. These models:

- **Accept raw data** (EEG signals, fMRI time series, or precomputed SPD matrices)
- **Output predictions** (class logits, features)
- Are **ready for training** with standard PyTorch optimizers

**Architecture pattern** (most models follow this):

.. code-block:: text

   Input → [CovLayer] → BiMap → ReEig → BiMap → ReEig → LogEig → Vec → Linear → Output
           (optional)   └────────── SPD Processing ──────────┘    └─ Classifier ─┘

Example model usage:

.. code-block:: python

   from spd_learn.models import SPDNet

   # Create model for 22-channel EEG, 4 classes
   model = SPDNet(n_chans=22, n_outputs=4, subspacedim=16)

   # Input: raw EEG (batch, channels, time)
   X = torch.randn(32, 22, 500)
   logits = model(X)  # (32, 4)

Trivialization Approach
-----------------------

SPD Learn uses **trivialization** to handle manifold-valued parameters, enabling
unconstrained optimization with standard gradient descent.

**What is trivialization?**

A trivialization is a smooth mapping ``Φ: ℝ^d → M`` from an unconstrained
Euclidean space to the target manifold ``M``. Instead of optimizing directly on
the manifold (which requires Riemannian optimization), we optimize the
unconstrained parameters and map them to the manifold.

**Stiefel manifold** (orthogonal matrices):

For BiMap weights ``W ∈ St(n, m)`` (matrices with orthonormal columns):

.. code-block:: python

   # PyTorch's built-in orthogonal parametrization
   parametrizations.orthogonal(module, "weight")

   # Internally uses Cayley map or matrix exponential:
   # W = cayley(A) where A is skew-symmetric

**SPD manifold** (positive definite matrices):

For SPD parameters (e.g., bias in SPDBatchNormMeanVar):

.. code-block:: python

   # Map symmetric matrix S to SPD via matrix exponential
   # X = exp(S), where S is unconstrained symmetric
   X = matrix_exp(S)  # Always SPD for any symmetric S

This approach:

- Enables **standard optimizers** (Adam, SGD) without modification
- Provides **automatic constraint satisfaction** by construction
- Supports **stable gradients** via the chain rule through the mapping

Naming Conventions
------------------

- **Modules**: ``CamelCase`` (e.g., :class:`~spd_learn.modules.BiMap`, :class:`~spd_learn.modules.SPDBatchNormMeanVar`)
- **Functions**: ``snake_case`` (e.g., :func:`~spd_learn.functional.matrix_log`, :func:`~spd_learn.functional.log_euclidean_mean`)
- **Private methods**: prefix with ``_`` (e.g., ``_compute_mean``)
- **Type hints**: Use throughout for clarity and IDE support


Testing Guidelines
==================

Test Structure
--------------

Tests are organized in ``tests/`` mirroring the package structure:

.. code-block:: text

   tests/
   ├── test_functional.py
   ├── test_modules.py
   └── test_models.py

Writing Tests
-------------

- Test both **forward and backward passes**
- Test **edge cases** (empty batches, single samples)
- Test **numerical stability** with extreme values
- Use **parametrized tests** for multiple configurations

Example test:

.. code-block:: python

   import pytest
   import torch
   from spd_learn.modules import BiMap


   @pytest.mark.parametrize(
       "in_features,out_features",
       [
           (64, 32),
           (32, 16),
           (16, 8),
       ],
   )
   def test_bimap_output_shape(in_features, out_features):
       layer = BiMap(in_features, out_features)
       x = torch.randn(8, in_features, in_features)
       x = x @ x.transpose(-1, -2)  # Make SPD
       y = layer(x)
       assert y.shape == (8, out_features, out_features)


   def test_bimap_preserves_spd():
       layer = BiMap(32, 16)
       x = torch.randn(8, 32, 32)
       x = x @ x.transpose(-1, -2) + torch.eye(32)  # SPD
       y = layer(x)
       # Check positive definiteness via eigenvalues
       eigvals = torch.linalg.eigvalsh(y)
       assert (eigvals > 0).all()


Community
=========

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions
- **Discussions**: Questions and ideas

Thank you for contributing to SPD Learn!
