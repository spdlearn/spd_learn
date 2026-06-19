.. _installation:

============
Installation
============

SPD Learn requires Python 3.11 or higher and is built on PyTorch 2.0+.

Quick Install
-------------

The simplest way to install SPD Learn is via pip:

.. code-block:: bash

   pip install spd_learn

This will install the core package with minimal dependencies.

Dependencies
------------

**Core dependencies** (installed automatically):

- `PyTorch <https://pytorch.org/>`_ - Deep learning framework
- `einops <https://einops.rocks/>`_ - Flexible tensor operations
- `NumPy <https://numpy.org/>`_ - Numerical computing
- `pyRiemann <https://pyriemann.readthedocs.io/>`_ (>= 0.12) - SPD/Riemannian
  geometry. SPD Learn delegates geodesics, parallel transport, and Fréchet
  derivatives to pyRiemann's Array API backend (which runs natively on PyTorch
  tensors) and re-exports pyRiemann's broader toolkit. It keeps its own
  numerically-stable, custom-autograd matrix-function primitives
  (``matrix_log``/``matrix_exp``/``matrix_sqrt``/...), distances, and means,
  which stay finite/stable on ill-conditioned inputs — important for training.

.. note::

   The Array API (PyTorch) backend lives in pyRiemann **0.12+**, which is not
   yet on PyPI. SPD Learn therefore pins pyRiemann to its git ``master``::

       pyriemann @ git+https://github.com/pyRiemann/pyRiemann.git

   Because this is a direct-URL dependency, ``pip install spd_learn`` from PyPI
   is unavailable until pyRiemann 0.12 is released — install from source (see
   below) meanwhile. The pin becomes ``pyriemann>=0.12`` once 0.12 ships.

Installing from Source
----------------------

For the latest development version, you can install directly from GitHub:

.. code-block:: bash

   git clone https://github.com/spdlearn/spd_learn.git
   cd spd_learn
   pip install -e .

Optional Dependencies
---------------------

SPD Learn provides optional dependency groups for different use cases:

Brain-Computer Interface (BCI) & Neuroimaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For EEG and fMRI applications:

.. code-block:: bash

   pip install "spd_learn[brain]"

This installs:

- `Braindecode <https://braindecode.org/>`_ - Deep learning for EEG
- `Nilearn <https://nilearn.github.io/>`_ - Machine learning for neuroimaging

(pyRiemann is now a **core** dependency — see above — so it is always installed.)

Documentation
^^^^^^^^^^^^^

To build the documentation locally:

.. code-block:: bash

   pip install "spd_learn[docs]"

Testing
^^^^^^^

For running the test suite:

.. code-block:: bash

   pip install "spd_learn[tests]"

All Dependencies
^^^^^^^^^^^^^^^^

To install all optional dependencies:

.. code-block:: bash

   pip install "spd_learn[all]"

Verifying Installation
----------------------

After installation, verify that SPD Learn is working correctly:

.. code-block:: python

   import torch
   from spd_learn.models import SPDNet

   # Create SPDNet for 16-channel data with 2 output classes
   model = SPDNet(
       n_chans=16,
       n_outputs=2,
       subspacedim=8,
       input_type="cov",  # Input is pre-computed covariance
   )

   # Generate random SPD matrix (must be positive definite)
   X = torch.randn(4, 16, 16)
   X = X @ X.transpose(-1, -2) + 0.1 * torch.eye(16)  # Make SPD

   # Forward pass
   output = model(X)
   print(f"Input shape: {X.shape}")  # (4, 16, 16)
   print(f"Output shape: {output.shape}")  # (4, 2)
   print("Installation successful!")

Development Installation
------------------------

For contributing to SPD Learn:

.. code-block:: bash

   git clone https://github.com/spdlearn/spd_learn.git
   cd spd_learn
   pip install -e ".[all]"

This installs the package in editable mode with all dependencies for development, testing, and documentation.

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**ImportError: No module named 'spd_learn'**

Make sure you've installed the package:

.. code-block:: bash

   pip install spd_learn

**CUDA out of memory**

Try reducing batch size or using CPU:

.. code-block:: python

   model = model.cpu()
   X = X.cpu()

**Numerical instability with eigenvalue decomposition**

Use the ReEig layer with an appropriate threshold:

.. code-block:: python

   from spd_learn.modules import ReEig

   # Add small threshold to prevent numerical issues
   reeig = ReEig(threshold=1e-4)

Getting Help
^^^^^^^^^^^^

- Check the :doc:`API documentation <api>`
- Browse the :doc:`examples <generated/auto_examples/index>`
- Open an issue on `GitHub <https://github.com/spdlearn/spd_learn/issues>`__
