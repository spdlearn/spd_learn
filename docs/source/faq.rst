.. _faq:

==========================
Frequently Asked Questions
==========================

This page answers common questions about SPD Learn and working with SPD matrices.

.. contents:: Contents
   :local:
   :depth: 2


General Questions
=================

What is an SPD matrix?
----------------------

A **Symmetric Positive Definite (SPD)** matrix is a square matrix :math:`X` that satisfies:

1. **Symmetry**: :math:`X = X^\top`
2. **Positive definiteness**: :math:`z^\top X z > 0` for all non-zero vectors :math:`z`

Equivalently, all eigenvalues of an SPD matrix are strictly positive.

**Common examples**:

- Covariance matrices of multivariate data
- Correlation matrices
- Diffusion tensors in medical imaging
- Kernel matrices in machine learning


Why use SPD-specific neural networks?
-------------------------------------

SPD matrices lie on a **Riemannian manifold**, not a flat Euclidean space.
Standard neural network operations (like addition, scaling) can produce matrices
that are no longer SPD, leading to:

- Numerical instability (negative eigenvalues)
- Loss of geometric structure
- Suboptimal learning

SPD Learn's layers (:class:`~spd_learn.modules.BiMap`, :class:`~spd_learn.modules.ReEig`, :class:`~spd_learn.modules.LogEig`) are designed to:

- **Preserve positive definiteness** throughout the network
- **Respect the manifold geometry** for more principled learning
- **Enable stable gradient computation** through eigenvalue operations


How is SPD Learn different from pyRiemann?
------------------------------------------

**pyRiemann** focuses on classical Riemannian methods:

- Riemannian classifiers (MDM, FgMDM)
- Tangent space projections
- Geodesic operations
- scikit-learn compatible transformers

**SPD Learn** focuses on deep learning:

- Neural network layers for SPD manifolds
- End-to-end differentiable architectures
- GPU acceleration via PyTorch
- Integration with modern DL frameworks (Braindecode, skorch)

They are **complementary**: you can use pyRiemann for preprocessing and
SPD Learn for deep learning.


Technical Questions
===================

How do I ensure my input is SPD?
--------------------------------

If starting from raw covariance estimates, they may not be strictly SPD due to:

- Numerical precision issues
- Insufficient samples
- Rank deficiency

**Solutions**:

1. **Add regularization** (Ledoit-Wolf shrinkage):

   .. code-block:: python

      from spd_learn.modules import Shrinkage

      shrinkage = Shrinkage(n_chans=64, init_shrinkage=0.1, learnable=True)
      X_reg = shrinkage(X)

2. **Use ReEig to clamp eigenvalues**:

   .. code-block:: python

      from spd_learn.modules import ReEig

      reeig = ReEig(threshold=1e-4)
      X_spd = reeig(X)

3. **Add small identity matrix**:

   .. code-block:: python

      X_spd = X + 1e-5 * torch.eye(X.shape[-1])


Why am I getting NaN values during training?
--------------------------------------------

NaN values typically occur due to:

1. **Negative eigenvalues**: Use :class:`~spd_learn.modules.ReEig` with appropriate threshold

   .. code-block:: python

      reeig = ReEig(threshold=1e-4)  # Increase threshold if still getting NaNs

2. **Eigenvalue decomposition instability**: Enable autograd mode

   .. code-block:: python

      logeig = LogEig(autograd=True)  # LogEig with more stable backward pass

3. **Learning rate too high**: Reduce learning rate

   .. code-block:: python

      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

4. **Poorly conditioned matrices**: Add regularization or shrinkage


How do I handle different matrix sizes?
---------------------------------------

If you have matrices of varying sizes (e.g., different channel counts):

1. **Pad smaller matrices**:

   .. code-block:: python

      import torch.nn.functional as F

      # Pad to max_size
      X_padded = F.pad(X, (0, max_size - X.shape[-1], 0, max_size - X.shape[-2]))

2. **Use** :class:`~spd_learn.modules.BiMapIncreaseDim` to expand dimensions:

   .. code-block:: python

      from spd_learn.modules import BiMapIncreaseDim

      expand = BiMapIncreaseDim(in_features=16, out_features=32)

3. **Create separate models** for different sizes


What's the difference between LogEig and tangent space projection?
------------------------------------------------------------------

Both map SPD matrices to Euclidean space, but:

**LogEig** (matrix logarithm):

- Maps to the space of symmetric matrices
- :math:`\logeig(X) = U \log(\Lambda) U^\top`
- Preserves more structure, but requires eigendecomposition

**Tangent space projection** (at identity or reference):

- Projects to tangent space at a reference point
- :math:`\Log{\I}(X) = \log(X)` (at identity)
- :math:`\Log{\frechet}(X) = \frechet^{-1/2} \log(\frechet^{-1/2} X \frechet^{-1/2}) \frechet^{-1/2}` (at :math:`\frechet`)

In SPD Learn, :class:`~spd_learn.modules.LogEig` implements the matrix logarithm. For tangent space
at a reference (like batch mean), use :class:`~spd_learn.modules.SPDBatchNormMeanVar` first.


Model-Specific Questions
========================

When should I use filter banks vs. raw EEG?
-------------------------------------------

**Use filter banks** (TensorCSPNet) when:

- You know relevant frequency bands (e.g., mu/beta for motor imagery)
- You want explicit frequency decomposition
- You have sufficient data for the larger model

**Use raw EEG** (EEGSPDNet, GREEN, MAtt) when:

- You want the model to learn frequency representations
- You prefer end-to-end learning
- You have limited prior knowledge about discriminative frequencies


How do I choose BiMap dimensions?
---------------------------------

The BiMap layer reduces dimensionality: ``(n, n) → (m, m)`` where ``m < n``.

**Guidelines**:

- Start with ``m = n // 2`` (50% reduction)
- For small datasets: more aggressive reduction (prevent overfitting)
- For large datasets: preserve more dimensions
- Multiple BiMap layers: gradual reduction (e.g., 64 → 32 → 16)

.. code-block:: python

   # Single layer
   model = SPDNet(n_chans=64, subspacedim=32)

   # Progressive reduction in EEGSPDNet
   model = EEGSPDNet(n_chans=22, bimap_sizes=(2, 3))  # 220→110→55→27


How does SPDBatchNormMeanVar enable domain adaptation?
------------------------------------------------------

:class:`~spd_learn.modules.SPDBatchNormMeanVar` maintains running statistics (Fréchet mean) that can be
domain-specific:

1. **Training**: Learns normalization from source domain
2. **Adaptation**: Updates running mean on target domain (unlabeled)
3. **Inference**: Uses adapted statistics

This enables **Source-Free Unsupervised Domain Adaptation (SFUDA)** where
you adapt to a new domain without source data.


Performance Questions
=====================

How can I speed up training?
----------------------------

1. **Use GPU**:

   .. code-block:: python

      model = model.cuda()
      X = X.cuda()

2. **Increase batch size** (if memory allows):

   .. code-block:: python

      batch_size = 64  # or higher

3. **Use autograd=False** for :class:`~spd_learn.modules.LogEig` (faster but less stable):

   .. code-block:: python

      logeig = LogEig(autograd=False)

4. **Reduce model complexity**:

   .. code-block:: python

      model = SPDNet(n_chans=64, subspacedim=16)  # Smaller subspace


How much GPU memory do SPD models need?
---------------------------------------

Memory scales with:

- **Matrix size**: :math:`O(n^2)` for ``n x n`` matrices
- **Batch size**: Linear scaling
- **Eigendecomposition**: Requires temporary storage

**Typical requirements** (22-channel EEG, batch_size=32):

- SPDNet: ~500 MB
- TensorCSPNet: ~1 GB
- MAtt: ~1.5 GB

**Reduce memory**:

.. code-block:: python

   # Gradient checkpointing
   from torch.utils.checkpoint import checkpoint

   # Mixed precision (PyTorch 2.0+)
   with torch.autocast("cuda"):
       output = model(X)


Why is my model not learning?
-----------------------------

Common issues:

1. **Data not SPD**: Verify inputs are positive definite

   .. code-block:: python

      eigvals = torch.linalg.eigvalsh(X)
      assert (eigvals > 0).all(), "Not SPD!"

2. **Learning rate**: Try different rates (1e-3, 1e-4, 1e-5)

3. **Initialization**: Check :class:`~spd_learn.modules.BiMap` weights are orthogonal

4. **Regularization**: Add dropout or weight decay

   .. code-block:: python

      optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4)

5. **Data preprocessing**: Normalize/standardize inputs


Integration Questions
=====================

How do I use SPD Learn with MOABB?
----------------------------------

.. code-block:: python

   from moabb.datasets import BNCI2014_001
   from moabb.paradigms import MotorImagery
   from braindecode import EEGClassifier
   from spd_learn.models import SPDNet

   # Load data
   dataset = BNCI2014_001()
   paradigm = MotorImagery()
   X, y, meta = paradigm.get_data(dataset, subjects=[1])

   # Create classifier
   clf = EEGClassifier(
       SPDNet(n_chans=22, n_outputs=4),
       criterion=torch.nn.CrossEntropyLoss,
       optimizer=torch.optim.Adam,
       batch_size=32,
       max_epochs=100,
   )

   clf.fit(X, y)


How do I use SPD Learn with Nilearn?
------------------------------------

.. code-block:: python

   from nilearn.connectome import ConnectivityMeasure
   from spd_learn.models import SPDNet

   # Compute connectivity matrices
   conn = ConnectivityMeasure(kind="covariance")
   X = conn.fit_transform(time_series_list)  # (n_subjects, n_rois, n_rois)

   # Convert to tensor
   X_tensor = torch.tensor(X, dtype=torch.float32)

   # Create model for pre-computed covariance
   model = SPDNet(n_chans=X.shape[1], n_outputs=2, input_type="cov")


Can I use SPD Learn with scikit-learn pipelines?
------------------------------------------------

Yes, via Braindecode's ``EEGClassifier`` or custom wrappers:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from braindecode import EEGClassifier

   pipe = Pipeline([("clf", EEGClassifier(SPDNet(n_chans=22, n_outputs=4)))])

   # Cross-validation
   from sklearn.model_selection import cross_val_score

   scores = cross_val_score(pipe, X, y, cv=5)


Getting Help
============

If your question isn't answered here:

1. Check the :doc:`user_guide` for conceptual explanations
2. Check the :doc:`api` for detailed function documentation
3. Browse the :doc:`generated/auto_examples/index` for code examples
4. Open an issue on `GitHub <https://github.com/spdlearn/spd_learn/issues>`_
