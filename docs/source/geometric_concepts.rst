:html_theme.sidebar_secondary.remove: true

.. _geometric_concepts:

==================
Geometric Concepts
==================

This page provides a comprehensive introduction to the geometric foundations
underlying Riemannian methods for EEG/MEG analysis and deep learning with
covariance matrices on the SPD (Symmetric Positive Definite) manifold.

.. contents:: Contents
   :local:
   :depth: 3


The SPD Manifold
================

What is an SPD Matrix?
----------------------

A **Symmetric Positive Definite (SPD)** matrix :math:`X \in \reals^{n \times n}`
satisfies two conditions:

1. **Symmetry**: :math:`X = X^\top`
2. **Positive Definiteness**: :math:`z^\top X z > 0` for all non-zero vectors :math:`z \in \reals^n`

Equivalently, an SPD matrix has all positive eigenvalues :cite:p:`bhatia2007positive`. The set of all
:math:`n \times n` SPD matrices is denoted :math:`\spd` :cite:p:`congedo2017riemannian`.

**Eigendecomposition:**

For any SPD matrix :math:`X`, there exists an orthogonal matrix :math:`U` such that:

.. math::

    X = U \, \text{diag}(\lambda_1, \ldots, \lambda_n) \, U^\top

where :math:`\lambda_i > 0` are the positive eigenvalues. This decomposition is
fundamental to computing matrix functions on SPD matrices.


Why SPD Matrices Form a Manifold, or why not a Vector Space?
------------------------------------------------------------

SPD matrices do **not** form a vector space because:

1. **Not closed under subtraction**: If :math:`A, B` are SPD, :math:`A - B`
   may not be SPD (positive definiteness can be violated).

2. **Not closed under negative scaling**: If :math:`A` is SPD,
   :math:`-A` is negative definite.

3. **The "swelling effect"**: The Euclidean mean of SPD matrices can have
   larger determinant than any of the original matrices, which is geometrically
   undesirable for covariance estimation.

Instead, SPD matrices form an **open cone** in the space of symmetric matrices.
This cone has a natural manifold structure with well-defined
notions of distance, geodesics, and curvature. These geometric properties require
specialized tools to work with, which is the focus of SPD Learn!


The SPD Cone (2x2 Example)
--------------------------

For :math:`2 \times 2` matrices, we can visualize the SPD cone. A symmetric
:math:`2 \times 2` matrix has three free parameters:

.. math::

   X = \begin{pmatrix} a & b \\ b & c \end{pmatrix}

The positive definiteness constraints are:

- :math:`a > 0` (first leading minor)
- :math:`ac - b^2 > 0` (determinant, second leading minor)

This defines an open cone in :math:`(a, b, c)` space, where the boundary
corresponds to singular (rank-deficient) matrices.

The interactive visualization below shows the SPD cone with sample EEG covariance
matrices plotted as points. The identity matrix serves as a reference point,
and the tangent space at the identity (the space of symmetric matrices) is shown
as a plane.

.. raw:: html

   <iframe src="_static/spd_manifold_eeg.html" width="100%" height="600px" style="border:none;"></iframe>

.. tip::

   Use your mouse to rotate, zoom, and explore the 3D visualization. Hover over
   points to see their details.


Tangent Spaces and Exponential Maps
===================================

Tangent Space at a Point
------------------------

At any point :math:`P` on the SPD manifold, the **tangent space**
:math:`\tangent{P}` can be identified with symmetric matrices :cite:p:`congedo2017riemannian`. This is a vector
space where we can perform standard linear algebra operations.

.. math::

   \tangentspd{P} \cong \sym = \{ S \in \reals^{n \times n} : S = S^\top \}

The tangent space at the identity :math:`\I` is particularly important because
many operations are simplified there.


The Exponential Map
-------------------

The **exponential map** :math:`\Exp{P}: \tangent{P} \to \manifold`
projects tangent vectors back onto the manifold. At the identity:

.. math::

   \Exp{\I}(S) = \exp(S)

where :math:`\exp` is the matrix exponential. This maps any symmetric matrix
to an SPD matrix, ensuring we stay on the manifold.

For a symmetric matrix :math:`S` with eigendecomposition :math:`S = U \Lambda U^\top`:

.. math::

    \exp(S) = U \, \text{diag}(\exp(\lambda_1), \ldots, \exp(\lambda_n)) \, U^\top

Since :math:`\exp(\lambda_i) > 0` for all real :math:`\lambda_i`, the result is
always SPD.


The Logarithmic Map
-------------------

The **logarithmic map** :math:`\Log{P}: \manifold \to \tangent{P}`
is the inverse, projecting from the manifold to the tangent space:

.. math::

   \Log{\I}(X) = \log(X)

For an SPD matrix :math:`X` with eigendecomposition :math:`X = U \Lambda U^\top`:

.. math::

    \log(X) = U \, \text{diag}(\log(\lambda_1), \ldots, \log(\lambda_n)) \, U^\top

.. warning::

    The matrix logarithm is only defined for SPD matrices. If any eigenvalue
    :math:`\lambda_i \leq 0`, the logarithm is undefined or complex-valued.

This is the key operation in SPD Learn's ``LogEig`` layer, which maps SPD
matrices to a vector space for classification.


Riemannian Metrics on SPD Manifolds
===================================

A **Riemannian metric** defines inner products on tangent spaces, enabling
us to measure distances and angles on the manifold. The space of SPD matrices
can be equipped with various Riemannian metrics, each leading to distinct
geometric structures. This section reviews four principal Riemannian metrics
that are widely used in the analysis and learning of SPD matrices.


Affine-Invariant Riemannian Metric (AIRM)
-----------------------------------------

The **Affine-Invariant Riemannian Metric** :cite:p:`pennec2006riemannian` endows the SPD manifold
with a geometry that is invariant under congruence transformations. Specifically,
for any non-singular matrix :math:`W \in \gl` and SPD matrices :math:`P, Q \in \spd`:

.. math::

   \dairm{WPW^\top}{WQW^\top} = \dairm{P}{Q}

**Riemannian inner product:** At :math:`P \in \spd`, for tangent vectors
:math:`v, w \in \tangentspd{P}`:

.. math::

   \gairm{P}(v, w) = \frobinner{P^{-1/2} v P^{-1/2}}{P^{-1/2} w P^{-1/2}}

**Geodesic distance:**

.. math::

   \dairm{A}{B} = \frob{\log(A^{-1/2} B A^{-1/2})}

**Geodesic (shortest path):**

.. math::

   \gamma(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}

.. note::

   **Extrapolation property:** Unlike many metrics, the AIRM geodesic
   :math:`\gamma(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}` remains SPD
   for all :math:`t \in \reals`, not just :math:`t \in [0, 1]`. This allows
   extrapolation beyond the endpoints, which can be useful for data augmentation
   or exploring the manifold structure.

**Distance to identity:** :math:`\dairm{P}{\I} = \frob{\log(P)}`

**Key Properties:**

- **Geodesically complete**: The SPD manifold with AIRM forms a Hadamard manifold
  (complete, simply connected with non-positive sectional curvature), guaranteeing
  unique geodesics between any two SPD matrices.
- **Boundary avoidance**: Geodesics between SPD matrices never reach singular matrices
  (zero eigenvalues are infinitely distant).
- **Affine-invariant**: :math:`d(GAG^\top, GBG^\top) = d(A, B)` for invertible :math:`G`.
- **Fréchet mean uniqueness**: The Fréchet mean of a finite set of SPD matrices always
  exists and is unique.
- **Computationally expensive**: Requires eigendecomposition; computing the exact
  Fréchet mean requires iterative solvers such as the Karcher flow.

.. code-block:: python

   from spd_learn.functional import (
       airm_distance,
       airm_geodesic,
   )

   # Distance between SPD matrices
   dist = airm_distance(A, B)

   # Geodesic interpolation (t=0 gives A, t=1 gives B)
   midpoint = airm_geodesic(A, B, t=0.5)

.. seealso::

   :func:`~spd_learn.functional.airm_distance`,
   :func:`~spd_learn.functional.airm_geodesic`


Properties of the Geometric Mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Riemannian (geometric) mean under AIRM satisfies all 10 axiomatic properties
established by :cite:t:`ando2004geometric`:

1. **Consistency with scalars:** Reduces to ordinary geometric mean for 1×1 matrices
2. **Joint homogeneity:** :math:`\geomean(\alpha P_1, \ldots, \alpha P_k) = \alpha \geomean(P_1, \ldots, P_k)`
3. **Permutation invariance:** Independent of matrix ordering
4. **Monotonicity:** If :math:`P_i \leq Q_i`, then :math:`\geomean(P_1, \ldots) \leq \geomean(Q_1, \ldots)`
5. **Continuity from above:** Continuous under decreasing sequences
6. **Congruence invariance:** :math:`\geomean(B^\top P_1 B, \ldots) = B^\top \geomean(P_1, \ldots) B`
7. **Joint concavity:** The mean function is jointly concave
8. **Self-duality:** :math:`\geomean(P_1^{-1}, \ldots, P_k^{-1}) = \geomean(P_1, \ldots, P_k)^{-1}`
9. **AGH inequality:** Harmonic mean :math:`\leq` Geometric mean :math:`\leq` Arithmetic mean
10. **Determinant identity:** :math:`\det \geomean = (\prod_i \det P_i)^{1/k}` (unweighted case)

**Riccati Equation Characterization:**

The geometric mean :math:`\geomean` of two SPD matrices :math:`P` and :math:`Q` is the unique
positive definite solution to the **Riccati equation**:

.. math::

   \geomean \, Q^{-1} \, \geomean = P

This characterization provides an algebraic interpretation of the geometric mean
and connects it to control theory.


Log-Euclidean Metric (LEM)
--------------------------

The **Log-Euclidean Metric** :cite:p:`arsigny2007geometric` simplifies computations by exploiting
the matrix logarithm to map the SPD manifold diffeomorphically to the Euclidean
vector space of symmetric matrices :math:`\syms`. The mapping
:math:`\log: \spd \to \syms` is a global diffeomorphism
(a smooth, invertible map with smooth inverse).

**Riemannian inner product:** The LEM is defined as the pullback of the Euclidean
metric through the logarithm. For any :math:`P \in \spd` and tangent
vectors :math:`v, w`:

.. math::

   \glem{P}(v, w) = \frobinner{D_P \log(v)}{D_P \log(w)}

where :math:`D_P \log` denotes the differential of the logarithm at :math:`P`.

**Distance:**

.. math::

   \dlem{A}{B} = \frob{\log(A) - \log(B)}

**Distance to identity:** :math:`\dlem{P}{\I} = \frob{\log(P)}` (same as AIRM at identity)

**Fréchet mean (closed-form):**

.. math::

   \bar{X} = \exp\left( \frac{1}{n} \sum_{i=1}^n \log(X_i) \right)

**Key Properties:**

- **Lie group structure**: Gives SPD matrices the structure of a commutative Lie group.
- **Computationally efficient**: Enables computations in the log-domain using standard
  Euclidean operations.
- **Not affine-invariant**: Unlike AIRM, LEM is only invariant under orthogonal
  transformations (rotations), not general affine transformations.
- **Closed-form mean**: The Fréchet mean can be computed directly without iteration.

.. code-block:: python

   from spd_learn.functional import (
       log_euclidean_distance,
       log_euclidean_mean,
   )

   # Distance
   dist = log_euclidean_distance(A, B)

   # Weighted mean (using uniform weights for unweighted mean)
   weights = torch.ones(batch_size, batch_size) / batch_size
   mean = log_euclidean_mean(weights, batch_of_spd_matrices)

.. seealso::

   :func:`~spd_learn.functional.log_euclidean_distance`,
   :func:`~spd_learn.functional.log_euclidean_mean`


Bures-Wasserstein Metric (BWM)
------------------------------

The **Bures-Wasserstein Metric** :cite:p:`bhatia2019bures` originates from quantum information
theory and optimal transport. It corresponds to the 2-Wasserstein distance between
centered Gaussian distributions.

**Riemannian inner product:** At :math:`P \in \spd`, for tangent matrices
:math:`V, W`:

.. math::

   \gbw{P}(V, W) = \tr(\lyap{P}[V] W)

where :math:`\lyap{P}` is the **Lyapunov operator** that assigns to each
:math:`V \in \syms` the unique solution :math:`X` of the Lyapunov equation:

.. math::

   PX + XP = V

**Distance:**

.. math::

   \dbw{A}{B}^2 = \tr(A) + \tr(B) - 2\tr\left((A^{1/2} B A^{1/2})^{1/2}\right)

**Distance to identity:** :math:`\dbw{P}{\I}^2 = \tr(P) + n - 2\tr(P^{1/2})`

**Geodesic:**

.. math::

   \gamma(t) = (1-t)^2 A + t^2 B + t(1-t)(M + M^\top)

where :math:`M = (A^{1/2} B A^{1/2})^{1/2}`.

**Key Properties:**

- **Positively curved**: Unlike AIRM (non-positive curvature), BWM endows
  :math:`\spd` with a positively curved Riemannian structure.
- **Optimal transport interpretation**: The distance equals the 2-Wasserstein distance
  between :math:`\mathcal{N}(0, A)` and :math:`\mathcal{N}(0, B)`.
- **Closed-form expressions**: Distances, geodesics, and Fréchet means have closed-form
  solutions (Fréchet means via fixed-point iteration).
- **No eigendecomposition**: Avoids eigenvalue decomposition, using matrix square roots.
- **Not affine-invariant**: Invariant only under unitary transformations.

.. code-block:: python

   from spd_learn.functional import bures_wasserstein_distance, bures_wasserstein_mean

   # Distance
   dist = bures_wasserstein_distance(A, B)

   # Fréchet mean (fixed-point iteration)
   mean = bures_wasserstein_mean(matrices, weights)

.. seealso::

   :func:`~spd_learn.functional.bures_wasserstein_distance`,
   :func:`~spd_learn.functional.bures_wasserstein_mean`,
   :func:`~spd_learn.functional.bures_wasserstein_geodesic`,
   :func:`~spd_learn.functional.bures_wasserstein_transport`


Log-Cholesky Metric (LCM)
-------------------------

The **Log-Cholesky Metric** :cite:p:`lin2019riemannian` builds upon the Cholesky decomposition
:math:`P = LL^\top`, where :math:`L` is a lower-triangular matrix with positive
diagonal entries. There exists a smooth bijection (diffeomorphism)
:math:`\varphi: \choleskyspace \to \spd`, where :math:`\choleskyspace`
denotes the **Cholesky space** of lower-triangular matrices with positive diagonals.

**Riemannian inner product:** At :math:`P = LL^\top \in \spd`:

.. math::

   \glcm{P}(v, w) = \bar{g}_L\left(L(L^{-1}vL^{-\top})_\triangle, L(L^{-1}wL^{-\top})_\triangle\right)

where :math:`(\cdot)_\triangle` extracts the lower-triangular part and scales diagonal
elements by :math:`\frac{1}{2}`. The metric :math:`\bar{g}_L` on :math:`\choleskyspace` is:

.. math::

   \bar{g}_L(X, Y) = \sum_{i>j} X_{ij}Y_{ij} + \sum_{j=1}^{n} X_{jj}Y_{jj}L_{jj}^{-2}

**Distance:**

.. math::

   \dlcm{A}{B} = \frob{\logchol(L_A) - \logchol(L_B)}

where :math:`\logchol(L) = \tril{L, -1} + \diag(\log(\diag(L)))`.

**Key Properties:**

- **Fastest computation**: Complexity :math:`O(n^3/3)` vs :math:`O(n^3)` for eigendecomposition.
- **Numerically stable**: Cholesky decomposition is well-conditioned for SPD matrices.
- **Globally flat geometry**: Inherits Euclidean structure from Cholesky space.
- **Closed-form geodesics and means**: No iterative optimization required.
- **Not affine-invariant**: Invariant under lower-triangular transformations with
  positive diagonal.
- **Ideal for optimization**: Avoids explicit matrix inversions and logarithms,
  yielding improved differentiability for deep learning.

.. code-block:: python

   from spd_learn.functional import log_cholesky_distance, log_cholesky_mean

   # Fast distance computation
   dist = log_cholesky_distance(A, B)

   # Closed-form mean
   mean = log_cholesky_mean(matrices)

.. seealso::

   :func:`~spd_learn.functional.log_cholesky_distance`,
   :func:`~spd_learn.functional.log_cholesky_mean`,
   :func:`~spd_learn.functional.log_cholesky_geodesic`,
   :class:`~spd_learn.functional.cholesky_log`,
   :class:`~spd_learn.functional.cholesky_exp`


Metric Comparison Summary
-------------------------

These four metrics capture distinct geometric perspectives on :math:`\spd`
and serve different computational and modeling goals:

.. list-table::
   :header-rows: 1
   :widths: 18 15 18 15 34

   * - Metric
     - Complexity
     - Invariance
     - Curvature
     - Best For
   * - **AIRM**
     - :math:`O(n^3)`
     - Full affine
     - Non-positive
     - Theoretical analysis, domain adaptation :cite:p:`zanini2017transfer`
   * - **Log-Euclidean**
     - :math:`O(n^3)`
     - Orthogonal
     - Flat
     - General use, closed-form mean
   * - **Bures-Wasserstein**
     - :math:`O(n^3)`
     - Unitary
     - Positive
     - Optimal transport, ill-conditioned matrices
   * - **Log-Cholesky**
     - :math:`O(n^3/3)`
     - Lower-triangular
     - Flat
     - Speed-critical, deep learning

**Choosing a metric:**

- Use **AIRM** when affine invariance is important (e.g., domain adaptation across
  subjects/sessions where the covariance scale may differ).
- Use **LEM** for general-purpose applications where a closed-form mean is desirable
  and affine invariance is not critical.
- Use **BWM** when working with ill-conditioned matrices or when an optimal transport
  interpretation is meaningful.
- Use **LCM** when computational speed is paramount or in deep learning where
  gradient stability is important


Invariance Properties
---------------------

Different metrics satisfy different invariance properties, which determine
their behavior under geometric transformations:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Metric
     - Rotation
     - Affinity
     - Inversion
     - Notes
   * - **AIRM**
     - Yes
     - Yes
     - Yes
     - Full invariance
   * - **Log-Euclidean**
     - Yes
     - No
     - Yes
     - Orthogonal only; inversion invariant
   * - **Bures-Wasserstein**
     - Yes
     - No
     - No
     - Unitary only
   * - **Log-Cholesky**
     - No
     - No
     - No
     - Lower-triangular only

**Definitions:**

- **Rotation invariance:** :math:`d(U^\top P U, U^\top Q U) = d(P, Q)` for orthogonal :math:`U`
- **Affinity (congruence) invariance:** :math:`d(B^\top P B, B^\top Q B) = d(P, Q)` for invertible :math:`B`
- **Inversion invariance:** :math:`d(P^{-1}, Q^{-1}) = d(P, Q)`


Parallel Transport
==================

**Parallel transport** moves tangent vectors between different tangent spaces
while preserving their geometric properties :cite:p:`zanini2017transfer`. This is essential for:

- Domain adaptation (transferring learned representations)
- Comparing tangent vectors at different reference points

Under the AIRM, parallel transport from :math:`\tangent{P}` to :math:`\tangent{Q}`:

.. math::

   \Gamma_{P \to Q}(V) = E \cdot V \cdot E^\top

where :math:`E = (Q P^{-1})^{1/2}`.

.. code-block:: python

   from spd_learn.functional import parallel_transport_airm

   # Transport tangent vector V from T_P to T_Q
   V_transported = parallel_transport_airm(V, P, Q)

.. seealso::

   :func:`~spd_learn.functional.parallel_transport_airm`


Trivialization
===============

When optimizing functions on manifolds (like the SPD manifold of covariance
matrices), we face a fundamental challenge: manifolds are curved spaces where
standard Euclidean gradient descent doesn't directly apply.

**Trivialization** is a technique that transforms manifold-constrained
optimization into unconstrained optimization by parametrizing the manifold
through its tangent space.

The animation below illustrates the concept of **Trivialization**
from :cite:t:`lezcano2019trivializations`:

.. only:: html

   .. image:: _static/images/dynamic_trivialization.gif
      :alt: Trivialization Animation
      :align: center
      :width: 100%

**Key concepts illustrated:**

1. **Manifold** :math:`\manifold` — The curved space where our data lives
   (e.g., SPD matrices representing EEG spatial covariance)

2. **Tangent Space** :math:`\tangent{p} \cong \reals^n` — A flat
   Euclidean approximation at point :math:`p`, where standard optimization
   algorithms can be applied

3. **Exponential Map** :math:`\phi_p` — Projects points from the tangent space
   back onto the manifold

4. **Dynamic Update** — When optimization moves too far from the base point,
   we update: :math:`p_{i+1} := \phi_{p_i}(y_{i,k})` and continue optimizing
   in the new tangent space





Practical Implications
----------------------

1. **Distance computation**: Use :func:`~spd_learn.functional.log_euclidean_distance`
   for speed, :func:`~spd_learn.functional.airm_distance` for affine
   invariance, :func:`~spd_learn.functional.bures_wasserstein_distance` for
   ill-conditioned matrices.

2. **Averaging**: Always use geometric means (:func:`~spd_learn.functional.log_euclidean_mean`,
   :func:`~spd_learn.functional.bures_wasserstein_mean`, or
   :func:`~spd_learn.functional.log_cholesky_mean`) instead of arithmetic means for SPD matrices.

3. **Classification**: Project to tangent space (:class:`~spd_learn.modules.LogEig`)
   before applying standard classifiers.

4. **Domain adaptation**: Use parallel transport
   (:func:`~spd_learn.functional.parallel_transport_airm`) to align representations
   across subjects or sessions.


SPD Layer Visualizations
========================

Understanding how SPD network layers transform data on the manifold is crucial
for building intuition about geometric deep learning. The visualizations below
show each layer's operation using 2x2 SPD matrices represented as ellipsoids.

:class:`~spd_learn.modules.CovLayer` — Transforms time series into SPD covariance matrices:

.. math::

    \Sigma = \frac{1}{T-1} (X - \bar{X})(X - \bar{X})^T

See :ref:`sphx_glr_generated_auto_examples_visualizations_plot_covlayer_animation.py`

:class:`~spd_learn.modules.BiMap` — Bilinear mapping that reduces/expands dimensionality:

.. math::

    Y = W^T X W

where :math:`W` is constrained to the Stiefel manifold (:math:`W^T W = I`).
See :ref:`sphx_glr_generated_auto_examples_visualizations_plot_bimap_animation.py`

:class:`~spd_learn.modules.ReEig` — Eigenvalue rectification (ReLU for SPD matrices):

.. math::

    \reeig(X) = U \max(\Lambda, \epsilon) U^\top

See :ref:`sphx_glr_generated_auto_examples_visualizations_plot_reeig_animation.py`

:class:`~spd_learn.modules.LogEig` — Projects SPD matrices to the tangent space:

.. math::

    \logeig(X) = U \log(\Lambda) U^\top

See :ref:`sphx_glr_generated_auto_examples_visualizations_plot_logeig_animation.py`

:class:`~spd_learn.modules.SPDBatchNormMeanVar` — Riemannian batch normalization:

.. math::

    \tilde{X}_i = \frechet^{-1/2} X_i \frechet^{-1/2}

where :math:`\frechet` is the Fréchet mean of the batch.
See :ref:`sphx_glr_generated_auto_examples_visualizations_plot_batchnorm_animation.py`


References
==========

.. bibliography::
   :filter: docname in docnames


.. seealso::

   - :doc:`generated/auto_examples/visualizations/index` — All visualization examples
   - :doc:`user_guide` — Getting started with SPD Learn
   - :doc:`api` — API Reference for all geometric operations
   - :doc:`faq` — Frequently asked questions
   - :doc:`contributing` — Contributing to SPD Learn
