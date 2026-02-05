:html_theme.sidebar_secondary.remove: true

.. _notation:

========
Notation
========

This page defines the mathematical notation used throughout SPD Learn documentation
and source code. Consistent notation follows the conventions established in
:cite:t:`congedo2017riemannian` and related literature.

.. contents:: Contents
   :local:
   :depth: 2


Spaces and Manifolds
====================

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Symbol
     - Description
     - Example
   * - :math:`\mathcal{S}^n_{++}`
     - SPD manifold: :math:`n \times n` symmetric positive definite matrices
     - :cite:p:`congedo2017riemannian`
   * - :math:`\text{Sym}(n)`
     - Space of :math:`n \times n` symmetric matrices
     - Tangent space at identity
   * - :math:`\mathcal{S}^n`
     - Space of symmetric matrices (same as :math:`\text{Sym}(n)`)
     - Log-Euclidean codomain
   * - :math:`\mathbb{R}^{n \times n}`
     - Space of :math:`n \times n` real matrices
     - General matrix space
   * - :math:`\mathbb{R}^n`
     - :math:`n`-dimensional Euclidean space
     - Vector space
   * - :math:`\mathcal{M}`
     - Generic Riemannian manifold
     - Abstract manifold notation
   * - :math:`\mathcal{L}_+`
     - Cholesky space: lower-triangular matrices with positive diagonal
     - Log-Cholesky metric


Groups
======

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Symbol
     - Description
     - Usage
   * - :math:`GL(n)`
     - General linear group: invertible :math:`n \times n` matrices
     - Affine invariance
   * - :math:`\text{St}(n, k)`
     - Stiefel manifold: :math:`n \times k` matrices with orthonormal columns
     - BiMap weights


Tangent Spaces
==============

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Symbol
     - Description
     - Convention
   * - :math:`T_P \mathcal{M}`
     - Tangent space at point :math:`P` on manifold :math:`\mathcal{M}`
     - **Uppercase** :math:`P`
   * - :math:`T_P \mathcal{S}^n_{++}`
     - Tangent space at :math:`P` on SPD manifold
     - :math:`\cong \text{Sym}(n)`

.. note::

   We use **uppercase** letters (e.g., :math:`P`, :math:`Q`) for points on the
   SPD manifold to distinguish them from lowercase scalars or indices. This
   convention is consistent with :cite:t:`pennec2006riemannian`.

   The tangent space :math:`T_P \mathcal{S}^n_{++}` is **isomorphic** to
   :math:`\text{Sym}(n)`, denoted by :math:`\cong` (not :math:`\equiv`).


Maps and Operations
===================

Exponential and Logarithmic Maps
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Symbol
     - Description
     - Domain → Codomain
   * - :math:`\text{Exp}_P`
     - Riemannian exponential map at :math:`P`
     - :math:`T_P\mathcal{M} \to \mathcal{M}`
   * - :math:`\text{Log}_P`
     - Riemannian logarithmic map at :math:`P`
     - :math:`\mathcal{M} \to T_P\mathcal{M}`
   * - :math:`\exp(S)`
     - Matrix exponential
     - :math:`\text{Sym}(n) \to \mathcal{S}^n_{++}`
   * - :math:`\log(X)`
     - Matrix logarithm
     - :math:`\mathcal{S}^n_{++} \to \text{Sym}(n)`


Cholesky Operations
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Symbol
     - Description
     - Notes
   * - :math:`\log_{\text{chol}}`
     - Log-Cholesky logarithm
     - Applies log to diagonal
   * - :math:`\exp_{\text{chol}}`
     - Log-Cholesky exponential
     - Applies exp to diagonal
   * - :math:`\text{tril}(L, k)`
     - Lower-triangular part with offset :math:`k`
     - :math:`k=-1` excludes diagonal


Riemannian Metrics
==================

Inner Products
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Symbol
     - Definition
     - Metric
   * - :math:`g^{\text{AIRM}}_P(v, w)`
     - :math:`\langle P^{-1/2} v P^{-1/2}, P^{-1/2} w P^{-1/2} \rangle_F`
     - Affine-Invariant
   * - :math:`g^{\text{LEM}}_P(v, w)`
     - :math:`\langle D_P \log(v), D_P \log(w) \rangle_F`
     - Log-Euclidean
   * - :math:`g^{\text{BW}}_P(V, W)`
     - :math:`\text{tr}(\mathcal{L}_P[V] W)`
     - Bures-Wasserstein
   * - :math:`g^{\text{LCM}}_P(v, w)`
     - See :doc:`geometric_concepts`
     - Log-Cholesky


Distance Functions
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Symbol
     - Formula
     - Metric
   * - :math:`d_{\text{AIRM}}(A, B)`
     - :math:`\| \log(A^{-1/2} B A^{-1/2}) \|_F`
     - Affine-Invariant
   * - :math:`d_{\text{LEM}}(A, B)`
     - :math:`\| \log(A) - \log(B) \|_F`
     - Log-Euclidean
   * - :math:`d_{\text{BW}}(A, B)`
     - :math:`\sqrt{\text{tr}(A) + \text{tr}(B) - 2\text{tr}((A^{1/2} B A^{1/2})^{1/2})}`
     - Bures-Wasserstein
   * - :math:`d_{\text{LCM}}(A, B)`
     - :math:`\| \text{logchol}(L_A) - \text{logchol}(L_B) \|_F`
     - Log-Cholesky


Special Symbols
===============

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Symbol
     - Meaning
     - LaTeX
   * - :math:`I` or :math:`I_n`
     - Identity matrix (of size :math:`n \times n`)
     - ``I`` or ``I_n``
   * - :math:`\cong`
     - Isomorphic to
     - ``\cong``
   * - :math:`\langle \cdot, \cdot \rangle_F`
     - Frobenius inner product
     - ``\langle \cdot, \cdot \rangle_F``
   * - :math:`\| \cdot \|_F`
     - Frobenius norm
     - ``\| \cdot \|_F``
   * - :math:`\text{tr}(\cdot)`
     - Matrix trace
     - ``\text{tr}(\cdot)``
   * - :math:`\text{diag}(\cdot)`
     - Diagonal matrix or diagonal elements
     - ``\text{diag}(\cdot)``
   * - :math:`\odot`
     - Log-Euclidean multiplication
     - ``\odot``
   * - :math:`\circledast`
     - Log-Euclidean scalar multiplication
     - ``\circledast``
   * - :math:`d(\cdot, \cdot)`
     - Generic distance function on the manifold
     - ``d(\cdot, \cdot)``
   * - :math:`G(P_1, \ldots, P_k)`
     - Geometric (Riemannian) mean of SPD matrices
     - ``G(P_1, \ldots, P_k)``


Log-Euclidean Group Operations
------------------------------

The Log-Euclidean framework defines group operations on SPD matrices:

.. math::

   A \odot B = \exp(\log(A) + \log(B))

.. math::

   t \circledast A = \exp(t \cdot \log(A)) = A^t


Layer Notation
==============

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Symbol
     - Operation
     - Module
   * - :math:`\text{ReEig}(X)`
     - :math:`U \max(\Lambda, \varepsilon) U^\top`
     - :class:`~spd_learn.modules.ReEig`
   * - :math:`\text{LogEig}(X)`
     - :math:`U \log(\Lambda) U^\top`
     - :class:`~spd_learn.modules.LogEig`
   * - :math:`\text{ExpEig}(X)`
     - :math:`U \exp(\Lambda) U^\top`
     - :class:`~spd_learn.modules.ExpEig`
   * - :math:`\mathcal{G}`
     - Fréchet mean (batch normalization)
     - :class:`~spd_learn.modules.SPDBatchNormMeanVar`


Common Variables
================

.. list-table::
   :header-rows: 1
   :widths: 15 60 25

   * - Symbol
     - Description
     - Type
   * - :math:`X, A, B, P, Q`
     - SPD matrices
     - :math:`\in \mathcal{S}^n_{++}`
   * - :math:`S, V, W`
     - Symmetric matrices (tangent vectors)
     - :math:`\in \text{Sym}(n)`
   * - :math:`U`
     - Orthogonal matrix (eigenvectors)
     - :math:`U^\top U = I`
   * - :math:`\Lambda`
     - Diagonal matrix (eigenvalues)
     - :math:`\lambda_i > 0`
   * - :math:`L`
     - Lower-triangular Cholesky factor
     - :math:`\in \mathcal{L}_+`
   * - :math:`W`
     - Stiefel matrix (BiMap weights)
     - :math:`\in \text{St}(n, k)`
   * - :math:`n`
     - Matrix dimension
     - Positive integer
   * - :math:`t`
     - Geodesic parameter
     - :math:`t \in [0, 1]`


Documentation Macros
====================

SPD Learn provides LaTeX-style macros for consistent notation in documentation.
These macros work in both HTML (via MathJax) and PDF (via LaTeX) output.

**Usage in RST files:**

.. code-block:: rst

   The SPD manifold :math:`\spd` equipped with the AIRM metric...

   The distance :math:`\dairm{A}{B}` between matrices...

   The tangent space :math:`\tangent{P}` at point :math:`P`...

**Available Macros:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Macro
     - Renders As
     - Description
   * - ``\spd``
     - :math:`\mathcal{S}^n_{++}`
     - SPD manifold
   * - ``\sym``
     - :math:`\text{Sym}(n)`
     - Symmetric matrices
   * - ``\manifold``
     - :math:`\mathcal{M}`
     - Generic manifold
   * - ``\choleskyspace``
     - :math:`\mathcal{L}_+`
     - Cholesky space
   * - ``\tangent{P}``
     - :math:`T_P \mathcal{M}`
     - Tangent space at P
   * - ``\tangentspd{P}``
     - :math:`T_P \mathcal{S}^n_{++}`
     - Tangent space on SPD at P
   * - ``\Exp{P}``
     - :math:`\text{Exp}_P`
     - Riemannian exponential at P
   * - ``\Log{P}``
     - :math:`\text{Log}_P`
     - Riemannian logarithm at P
   * - ``\dairm{A}{B}``
     - :math:`d_{\text{AIRM}}(A, B)`
     - AIRM distance
   * - ``\dlem{A}{B}``
     - :math:`d_{\text{LEM}}(A, B)`
     - Log-Euclidean distance
   * - ``\dbw{A}{B}``
     - :math:`d_{\text{BW}}(A, B)`
     - Bures-Wasserstein distance
   * - ``\dlcm{A}{B}``
     - :math:`d_{\text{LCM}}(A, B)`
     - Log-Cholesky distance
   * - ``\gairm{P}``
     - :math:`g^{\text{AIRM}}_P`
     - AIRM inner product at P
   * - ``\frob{X}``
     - :math:`\| X \|_F`
     - Frobenius norm
   * - ``\frobinner{X}{Y}``
     - :math:`\langle X, Y \rangle_F`
     - Frobenius inner product
   * - ``\tr``
     - :math:`\text{tr}`
     - Trace operator
   * - ``\diag``
     - :math:`\text{diag}`
     - Diagonal operator
   * - ``\reeig``
     - :math:`\text{ReEig}`
     - ReEig layer
   * - ``\logeig``
     - :math:`\text{LogEig}`
     - LogEig layer
   * - ``\frechet``
     - :math:`\mathcal{G}`
     - Fréchet mean
   * - ``\geomean``
     - :math:`G`
     - Geometric mean
   * - ``\I`` or ``\In``
     - :math:`I` or :math:`I_n`
     - Identity matrix
   * - ``\transpose``
     - :math:`^\top`
     - Transpose symbol

.. note::

   Macros are defined in ``docs/source/conf.py`` under ``mathjax3_config`` (for HTML)
   and ``latex_elements["preamble"]`` (for PDF). When adding new notation, update both
   locations and this reference table.


References
==========

.. bibliography::
   :filter: docname in docnames


.. seealso::

   - :doc:`geometric_concepts` — Detailed explanations of geometric operations
   - :doc:`glossary` — Definitions of key terms
   - :doc:`api` — API reference with mathematical specifications
