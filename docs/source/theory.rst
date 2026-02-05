:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true

.. _theory:

======
Theory
======

This section provides the theoretical foundations for understanding SPD Learn,
covering both the mathematical concepts and the terminology used throughout
the documentation.

.. only:: html

   .. grid:: 1 2 2 5
      :gutter: 4

      .. grid-item-card:: Background
         :link: background/index
         :link-type: doc

         Concise background for using SPD Learn: data representations, minimal
         geometry, SPDNet pipeline, and limitations.

      .. grid-item-card:: Geometric Concepts
         :link: geometric_concepts
         :link-type: doc

         Riemannian geometry on SPD manifolds: metrics, geodesics,
         exponential/logarithmic maps, parallel transport, and layer visualizations.

      .. grid-item-card:: Numerical Stability
         :link: numerical_stability
         :link-type: doc

         Managing numerical precision in SPD operations: dtype-aware thresholds,
         eigenvalue clamping, and troubleshooting common issues.

      .. grid-item-card:: Notation
         :link: notation
         :link-type: doc

         Standard mathematical notation conventions used throughout SPD Learn,
         including symbols for manifolds, metrics, and operations.

      .. grid-item-card:: Glossary
         :link: glossary
         :link-type: doc

         Quick reference for key terms and definitions used throughout SPD Learn,
         from SPD matrices to domain adaptation.

      .. grid-item-card:: References
         :link: references
         :link-type: doc

         Literature map, model legend, and complete bibliography of foundational
         works implemented in SPD Learn.


Overview
========

Understanding the geometry of Symmetric Positive Definite (SPD) matrices is
essential for effective use of SPD Learn. The pages below provide six
complementary perspectives:

**Background** provides the minimal context needed to use SPD Learn:

- What data representations the library targets (EEG/fMRI covariances)
- Minimal geometric ideas and metric choices
- How SPDNet is implemented in the package
- Practical limitations and decision checklist

**Geometric Concepts** offers an in-depth exploration of:

- The SPD manifold structure and why covariance matrices live on it
- Riemannian metrics (Affine-Invariant, Log-Euclidean, Bures-Wasserstein, Log-Cholesky)
- Tangent spaces, exponential and logarithmic maps
- Parallel transport for domain adaptation
- trivialization for optimization
- Visualizations of SPD network layers

**Numerical Stability** covers practical considerations:

- Dtype-aware numerical thresholds
- Configuration system for stability parameters
- Safe eigenvalue clamping and validation
- Recommendations for different training scenarios
- Troubleshooting NaN values and convergence issues

**Notation** establishes consistent mathematical conventions:

- Spaces and manifolds (:math:`\spd`, :math:`\sym`, etc.)
- Tangent space notation (:math:`\tangent{P}`)
- Distance and metric symbols
- Layer operations and special symbols

**Glossary** provides concise definitions for:

- Core mathematical concepts (SPD matrices, Riemannian manifolds, geodesics)
- SPD Learn layers and modules (BiMap, ReEig, LogEig, BatchNorm variants)
- Model architectures (SPDNet, TSMNet, TensorCSPNet, GREEN, etc.)
- Application-specific terms (BCI, motor imagery, domain adaptation)

**References** offers a comprehensive bibliography including:

- Interactive literature map showing model relationships and citation networks
- Color-coded legend for all model families implemented in SPD Learn
- Full bibliographic entries with DOIs and links
- BibTeX entries for easy citation


.. toctree::
   :maxdepth: 2
   :hidden:

   background/index
   geometric_concepts
   numerical_stability
   notation
   glossary
   references


.. seealso::

   - :doc:`user_guide` -- Getting started with SPD Learn
   - :doc:`api` -- Complete API reference
   - :doc:`generated/auto_examples/index` -- Practical examples
