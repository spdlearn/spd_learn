.. _background_geometry_essentials:

===========================
Geometry Essentials
===========================

This page provides the minimal geometric background needed to understand why
SPD Learn uses specialized operations. Full derivations and formulas live in
:doc:`/geometric_concepts`, and we recommend the survey paper :cite:p:`ju2026spdmatrixlearningneuroimaging`
for more details.


Why Geometry Matters
====================

SPD matrices do not form a vector space: simple Euclidean operations (such as
subtraction or arithmetic averaging) can break positive definiteness. The SPD
manifold provides the appropriate setting for distances, averages, and
interpolation.

Two ideas are used throughout the library:

* **Tangent space**: a local linearization that allows Euclidean tools.
* **Log/Exp maps**: conversions between SPD matrices and the tangent space.


Metric Choices in SPD Learn
===========================

A **metric** (specifically a Riemannian metric) defines how to measure distances
and angles at each point on the manifold. It determines the "shortest path"
(geodesic) between two SPD matrices and how they are averaged geometrically.

SPD Learn implements multiple metrics. The choice trades off invariance,
stability, and computational cost.

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Metric
     - When to use
     - Tradeoffs
   * - **AIRM**
     - When affine invariance matters (e.g., within-session EEG)
     - Most faithful geometry, highest cost
   * - **LEM**
     - Default for efficiency
     - Loses affine invariance
   * - **LCM**
     - Large matrices or stability-first workflows
     - Less geometric fidelity than AIRM
   * - **BWM**
     - When optimal-transport interpretation is useful
     - Different curvature; not affine invariant

In practice, Log-Euclidean (LEM) is a strong default. Use AIRM when affine
invariance is a requirement rather than an optional benefit.

For details and formulas, see :doc:`/geometric_concepts`.
