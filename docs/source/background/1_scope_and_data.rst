.. _background_scope_data:

================================
Scope and Data Representations
================================

What SPD Learn Targets
======================

SPDLearn focuses on geometric deep learning on symmetric positive definite (SPD) manifolds for neural decoding with covariance or connectivity matrices, with a primary emphasis on EEG and fMRI.

* **EEG covariance matrices** from multichannel time series
* **fMRI functional connectivity** matrices from regional time series

The library is designed to integrate with established EEG and fMRI workflows
(e.g., `MOABB <https://moabb.neurotechx.com/>`_ and `Nilearn <https://nilearn.github.io/>`_)
and to support reproducible pipelines built around
SPDNet-based models.



SPD Data in Practice
====================

Given a multichannel signal matrix :math:`X \in \reals^{C \times T}`
(channels x time), a sample covariance matrix is:

.. math::

   C = \frac{1}{T - 1} (X - \bar{X})(X - \bar{X})^\top

This yields an SPD matrix when the sample count is sufficient. SPD Learn provides
utility layers such as :class:`~spd_learn.modules.CovLayer` to compute covariance
matrices from raw signals and feed them to neural network layers. The following
covariance estimation methods are supported:

* :func:`~spd_learn.functional.covariance`: Empirical covariance.
* :func:`~spd_learn.functional.sample_covariance`: Sample covariance with Bessel correction.
* :func:`~spd_learn.functional.real_covariance`: Real part of the covariance for complex signals.
* :func:`~spd_learn.functional.cross_covariance`: Cross-frequency covariance matrix.



Estimation Essentials
=====================

* **Sample size**: if the temporal dimension is smaller than the number of channels, i.e., :math:`T < C`,
  empirical covariances are rank-deficient. Use shrinkage or reduce dimensionality before forming covariances.
* **Regularization**: :class:`~spd_learn.modules.Shrinkage` estimators
  (e.g., :func:`~spd_learn.functional.regularize.ledoit_wolf` or OAS) stabilize
  covariance estimates when samples are limited.
* **Numerical safety**: handling "bad values" (e.g., non-finite inputs or
  rank-deficient matrices) is critical for Riemannian stability. SPD Learn
  enforces positive definiteness through small diagonal jitter or eigenvalue
  clipping, ensuring that downstream operations like matrix logarithms and
  inversions remain well-conditioned :cite:p:`higham2002accuracy`.
* **Precision considerations**: numerical precision (e.g., ``float32`` vs ``float64``)
  directly impacts the accuracy of geometric computations. While ``float32`` is
  often sufficient for deep learning, double precision (``float64``) is recommended
  for ill-conditioned matrices to prevent numerical drift. The library provides
  stable alternatives like the Log-Cholesky representation
  :cite:p:`lin2019riemannian`, which avoids expensive eigendecompositions while
  maintaining stability. Please refer to :doc:`/numerical_stability` for more details
  on handling different precision data.

For practical examples, see :doc:`/user_guide` and the API docs for
:class:`~spd_learn.modules.CovLayer` and :class:`~spd_learn.modules.Shrinkage`.
