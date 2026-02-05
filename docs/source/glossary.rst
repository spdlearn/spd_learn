.. _glossary:

========
Glossary
========

This glossary defines key terms used throughout SPD Learn documentation.

.. glossary::
   :sorted:

   SPD Matrix
      **Symmetric Positive Definite Matrix**. A square matrix :math:`X` that is
      symmetric (:math:`X = X^\top`) and has all positive eigenvalues. SPD
      matrices form a Riemannian manifold and commonly arise as covariance
      matrices.

   Riemannian Manifold
      A smooth manifold equipped with a Riemannian metric that allows measuring
      distances and angles. The space of SPD matrices forms a Riemannian manifold
      where standard Euclidean operations do not apply directly.

   Tangent Space
      The vector space of all possible directions at a point on a manifold.
      For SPD matrices, the tangent space at a point is the space of symmetric
      matrices. The :term:`LogEig` operation projects SPD matrices to the
      tangent space at the identity.

   Fréchet Mean
      The generalization of the arithmetic mean to Riemannian manifolds. For
      SPD matrices, it minimizes the sum of squared geodesic distances. Used
      in :term:`SPDBatchNormMean` and :term:`SPDBatchNormMeanVar` for centering.

   Geodesic
      The shortest path between two points on a manifold. On the SPD manifold,
      geodesics are curves that preserve the Riemannian structure. The
      :term:`geodesic distance` measures length along these paths.

   Geodesic Distance
      The length of the shortest path (geodesic) between two points on a
      manifold. For SPD matrices :math:`A` and :math:`B`, the affine-invariant
      Riemannian distance is :math:`\frob{\log(A^{-1/2} B A^{-1/2})}`.

   Log-Euclidean Distance
      A computationally efficient distance metric for SPD matrices:
      :math:`\frob{\log(A) - \log(B)}`. Faster than geodesic distance but
      less geometrically accurate.

   BiMap
      **Bilinear Mapping Layer**. A neural network layer that performs the
      transformation :math:`Y = W^\top X W` where :math:`W` is constrained
      to the :term:`Stiefel manifold`. This is the SPD analog of a linear layer.

   ReEig
      **Rectified Eigenvalue Layer**. A non-linearity for SPD matrices that
      clamps eigenvalues to be above a threshold:
      :math:`\reeig(X) = U \max(\Lambda, \varepsilon) U^\top`. Similar
      to ReLU in standard networks.

   LogEig
      **Logarithmic Eigenvalue Layer**. Maps SPD matrices to the tangent space
      via matrix logarithm: :math:`\logeig(X) = U \log(\Lambda) U^\top`.
      Converts the curved SPD manifold to flat Euclidean space for classification.

   ExpEig
      **Exponential Eigenvalue Layer**. The inverse of :term:`LogEig`, mapping
      symmetric matrices back to the SPD manifold via matrix exponential:
      :math:`\expeig(X) = U \exp(\Lambda) U^\top`.

   Stiefel Manifold
      The manifold of orthonormal matrices :math:`W` satisfying
      :math:`W^\top W = I`. The weight matrices in :term:`BiMap` are constrained
      to this manifold to preserve the SPD structure.

   Trivialization
      A technique for optimizing manifold-constrained parameters by mapping
      from unconstrained Euclidean space to the manifold. SPD Learn uses
      trivialization for :term:`Stiefel manifold` (BiMap weights) and SPD
      (batch norm parameters) constraints.

   SPDBatchNormMean
      Riemannian batch normalization that centers SPD matrices at their
      :term:`Fréchet mean`: :math:`\tilde{P}_i = G^{-1/2} P_i G^{-1/2}`.
      Based on Brooks et al. (2019).

   SPDBatchNormMeanVar
      Extended batch normalization for SPD matrices that normalizes both
      mean (via :term:`Fréchet mean`) and dispersion (via power transformation).
      Enables domain adaptation through domain-specific statistics.

   Covariance Matrix
      A matrix capturing the pairwise statistical relationships between
      variables: :math:`\Sigma_{ij} = \text{Cov}(X_i, X_j)`. Covariance
      matrices are symmetric and positive semi-definite (SPD if full rank).

   Sample Covariance
      An estimator of the covariance matrix from samples:
      :math:`\hat{\Sigma} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(x_i - \bar{x})^\top`.
      May not be SPD with few samples; use regularization.

   Shrinkage
      A regularization technique that interpolates between the sample
      covariance and a structured estimate (e.g., diagonal):
      :math:`\hat{\Sigma}_{\text{shrink}} = (1-\alpha)\hat{\Sigma} + \alpha \cdot \text{target}`.
      Ledoit-Wolf shrinkage automatically selects optimal :math:`\alpha`.

   CSP
      **Common Spatial Patterns**. A spatial filtering technique for EEG that
      maximizes variance differences between classes. :term:`TensorCSPNet`
      learns CSP-like filters through :term:`BiMap` layers.

   Filter Bank
      Decomposition of a signal into multiple frequency bands using bandpass
      filters. :term:`TensorCSPNet` processes filter bank representations
      to capture frequency-specific patterns :cite:p:`ju2022tensor`.

   TensorCSPNet
      A model that stacks covariance matrices from multiple frequency bands
      into a tensor structure and processes them through SPDNet layers to
      capture temporal, spectral, and spatial information jointly.

   TSMNet
      **Tangent Space Mapping Network**. A model combining convolutional
      features, covariance pooling, and :term:`SPDBatchNormMeanVar` for domain
      adaptation in EEG classification.

   MAtt
      **Manifold Attention Network**. A model that applies attention mechanisms
      on the SPD manifold, computing attention weights based on
      :term:`Log-Euclidean distance` between SPD matrices.

   GREEN
      **Gabor Riemann EEGNet**. A lightweight model using learnable Gabor
      wavelets for time-frequency feature extraction combined with SPDNet
      layers for classification.

   EEGSPDNet
      A model using channel-specific (grouped) convolutions before covariance
      computation, allowing each EEG channel to learn independent temporal
      filters.

   PhaseSPDNet
      A model that applies phase-space embedding (time-delay coordinates)
      to input signals before SPDNet processing, capturing nonlinear
      dynamical structure.

   Phase-Space Embedding
      Reconstruction of a dynamical system's state space from a single time
      series using time-delayed copies: :math:`[x(t), x(t-\tau), x(t-2\tau), \ldots]`.
      Based on Takens' embedding theorem.

   Eigendecomposition
      Factorization of a matrix as :math:`X = U \Lambda U^\top` where
      :math:`U` contains eigenvectors and :math:`\Lambda` is diagonal with
      eigenvalues. Fundamental operation for SPD matrix computations.

   Loewner Matrix
      A matrix used in computing gradients through eigenvalue functions.
      For function :math:`f`, element :math:`L_{ij} = (f(\lambda_i) - f(\lambda_j)) / (\lambda_i - \lambda_j)`
      when :math:`i \neq j`, and :math:`L_{ii} = f'(\lambda_i)`.

   BCI
      **Brain-Computer Interface**. A system that translates brain activity
      (typically EEG) into commands for external devices. Motor imagery
      classification is a common BCI paradigm.

   Motor Imagery
      Mental rehearsal of a motor action without physical execution. Produces
      characteristic EEG patterns (mu/beta desynchronization) that can be
      decoded for BCI applications.

   Domain Adaptation
      Techniques for adapting a model trained on one domain (e.g., one
      recording session) to perform well on a different domain (e.g., another
      session) :cite:p:`zanini2017transfer`. :term:`TSMNet` with :term:`SPDBatchNormMeanVar` enables source-free
      unsupervised domain adaptation.

   SFUDA
      **Source-Free Unsupervised Domain Adaptation**. Domain adaptation
      without access to source domain data during adaptation. Achieved in
      SPD Learn by updating :term:`SPDBatchNormMeanVar` running statistics on
      unlabeled target data.
