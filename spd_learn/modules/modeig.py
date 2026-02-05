# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch
import torch.nn as nn

from ..functional import (
    clamp_eigvals,
    matrix_exp,
    matrix_log,
    sym_to_upper,
)
from ..functional.autograd import (
    clamp_eigvals_func,
    matrix_exp_func,
    matrix_log_func,
)
from ..functional.numerical import get_epsilon


class ReEig(nn.Module):
    r"""Rectified Eigenvalue Layer (ReEig).

    This layer rectifies eigenvalues to ensure numerical stability and maintains
    positive definiteness :cite:p:`huang2017riemannian`. It applies a rectified
    linear unit (ReLU) to the eigenvalues of a symmetric matrix, introducing
    non-linearity while preserving the SPD property.

    .. math::

        \text{ReEig}(X) = U \max(\Lambda, \varepsilon I) U^\top

    where :math:`X = U \Lambda U^\top` is the eigendecomposition and
    :math:`\varepsilon > 0` is a rectification threshold that prevents eigenvalues
    from collapsing to zero.

    This operation prevents eigenvalues from becoming arbitrarily small and
    ensures that the output remains well-conditioned on the SPD manifold, which
    is crucial for stable training of subsequent spectral layers (e.g.,
    :class:`LogEig`).

    Parameters
    ----------
    threshold : float, optional
        The threshold for the rectified linear unit. If None, uses a
        dtype-aware threshold from the unified numerical configuration.
        Default: None.
    autograd : bool, default=False
        Whether to use the autograd backend.

    Notes
    -----
    When ``threshold=None``, the actual threshold is computed at runtime
    based on the input tensor's dtype using the unified numerical
    configuration. This ensures appropriate thresholds for different
    precisions (float16, float32, float64).

    See Also
    --------
    :class:`LogEig` : Projects SPD matrices to the tangent space via matrix logarithm.
    :class:`BiMap` : Bilinear mapping layer for dimensionality reduction.
    :class:`SPDBatchNormMeanVar` : Riemannian batch normalization for SPD matrices.
    :func:`~spd_learn.functional.clamp_eigvals` : Functional version of eigenvalue clamping.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import ReEig
    >>> # Use default dtype-aware threshold
    >>> layer = ReEig()
    >>> X = torch.randn(2, 5, 5)
    >>> X = X @ X.mT  # Make SPD
    >>> Y = layer(X)
    >>> # Use explicit threshold
    >>> layer_explicit = ReEig(threshold=1e-3)

    .. plot::
        :include-source:

        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from spd_learn.modules import ReEig

        # Visualize ReEig rectification function
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: ReEig function
        ax1 = axes[0]
        epsilon = 0.3
        x = np.linspace(0, 2.5, 200)
        y_reeig = np.maximum(x, epsilon)

        ax1.plot(x, x, 'k--', alpha=0.4, label='Identity', linewidth=2)
        ax1.plot(x, y_reeig, 'b-', linewidth=3, label=f'ReEig (eps={epsilon})')
        ax1.fill_between([0, epsilon], [epsilon, epsilon], [0, 0],
                         color='red', alpha=0.15, label='Clamped')
        ax1.axhline(y=epsilon, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlim(-0.1, 2.5)
        ax1.set_ylim(-0.1, 2.5)
        ax1.set_xlabel('Input eigenvalue')
        ax1.set_ylabel('Output eigenvalue')
        ax1.set_title('ReEig Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Right: Eigenvalue comparison
        ax2 = axes[1]
        torch.manual_seed(42)
        eigvals = torch.tensor([2.0, 0.5, 0.01, 0.001])
        eigvecs = torch.linalg.qr(torch.randn(4, 4))[0]
        X = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        X = X.unsqueeze(0)

        reeig = ReEig(threshold=1e-4)
        Y = reeig(X)

        ev_before = torch.linalg.eigvalsh(X[0]).numpy()
        ev_after = torch.linalg.eigvalsh(Y[0]).numpy()

        x_pos = np.arange(4)
        ax2.bar(x_pos - 0.2, ev_before, 0.35, label='Before', color='#3498db')
        ax2.bar(x_pos + 0.2, ev_after, 0.35, label='After', color='#e74c3c')
        ax2.axhline(y=1e-4, color='green', linestyle='--', label='Threshold')
        ax2.set_yscale('log')
        ax2.set_xlabel('Eigenvalue index')
        ax2.set_ylabel('Eigenvalue')
        ax2.set_title('Eigenvalue Rectification')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    """

    threshold_: torch.Tensor  # Type annotation for registered buffer

    def __init__(self, threshold=None, autograd=False, device=None, dtype=None):
        super().__init__()
        self._use_dynamic_threshold = threshold is None
        if threshold is None:
            # Will be computed dynamically based on input dtype
            self.register_buffer(
                "threshold_", torch.tensor(0.0, device=device, dtype=dtype)
            )
        else:
            self.register_buffer(
                "threshold_", torch.tensor(threshold, device=device, dtype=dtype)
            )
        self.autograd_ = autograd

    def _get_threshold(self, X: torch.Tensor) -> torch.Tensor:
        """Get the threshold for eigenvalue clamping.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor (used for dtype).

        Returns
        -------
        torch.Tensor
            The threshold value.
        """
        if self._use_dynamic_threshold:
            threshold_value = get_epsilon(X.dtype, "eigval_clamp")
            return torch.tensor(threshold_value, device=X.device, dtype=X.dtype)
        return self.threshold_.to(device=X.device, dtype=X.dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ReEig layer.

        Parameters
        ----------
        X : torch.Tensor
            Input symmetric matrix.

        Returns
        -------
        torch.Tensor
            The output matrix with rectified eigenvalues.
        """
        threshold = self._get_threshold(X)
        if self.autograd_:
            return clamp_eigvals_func(X, threshold)
        return clamp_eigvals.apply(X, threshold)


class LogEig(nn.Module):
    r"""Logarithmic Eigenvalue Layer (LogEig).

    This layer maps SPD matrices to the tangent space at the identity via the
    Riemannian logarithm under the affine-invariant metric
    :cite:p:`huang2017riemannian`. The output is then vectorized.

    .. math::

        \log(X) = U \log(\Lambda) U^\top

    where :math:`X = U \Lambda U^\top` is the eigendecomposition.

    This operation embeds SPD matrices into the tangent space at the identity,
    which is a vector space, thereby enabling the use of standard Euclidean
    layers such as linear classifiers or fully connected networks for downstream
    tasks.

    Parameters
    ----------
    upper : bool, default=True
        If `True`, only the upper triangular part of the matrix is
        vectorized.
    flatten : bool, default=True
        If `True`, the output is flattened.
    autograd : bool, default=False
        Whether to use the autograd backend.

    See Also
    --------
    :class:`ExpEig` : Inverse operation, maps from tangent space back to manifold.
    :class:`ReEig` : Eigenvalue rectification to ensure numerical stability before LogEig.
    :class:`BiMap` : Bilinear mapping for dimensionality reduction on SPD matrices.
    :func:`~spd_learn.functional.matrix_log` : Functional version of matrix logarithm.
    :func:`~spd_learn.functional.log_euclidean_distance` : Distance computation in log-domain.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import LogEig
    >>> layer = LogEig(upper=True)
    >>> X = torch.randn(2, 4, 4)
    >>> X = X @ X.mT + 0.1 * torch.eye(4)  # Make SPD
    >>> Y = layer(X)
    >>> Y.shape
    torch.Size([2, 10])

    .. plot::
        :include-source:

        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from spd_learn.modules import LogEig

        torch.manual_seed(42)

        # Create a 4x4 SPD matrix
        n = 4
        A = torch.randn(n, n)
        X = A @ A.T + 0.1 * torch.eye(n)
        X = X.unsqueeze(0)

        # Apply LogEig
        logeig_full = LogEig(upper=False, flatten=False)
        logeig_vec = LogEig(upper=True, flatten=True)

        log_matrix = logeig_full(X)
        log_vector = logeig_vec(X)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Input SPD matrix
        ax1 = axes[0]
        im1 = ax1.imshow(X[0].numpy(), cmap='RdBu_r', aspect='auto')
        ax1.set_title('Input SPD Matrix X')
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # Matrix logarithm
        ax2 = axes[1]
        im2 = ax2.imshow(log_matrix[0].numpy(), cmap='RdBu_r', aspect='auto')
        ax2.set_title(r'log(X) (Tangent Space)')
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        # Vectorized output
        ax3 = axes[2]
        vec = log_vector[0].numpy()
        ax3.bar(range(len(vec)), vec, color='#2ecc71', alpha=0.8)
        ax3.set_xlabel('Vector index')
        ax3.set_ylabel('Value')
        ax3.set_title(f'Vectorized (dim={len(vec)})')
        ax3.grid(True, alpha=0.3)

        plt.suptitle('LogEig: SPD to Tangent Space Mapping', fontweight='bold')
        plt.tight_layout()
        plt.show()
    """

    autograd_: torch.Tensor  # Type annotation for registered buffer

    def __init__(
        self, upper=True, flatten=True, autograd=False, device=None, dtype=None
    ):
        super().__init__()
        self.upper = upper
        self.flatten = flatten
        self.dtype = dtype
        self.device = device

        self.register_buffer(
            "autograd_", torch.tensor(autograd, device=device, dtype=dtype)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LogEig layer.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD matrix.

        Returns
        -------
        torch.Tensor
            The vectorized output in the tangent space.
        """
        if self.autograd_:
            X_log = matrix_log_func(X)
        else:
            X_log = matrix_log.apply(X)

        if self.upper:
            X_log = sym_to_upper(X_log)
        elif self.flatten:
            X_log = X_log.flatten(start_dim=-2)

        return X_log.to(device=self.device)


class ExpEig(nn.Module):
    r"""Exponential Eigenvalue Layer (ExpEig).

    This layer maps symmetric matrices from the tangent space back to the SPD
    manifold via the matrix exponential :cite:p:`huang2017riemannian`. It is
    the inverse operation of :class:`LogEig`.

    .. math::

        \exp(X) = U \exp(\Lambda) U^\top

    where :math:`X = U \Lambda U^\top` is the eigendecomposition of the
    symmetric input matrix, and :math:`\exp(\Lambda)` applies the exponential
    element-wise to eigenvalues.

    **Geometric Interpretation**

    The exponential map :math:`\exp_I: T_I\mathcal{S}_{++}^n \to \mathcal{S}_{++}^n`
    projects tangent vectors at the identity to points on the SPD manifold.
    This enables:

    1. **Manifold reconstruction**: After processing in tangent space (e.g.,
       via Euclidean layers), data can be projected back to valid SPD matrices.
    2. **Generative models**: Sampling in tangent space and mapping to manifold.
    3. **Residual connections**: Combined with LogEig for manifold-aware skip
       connections.

    Since :math:`\exp(\cdot)` always produces positive values, the output is
    guaranteed to be SPD regardless of the input symmetric matrix.

    Parameters
    ----------
    upper : bool, default=False
        If `True`, assumes input is vectorized upper triangular and reconstructs
        symmetric matrix before applying exponential.
    flatten : bool, default=False
        If `True`, the output is flattened.
    autograd : bool, default=False
        Whether to use the autograd backend.

    See Also
    --------
    :class:`LogEig` : Inverse operation, maps SPD to tangent space.
    :class:`ReEig` : Eigenvalue rectification for numerical stability.
    :func:`~spd_learn.functional.matrix_exp` : Functional version.
    :class:`LogEuclideanResidual` : Uses exp/log for manifold residuals.
    """

    autograd_: torch.Tensor  # Type annotation for registered buffer

    def __init__(
        self, upper=False, flatten=False, autograd=False, device=None, dtype=None
    ):
        super().__init__()
        self.upper = upper
        self.flatten = flatten
        self.dtype = dtype
        self.device = device

        self.register_buffer(
            "autograd_", torch.tensor(autograd, device=device, dtype=dtype)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ExpEig layer.

        Parameters
        ----------
        X : torch.Tensor
            Input matrix in the tangent space.

        Returns
        -------
        torch.Tensor
            The output SPD matrix.
        """
        if self.autograd_:
            X_exp = matrix_exp_func(X)
        else:
            X_exp = matrix_exp.apply(X)

        if self.upper:
            X_exp = sym_to_upper(X_exp)
        elif self.flatten:
            X_exp = X_exp.flatten(start_dim=-2)

        return X_exp.to(device=self.device)
