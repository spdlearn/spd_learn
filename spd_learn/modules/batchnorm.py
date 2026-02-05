# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch

from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from ..functional import (
    airm_geodesic,
    matrix_inv_sqrt,
    matrix_log,
    matrix_power,
    matrix_sqrt,
)
from ..functional.batchnorm import (
    karcher_mean_iteration,
    spd_centering,
    spd_rebiasing,
    tangent_space_variance,
)
from ..functional.numerical import numerical_config
from .manifold import PositiveDefiniteScalar, SymmetricPositiveDefinite


class SPDBatchNormMean(nn.Module):
    r"""Riemannian Batch Normalization for SPD Matrices (Mean-only).

    This class implements the Riemannian Batch Normalization (RBN) layer for
    the Symmetric Positive Definite (SPD) manifold
    :cite:p:`brooks2019riemannian`.

    .. math::

       \tilde{P}_i = \mathcal{G}^{-\frac{1}{2}} P_i \mathcal{G}^{-\frac{1}{2}}

    where :math:`\mathcal{G}` is the Fréchet mean of the batch.

    Parameters
    ----------
    num_features : int
        The size of the SPD matrices (number of features).
    momentum : float, default=0.1
        Momentum factor for updating the running mean.
    rebias : bool, default=True
        If `True`, the layer rebases the data.
    n_iter : int, default=1
        Number of Karcher flow iterations to estimate the batch mean.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import SPDBatchNormMean
    >>> bn = SPDBatchNormMean(num_features=4, momentum=0.1)
    >>> X = torch.randn(8, 4, 4)
    >>> X = X @ X.mT + 0.1 * torch.eye(4)  # Make SPD
    >>> Y = bn(X)
    >>> Y.shape
    torch.Size([8, 4, 4])

    .. plot::
        :include-source:

        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from spd_learn.modules import SPDBatchNormMean

        def spd_to_ellipse(spd_matrix, center=(0, 0), scale=1.0):
            eigvals, eigvecs = np.linalg.eigh(spd_matrix)
            width = 2 * np.sqrt(eigvals[1]) * scale
            height = 2 * np.sqrt(eigvals[0]) * scale
            angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
            return Ellipse(center, width, height, angle=angle)

        # Create batch of 2x2 SPD matrices
        torch.manual_seed(42)
        np.random.seed(42)
        batch_size = 6

        spd_batch = []
        for i in range(batch_size):
            scale = np.random.uniform(0.5, 2.0)
            angle = np.random.uniform(0, np.pi)
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            D = np.diag([scale, scale * np.random.uniform(0.3, 1.0)])
            S = R @ D @ D @ R.T
            spd_batch.append(S)

        X = torch.tensor(np.array(spd_batch), dtype=torch.float32)

        # Apply SPDBatchNormMean
        bn = SPDBatchNormMean(num_features=2, momentum=0.1, rebias=False)
        bn.train()
        Y = bn(X)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, batch_size))

        # Before normalization
        ax1 = axes[0]
        for i, S in enumerate(X.numpy()):
            ellipse = spd_to_ellipse(S, scale=0.5)
            ellipse.set_facecolor(colors[i])
            ellipse.set_alpha(0.6)
            ellipse.set_edgecolor('black')
            ax1.add_patch(ellipse)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)
        ax1.set_title('Before SPDBatchNormMean', fontweight='bold')

        # After normalization
        ax2 = axes[1]
        for i, S in enumerate(Y.detach().numpy()):
            ellipse = spd_to_ellipse(S, scale=0.5)
            ellipse.set_facecolor(colors[i])
            ellipse.set_alpha(0.6)
            ellipse.set_edgecolor('black')
            ax2.add_patch(ellipse)
        identity = Ellipse((0, 0), 1, 1, facecolor='none',
                           edgecolor='red', linewidth=2, linestyle='--')
        ax2.add_patch(identity)
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.axvline(x=0, color='k', linewidth=0.5)
        ax2.set_title('After SPDBatchNormMean', fontweight='bold')

        plt.suptitle('SPDBatchNormMean: Riemannian Centering', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()
    """

    def __init__(
        self,
        num_features,
        momentum=0.1,
        rebias=True,
        n_iter=1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.rebias = rebias
        self.n_iter = n_iter

        self.register_buffer(
            "running_mean",
            torch.empty(1, num_features, num_features, device=device, dtype=dtype),
        )

        self.bias = nn.Parameter(
            torch.empty(1, num_features, num_features, device=device, dtype=dtype),
        )

        self.reset_parameters()
        register_parametrization(self, "bias", SymmetricPositiveDefinite())

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_mean[0].fill_diagonal_(1)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.rebias:
            # Initialize bias to identity matrix (will become zeros in tangent space)
            self.bias.zero_()
            self.bias[0].fill_diagonal_(1.0)

    def forward(self, input):
        """Forward pass of the Riemannian Batch Normalization layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, h, n, n)`, where each slice
            along the batch dimension is an SPD matrix.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as the input.

        """
        if self.training:
            mean = input.mean(dim=0, keepdim=True)
            if input.shape[0] > 1:
                for _ in range(self.n_iter):
                    mean = karcher_mean_iteration(input, mean)
            with torch.no_grad():
                self.running_mean = airm_geodesic(
                    self.running_mean, mean, self.momentum
                )
        else:
            mean = self.running_mean

        mean_invsqrt = matrix_inv_sqrt.apply(mean)
        output = spd_centering(input, mean_invsqrt)

        if self.bias is not None:
            bias_sqrt = matrix_sqrt.apply(self.bias)
            output = spd_rebiasing(output, bias_sqrt)

        return output


class SPDBatchNormMeanVar(nn.Module):
    r"""SPD Batch Normalization (Mean and Variance).

    This class implements the SPD Batch Normalization (SPDBN) layer for the
    Symmetric Positive Definite (SPD) manifold :cite:p:`kobler2022spd`.

    The normalization performs Riemannian centering via parallel transport under
    the affine-invariant Riemannian metric. Given a batch of SPD matrices
    :math:`\{P_i\}_{i=1}^N` with batch Fréchet mean :math:`\mathcal{G}`, samples
    are centered through:

    .. math::

        \tilde{P}_i = \Gamma_{\mathcal{G} \to I}(P_i) = \mathcal{G}^{-\frac{1}{2}} P_i \mathcal{G}^{-\frac{1}{2}}

    where :math:`\Gamma_{\mathcal{G} \to I}` represents parallel transport from
    :math:`\mathcal{G}` to the identity matrix :math:`I`. This is followed by
    dispersion normalization and an optional learnable re-biasing via a congruence
    transformation with learnable SPD matrix :math:`G`:

    .. math::

        \hat{P}_i = \Gamma_{I \to G}(\tilde{P}_i) = G^{\frac{1}{2}} \tilde{P}_i G^{\frac{1}{2}}

    This acts as a manifold-consistent analogue of Euclidean batch normalization,
    absorbing global covariance shifts and improving conditioning for subsequent
    tangent-space operations. The momentum-based estimator updates the running
    mean in a single pass, reducing the need for repeated Karcher iterations.
    Implementation follows Kobler et al.'s SPDMBN/SPDBN: Fréchet mean/variance
    on SPD and dispersion normalization with momentum running statistics.

    Parameters
    ----------
    num_features : int
        The size of the SPD matrices (number of features).
    momentum : float, default=0.1
        Momentum factor for updating the running mean.
    affine : bool, default=True
        If `True`, this module has learnable affine parameters.
    n_iter : int, default=1
        Number of Karcher flow iterations to estimate the batch mean.
    bias_requires_grad : bool, default=True
        If `True`, the bias parameter requires a gradient.
    weight_requires_grad : bool, default=True
        If `True`, the weight parameter requires a gradient.
    eps : float, optional
        A value added to the denominator for numerical stability.
        If None, uses the value from the unified numerical configuration.
        Default: None.

    See Also
    --------
    :class:`SPDBatchNormMean` : Mean-only Riemannian batch normalization from Brooks et al.
    :class:`BiMap` : Bilinear mapping layer often used before batch normalization.
    :class:`ReEig` : Eigenvalue rectification for numerical stability.
    :class:`LogEig` : Projects normalized SPD matrices to tangent space.
    :func:`~spd_learn.functional.log_euclidean_mean` : Computes the Log-Euclidean mean of SPD matrices.
    :func:`~spd_learn.functional.parallel_transport_airm` : Parallel transport under AIRM.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import SPDBatchNormMeanVar
    >>> bn = SPDBatchNormMeanVar(num_features=4, momentum=0.1)
    >>> X = torch.randn(8, 4, 4)
    >>> X = X @ X.mT + 0.1 * torch.eye(4)  # Make SPD
    >>> Y = bn(X)
    >>> Y.shape
    torch.Size([8, 4, 4])

    .. plot::
        :include-source:

        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from spd_learn.modules import SPDBatchNormMeanVar

        def spd_to_ellipse(spd_matrix, center=(0, 0), scale=1.0):
            eigvals, eigvecs = np.linalg.eigh(spd_matrix)
            width = 2 * np.sqrt(eigvals[1]) * scale
            height = 2 * np.sqrt(eigvals[0]) * scale
            angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
            return Ellipse(center, width, height, angle=angle)

        # Create batch of 2x2 SPD matrices with different means
        torch.manual_seed(42)
        np.random.seed(42)
        batch_size = 6

        # Generate scattered SPD matrices
        spd_batch = []
        for i in range(batch_size):
            scale = np.random.uniform(0.5, 2.0)
            angle = np.random.uniform(0, np.pi)
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            D = np.diag([scale, scale * np.random.uniform(0.3, 1.0)])
            S = R @ D @ D @ R.T
            spd_batch.append(S)

        X = torch.tensor(np.array(spd_batch), dtype=torch.float32)

        # Apply SPDBatchNormMeanVar
        bn = SPDBatchNormMeanVar(num_features=2, momentum=0.1, affine=False)
        bn.train()
        Y = bn(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Before normalization
        ax1 = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, batch_size))
        for i, S in enumerate(X.numpy()):
            ellipse = spd_to_ellipse(S, scale=0.5)
            ellipse.set_facecolor(colors[i])
            ellipse.set_alpha(0.6)
            ellipse.set_edgecolor('black')
            ax1.add_patch(ellipse)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)
        ax1.set_title('Before SPDBatchNormMeanVar\n(Scattered)', fontweight='bold')

        # After normalization
        ax2 = axes[1]
        for i, S in enumerate(Y.detach().numpy()):
            ellipse = spd_to_ellipse(S, scale=0.5)
            ellipse.set_facecolor(colors[i])
            ellipse.set_alpha(0.6)
            ellipse.set_edgecolor('black')
            ax2.add_patch(ellipse)
        # Draw identity reference
        identity = Ellipse((0, 0), 1, 1, facecolor='none',
                           edgecolor='red', linewidth=2, linestyle='--')
        ax2.add_patch(identity)
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.axvline(x=0, color='k', linewidth=0.5)
        ax2.set_title('After SPDBatchNormMeanVar\n(Centered at Identity)', fontweight='bold')

        # Eigenvalue comparison
        ax3 = axes[2]
        eigvals_before = [np.linalg.eigvalsh(s) for s in X.numpy()]
        eigvals_after = [np.linalg.eigvalsh(s) for s in Y.detach().numpy()]

        x_pos = np.arange(batch_size)
        width = 0.35
        ax3.bar(x_pos - width/2,
                [np.prod(e) for e in eigvals_before],
                width, label='Before (det)', color='#3498db', alpha=0.8)
        ax3.bar(x_pos + width/2,
                [np.prod(e) for e in eigvals_after],
                width, label='After (det)', color='#e74c3c', alpha=0.8)
        ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Identity det=1')
        ax3.set_xlabel('Sample index')
        ax3.set_ylabel('Determinant')
        ax3.set_title('Determinant Normalization', fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.suptitle('SPDBatchNormMeanVar: Riemannian Batch Normalization', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()
    """

    def __init__(
        self,
        num_features,
        momentum=0.1,
        affine=True,
        n_iter=1,
        bias_requires_grad=True,
        weight_requires_grad=True,
        eps=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.n_iter = n_iter
        self.bias_requires_grad = bias_requires_grad
        self.weight_requires_grad = weight_requires_grad
        # Use unified numerical config if eps not specified
        self.eps = eps if eps is not None else numerical_config.batchnorm_var_eps

        if device is None:
            device = torch.device("cpu")

        self.register_buffer(
            "running_mean",
            torch.empty(1, num_features, num_features, dtype=dtype, device=device),
        )
        self.register_buffer("running_var", torch.empty((), dtype=dtype, device=device))

        if self.affine:
            self.bias = nn.Parameter(
                torch.empty(1, num_features, num_features, dtype=dtype, device=device),
                requires_grad=bias_requires_grad,
            )
            # the weight is a scalar in log space
            self.weight = nn.Parameter(
                torch.empty((), dtype=dtype, device=device),
                requires_grad=weight_requires_grad,
            )
            self.reset_parameters()
            self.parametrize()
        else:
            self.register_buffer("bias", None)
            self.register_buffer("weight", None)
            self.reset_parameters()

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_mean[0].fill_diagonal_(1)
        self.running_var.fill_(1.0)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            # Initialize bias to identity matrix (will become zeros in tangent space)
            self.bias.zero_()
            self.bias[0].fill_diagonal_(1.0)
            # Initialize weight to 1.0 (will become 0 in log space)
            self.weight.fill_(1.0)

    def parametrize(self):
        register_parametrization(
            module=self,
            tensor_name="weight",
            parametrization=PositiveDefiniteScalar(),
        )
        register_parametrization(
            module=self,
            tensor_name="bias",
            parametrization=SymmetricPositiveDefinite(),
        )

    def forward(self, input):
        """Forward pass of the SPD Batch Normalization layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, h, n, n)`, where each slice
            along the batch dimension is an SPD matrix.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as the input.

        """
        n_samples = input.shape[0]
        if self.training:
            # Kobler et al. SPDMBN/SPDBN: estimate batch Fréchet mean via Karcher step
            batch_mean = input.mean(dim=0, keepdim=True)
            if n_samples > 1:
                for _ in range(self.n_iter):
                    # Kobler et al. (Eq. 4): P2 L132-145; Karcher flow note: P2 L163-165
                    batch_mean = karcher_mean_iteration(input, batch_mean)

            # Scalar dispersion: mean squared Frobenius norm of log at the mean (a single scalar, not variance matrix)
            mean_inv_sqrt = matrix_inv_sqrt.apply(batch_mean)
            centered = spd_centering(input, mean_inv_sqrt)
            # Kobler et al. (Eq. 6): P2 L166-175
            tangent = matrix_log.apply(centered)
            # At the Fréchet mean, the tangent mean is zero (use zeros explicitly)
            tangent_ref = torch.zeros_like(tangent[:1])
            batch_var = tangent_space_variance(tangent, tangent_ref)

            # Momentum running stats (geodesic update for mean)
            run_mean = airm_geodesic(self.running_mean, batch_mean, self.momentum)
            run_var = (
                1.0 - self.momentum
            ) * self.running_var + self.momentum * batch_var

            norm_mean = batch_mean
            norm_var = batch_var
        else:
            run_mean = self.running_mean
            run_var = self.running_var
            norm_mean = run_mean
            norm_var = run_var

        # Kobler et al. (Eq. 10): P3 L241-250 — transport to identity (centering)
        mean_inv_sqrt = matrix_inv_sqrt.apply(norm_mean)
        output = spd_centering(input, mean_inv_sqrt)

        # Kobler et al. (Eq. 10): P3 L241-250 — dispersion normalization (matrix power)
        output = matrix_power.apply(
            output,
            (self.weight if self.weight is not None else 1.0)
            / (norm_var + self.eps).sqrt(),
        )

        # Kobler et al. (Eq. 10): P3 L241-250 — re-bias via congruence at learnable mean
        if self.bias is not None:
            bias_sqrt = matrix_sqrt.apply(self.bias)
            output = spd_rebiasing(output, bias_sqrt)

        if self.training:
            self.running_mean = run_mean.detach()
            self.running_var = run_var.detach()

        return output


class BatchReNorm(nn.Module):
    """Batch Re-Normalization.

    This class implements Batch Re-Normalization, which is a variant of
    batch normalization that can be used in recurrent neural networks or as a
    Euclidean baseline for comparison with Riemannian batch normalization methods.

    Parameters
    ----------
    num_features : int
        The number of features in the input.
    momentum : float, default=0.9
        The momentum for the running mean and variance.
    rebias : bool, default=True
        If `True`, the layer has a learnable bias parameter.
    renorm : bool, default=False
        If `True`, the layer uses re-normalization.
    """

    def __init__(
        self,
        num_features,
        momentum=0.9,
        rebias=True,
        renorm=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.rebias = rebias
        self.renorm = renorm

        self.register_buffer(
            "running_mean",
            torch.empty(num_features, device=device, dtype=dtype),
        )

        if self.rebias:
            self.bias = torch.nn.Parameter(
                torch.empty(num_features, device=device, dtype=dtype),
            )
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.rebias:
            self.bias.zero_()

    def forward(self, input):
        """Forward pass of the Batch Re-Normalization layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, [h,] d)`, where `d` is the
            number of features.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as the input.
        """
        if self.training:
            mean = input.mean(dim=0, keepdim=True)
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mean
        else:
            mean = self.running_mean

        if self.renorm:
            output = input - self.running_mean
        else:
            output = input - mean

        if self.bias is not None:
            output = output + self.bias

        return output
