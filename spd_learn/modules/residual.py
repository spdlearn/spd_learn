# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch

from torch import nn

from ..functional.metrics import log_euclidean_multiply


class LogEuclideanResidual(nn.Module):
    r"""Residual/skip connection for SPD networks using the Log-Euclidean metric.

    This module implements a Riemannian residual connection based on the
    Log-Euclidean framework :cite:p:`katsman2023riemannian`. It enables skip
    connections in SPD neural networks while respecting the manifold geometry,
    addressing a fundamental challenge in geometric deep learning.

    Notes
    -----
    Standard residual connections :math:`y = x + f(x)` from ResNets
    :cite:p:`he2016deep` cannot be directly applied to SPD manifolds because
    the sum of two SPD matrices does not preserve the geometric structure
    needed for Riemannian operations. The Log-Euclidean residual provides a
    geometrically principled alternative by performing addition in the tangent
    space (via matrix logarithm) and mapping back to the manifold (via matrix
    exponential).

    The residual connection is computed as:

    .. math::
        Z = \exp(\log(X) + \log(Y))

    where :math:`X` is the input tensor, :math:`Y` is the residual tensor, and
    :math:`\log` and :math:`\exp` denote the matrix logarithm and exponential,
    respectively.

    This formulation corresponds to the Log-Euclidean midpoint when both inputs
    have equal weight, making it a natural and geometrically meaningful choice
    for residual connections on SPD manifolds. The operation is:

    - **Commutative**: :math:`\text{residual}(X, Y) = \text{residual}(Y, X)`
    - **Identity-preserving**: :math:`\text{residual}(X, I) = X` when :math:`Y = I`
    - **SPD-preserving**: Output is guaranteed to be SPD if inputs are SPD

    The authors of :cite:t:`katsman2023riemannian` demonstrate that Riemannian
    ResNets:

    1. Outperform existing manifold neural networks on hyperbolic graph learning
    2. Achieve superior performance on video classification using SPD matrices
       (videos represented as covariance matrices)
    3. Exhibit improved training dynamics compared to non-residual architectures
    4. Require only the exponential map (geodesic information) for implementation

    Parameters
    ----------
    device : torch.device, optional
        The device on which the module will be allocated. Default: None.
    dtype : torch.dtype, optional
        The data type of the module. Default: None.

    See Also
    --------
    spd_learn.functional.matrix_log : Matrix logarithm for SPD matrices.
    spd_learn.functional.matrix_exp : Matrix exponential for symmetric matrices.
    spd_learn.functional.log_euclidean_mean : Weighted Log-Euclidean mean.

    Examples
    --------
    Basic usage with random SPD matrices:

    >>> import torch
    >>> from spd_learn.modules import LogEuclideanResidual
    >>> residual = LogEuclideanResidual()
    >>> X = torch.randn(4, 8, 8)
    >>> X = X @ X.mT + 0.1 * torch.eye(8)  # Make SPD
    >>> Y = torch.randn(4, 8, 8)
    >>> Y = Y @ Y.mT + 0.1 * torch.eye(8)  # Make SPD
    >>> Z = residual(X, Y)
    >>> Z.shape
    torch.Size([4, 8, 8])

    Using in a residual block within an SPD network:

    >>> from spd_learn.modules import BiMap, ReEig, LogEuclideanResidual
    >>> class SPDResidualBlock(torch.nn.Module):
    ...     def __init__(self, n_features):
    ...         super().__init__()
    ...         self.bimap = BiMap(n_features, n_features)
    ...         self.reeig = ReEig()
    ...         self.residual = LogEuclideanResidual()
    ...
    ...     def forward(self, x):
    ...         y = self.bimap(x)
    ...         y = self.reeig(y)
    ...         return self.residual(x, y)  # Skip connection
    >>> block = SPDResidualBlock(8)
    >>> X = torch.randn(2, 8, 8)
    >>> X = X @ X.mT + 0.1 * torch.eye(8)
    >>> out = block(X)
    >>> out.shape
    torch.Size([2, 8, 8])
    """

    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Forward pass computing the Log-Euclidean residual.

        Computes :math:`Z = \exp(\log(X) + \log(Y))`, which combines two SPD
        matrices in a geometrically principled way using the Log-Euclidean
        framework.

        Parameters
        ----------
        x : torch.Tensor
            Input SPD tensor of shape `(..., n, n)`. Must be symmetric positive
            definite.
        y : torch.Tensor
            Residual SPD tensor of shape `(..., n, n)`. Must be symmetric
            positive definite and have the same shape as `x`.

        Returns
        -------
        torch.Tensor
            The Log-Euclidean combination of `x` and `y`, with the same shape
            as the inputs. The output is guaranteed to be SPD.

        Notes
        -----
        The operation is performed in three steps:

        1. Map both inputs to the tangent space via matrix logarithm
        2. Add the tangent vectors (standard Euclidean addition)
        3. Map back to the SPD manifold via matrix exponential

        This is equivalent to computing:

        .. math::
            Z = X^{1/2} (X^{-1/2} Y X^{-1/2})^{1/2} X^{1/2}

        but the Log-Euclidean formulation is computationally more efficient
        and numerically stable.
        """
        result = log_euclidean_multiply(x, y)
        if self.device is not None or self.dtype is not None:
            return result.to(device=self.device, dtype=self.dtype)
        return result
