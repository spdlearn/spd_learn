# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import logging

from typing import Literal, Optional

import torch
import torch.nn as nn

from torch.nn.utils import parametrizations

from .. import init as spd_init
from ..functional.bilinear import bimap_increase_dim, bimap_transform


logger = logging.getLogger(__name__)


class BiMap(nn.Module):
    r"""Bilinear Mapping Layer for SPD Matrices.

    This class implements a bilinear mapping layer for Symmetric Positive
    Definite (SPD) matrices :cite:p:`huang2017riemannian`. The layer transforms
    an input SPD matrix :math:`X` as follows:

    .. math::

        Y = W^\top X W

    where :math:`W \in \mathbb{R}^{n \times k}` is a learnable weight matrix.

    **Stiefel Manifold Constraint**

    When :math:`W` has full column rank, the output remains symmetric positive
    definite. In practice, :math:`W` is constrained to the Stiefel manifold
    :math:`\text{St}(n, k) = \{W \in \mathbb{R}^{n \times k} : W^\top W = I_k\}`
    to preserve geometric structure and numerical stability during training.

    **Connection to Common Spatial Patterns (CSP)**

    From an information-geometric perspective, BiMap generalizes Common Spatial
    Patterns (CSP) :cite:p:`muller1999designing` to a learnable setting. CSP
    learns spatial filters :math:`W` from class-wise covariance matrices by
    maximizing the variance ratio between conditions via a generalized eigenvalue
    problem:

    .. math::

        \Sigma^{+} w_i = \lambda_i \Sigma^{-} w_i

    The subspace spanned by the top-:math:`k` CSP filters maximizes the symmetric
    Kullback-Leibler divergence between Gaussian models of the projected signals.
    BiMap extends this by learning :math:`W` end-to-end within a neural network,
    allowing adaptation to complex discriminative objectives beyond binary
    variance ratios.

    Parameters
    ----------
    in_features : int
        The dimensionality of the input SPD matrices.
    out_features : int
        The dimensionality of the output SPD matrices.
    depthwise : int, default=1
        The number of depthwise bilinear mappings to apply.
    parametrized : bool, default=True
        If `True`, the weight matrix `W` is parametrized as an orthogonal
        matrix via projection/retraction on the Stiefel manifold.
    orthogonal_map : str, optional
        The method used for orthogonal parametrization.
    init_method : {"kaiming_uniform", "orthogonal", "stiefel"}, default="kaiming_uniform"
        The initialization method for the weight matrix.
    seed : int, optional
        The seed for the random number generator used during Stiefel
        initialization.

    Notes
    -----
    The computational complexity scales as :math:`O(n^2 k)` for the bilinear
    product, making dimensionality reduction (:math:`k < n`) beneficial for
    large covariance matrices.

    See Also
    --------
    :class:`BiMapIncreaseDim` : Bilinear mapping for dimension expansion.
    :class:`ReEig` : Eigenvalue rectification, typically applied after BiMap.
    :class:`LogEig` : Projects to tangent space, typically the final SPD layer.
    :class:`SPDBatchNormMeanVar` : Riemannian batch normalization for SPD matrices.
    :class:`CovLayer` : Computes covariance matrices from time series input.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import BiMap
    >>> bimap = BiMap(in_features=8, out_features=4)
    >>> X = torch.randn(2, 8, 8)
    >>> X = X @ X.mT + 0.1 * torch.eye(8)  # Make SPD
    >>> Y = bimap(X)
    >>> Y.shape
    torch.Size([2, 4, 4])

    .. plot::
        :include-source:

        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from spd_learn.modules import BiMap

        torch.manual_seed(42)

        # Create an 8x8 SPD matrix
        n_in, n_out = 8, 4
        A = torch.randn(n_in, n_in)
        X = A @ A.T + 0.1 * torch.eye(n_in)
        X = X.unsqueeze(0)

        # Apply BiMap
        bimap = BiMap(in_features=n_in, out_features=n_out, parametrized=True)
        Y = bimap(X)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Input
        ax1 = axes[0]
        im1 = ax1.imshow(X[0].numpy(), cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'Input X ({n_in}x{n_in})')
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # Weight matrix W
        ax2 = axes[1]
        W = bimap.weight[0].detach().numpy()
        im2 = ax2.imshow(W, cmap='RdBu_r', aspect='auto')
        ax2.set_title(f'W ({n_in}x{n_out}, Stiefel)')
        ax2.set_xlabel('Output dim')
        ax2.set_ylabel('Input dim')
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        # W^T W (should be identity)
        ax3 = axes[2]
        WtW = (bimap.weight[0].T @ bimap.weight[0]).detach().numpy()
        im3 = ax3.imshow(WtW, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=1.1)
        ax3.set_title(r'$W^T W$ (Identity)')
        plt.colorbar(im3, ax=ax3, shrink=0.8)

        # Output
        ax4 = axes[3]
        im4 = ax4.imshow(Y[0].detach().numpy(), cmap='RdBu_r', aspect='auto')
        ax4.set_title(f'Output Y ({n_out}x{n_out})')
        plt.colorbar(im4, ax=ax4, shrink=0.8)

        plt.suptitle(r'BiMap: $Y = W^T X W$ (Bilinear Mapping)', fontweight='bold')
        plt.tight_layout()
        plt.show()
    """

    weight: nn.Parameter  # Type annotation for registered parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depthwise: int = 1,
        parametrized: bool = True,
        orthogonal_map: Optional[str] = None,
        init_method: Literal[
            "kaiming_uniform", "orthogonal", "stiefel"
        ] = "kaiming_uniform",
        seed: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if init_method not in ["kaiming_uniform", "orthogonal", "stiefel"]:
            raise ValueError(
                f"Unknown init_method: '{init_method}'. Choose from "
                "'kaiming_uniform', 'orthogonal', 'stiefel'."
            )

        if not parametrized and orthogonal_map is not None:
            raise ValueError("orthogonal_map is only used when parametrized is True")

        self._in_features = in_features
        self._out_features = out_features
        self._depthwise = depthwise
        self.parametrized = parametrized
        self.increase_dim = None
        self.orthogonal_map = orthogonal_map
        self.init_method = init_method
        self.seed = seed

        if out_features > in_features:
            self.increase_dim = BiMapIncreaseDim(
                in_features, out_features, device=device, dtype=dtype
            )
            self._in_features = out_features

        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.empty(
                    [self._depthwise, self._in_features, self._out_features],
                    device=device,
                    dtype=dtype,
                ),
                requires_grad=True,
            ),
        )

        self.reset_parameters()

        if self.parametrized:
            parametrizations.orthogonal(
                module=self, name="weight", orthogonal_map=self.orthogonal_map
            )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize weight matrix according to the specified method."""
        if self.init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weight, a=0.01)
        elif self.init_method == "orthogonal":
            nn.init.orthogonal_(self.weight)
        elif self.init_method == "stiefel":
            spd_init.stiefel_(self.weight, seed=self.seed)
        else:
            raise ValueError(
                f"Internal error: Invalid init_method '{self.init_method}'"
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply bilinear transformation to input SPD matrices.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD matrices with shape `(..., n, n)`.

        Returns
        -------
        torch.Tensor
            Transformed SPD matrices with shape `(..., k, k)`.
        """
        if self.increase_dim:
            X = self.increase_dim(X)
        return bimap_transform(X, self.weight)


class BiMapIncreaseDim(nn.Module):
    r"""Bilinear Mapping Layer for SPD Matrix Dimensionality Expansion.

    This layer transforms input SPD matrices from shape
    `(..., in_features, in_features)` to `(..., out_features, out_features)`
    using a semi-orthogonal projection and identity padding, preserving the
    SPD property.

    The transformation is defined as:

    .. math::
        Y = P + W X W^T

    where :math:`X` is the input SPD matrix, :math:`W` is a semi-orthogonal
    projection matrix, and :math:`P` is an identity padding matrix.

    Parameters
    ----------
    in_features : int
        Dimensionality of input SPD matrices.
    out_features : int
        Target dimensionality of output SPD matrices.
    device : torch.device, optional
        Target device for layer parameters.
    dtype : torch.dtype, optional
        Data type for layer parameters.
    """

    projection_matrix: torch.Tensor  # Type annotation for registered buffer
    add: torch.Tensor  # Type annotation for registered buffer

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(BiMapIncreaseDim, self).__init__()

        if out_features < in_features:
            raise ValueError("Output features must be >= input features")

        self.register_buffer(
            "projection_matrix",
            torch.eye(out_features, in_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "add",
            torch.diag((torch.arange(out_features, device=device) >= in_features)).to(
                dtype=dtype
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BiMapIncreaseDim layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(..., in_features, in_features)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(..., out_features, out_features)`.
        """
        orig_ndim = input.ndim

        if orig_ndim == 3:
            input = input.unsqueeze(1)

        projection_matrix = self.projection_matrix.view(
            1, 1, *self.projection_matrix.shape
        ).to(input.dtype)
        padding_matrix = self.add.view(1, 1, *self.add.shape).to(input.dtype)

        output = bimap_increase_dim(input, projection_matrix, padding_matrix)

        if orig_ndim == 3:
            output = output.squeeze(1)

        return output
