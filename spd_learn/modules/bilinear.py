# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import logging

from typing import Literal, Optional

import torch
import torch.nn as nn

from torch.nn.utils import parametrizations, parametrize

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
    orthogonal_map : {"cayley", "matrix_exp", "householder"}, optional
        The method used for orthogonal parametrization. If `None`, defaults to "cayley".
        When explicitly using ``"householder"``, exclude the parametrization's
        original parameter from optimizer weight decay: decaying its fixed sign
        coordinates can destroy orthogonality in PyTorch.
    init_method : {"kaiming_uniform", "orthogonal", "stiefel"}, default="kaiming_uniform"
        The initialization method for the weight matrix.

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

    weight: torch.Tensor  # A derived tensor when orthogonally parametrized

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depthwise: int = 1,
        parametrized: bool = True,
        orthogonal_map: Optional[Literal["cayley", "matrix_exp", "householder"]] = None,
        init_method: Literal[
            "kaiming_uniform", "orthogonal", "stiefel"
        ] = "kaiming_uniform",
        orthogonal: bool = True,
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

        if not orthogonal:
            parametrized = False

        if not parametrized and orthogonal_map is not None:
            raise ValueError("orthogonal_map is only used when parametrized is True")

        self._in_features = in_features
        self._out_features = out_features
        self._depthwise = depthwise
        self.parametrized = parametrized
        self.increase_dim = None
        self.orthogonal_map = orthogonal_map
        self.init_method = init_method
        self.orthogonal = orthogonal
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

        if seed is not None:
            torch.manual_seed(seed)

        self.reset_parameters()

        if self.parametrized:
            parametrizations.orthogonal(
                module=self,
                name="weight",
                orthogonal_map=(
                    "cayley" if self.orthogonal_map is None else self.orthogonal_map
                ),
            )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize weights, including an existing orthogonal parametrization."""
        is_parametrized = parametrize.is_parametrized(self, "weight")
        reference = (
            self.get_parameter("parametrizations.weight.original")
            if is_parametrized
            else self.weight
        )
        weight = torch.empty_like(reference)
        if self.init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(weight, a=0.01)
        elif self.init_method == "orthogonal":
            for w in weight:
                nn.init.orthogonal_(w)
        elif self.init_method == "stiefel":
            spd_init.stiefel_(weight)
        else:
            raise ValueError(
                f"Internal error: Invalid init_method '{self.init_method}'"
            )
        if is_parametrized:
            # Assign through the parametrization to update its stored state.
            self.weight = weight
        else:
            reference.copy_(weight)

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


class GraphBiMap(nn.Module):
    r"""Graph Bilinear Mapping Layer for SPD Matrices.

    This layer implements graph aggregation followed by node-wise bilinear
    transformation on Symmetric Positive Definite (SPD) manifolds as introduced
    in GraphCSPNet :cite:p:`ju2023graph`.

    Given a collection of SPD matrices :math:`\{X_i\}_{i=1}^{N}` associated with
    nodes on a graph (e.g., time-frequency segments), the layer first performs
    graph spatial aggregation using a row-stochastic propagation matrix :math:`P`:

    .. math::

        \tilde{X}_i = \sum_{j=1}^N P_{ij} X_j

    Because :math:`P_{ij} \ge 0` and :math:`\sum_{j=1}^N P_{ij} = 1`, each
    :math:`\tilde{X}_i` is a convex combination of SPD matrices and strictly
    preserves positive definiteness.

    Subsequently, a node-wise (depthwise) bilinear mapping is applied to each node:

    .. math::

        Y_i = W_i^\top \tilde{X}_i W_i

    where :math:`W_i \in \text{St}(d_{\text{in}}, d_{\text{out}})` is an
    orthogonally parametrized spatial filter for node :math:`i`.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    in_features : int
        Dimensionality of the input SPD matrices.
    out_features : int
        Dimensionality of the output SPD matrices.
    graph_matrix : torch.Tensor, optional
        Initial graph matrix of shape `(n_nodes, n_nodes)`. If `None`, defaults
        to identity matrix.
    learnable_graph : bool, default=False
        If `True`, the graph propagation matrix is learned via parameterized logits
        with softmax normalization. If `False`, it remains fixed as a registered buffer.
    parametrized : bool, default=True
        Whether to orthogonally parametrize the bilinear transformation weights.
    orthogonal_map : {"cayley", "matrix_exp", "householder"}, optional
        The method used for orthogonal parametrization.
    orthogonal : bool, default=True
        If `False`, disables orthogonal parametrization.
    init_method : {"kaiming_uniform", "orthogonal", "stiefel"}, default="kaiming_uniform"
        Initialization method for the bilinear transformation weights.
    seed : int, optional
        Random seed for parameter initialization.
    device : torch.device, optional
        Target device for layer parameters.
    dtype : torch.dtype, optional
        Data type for layer parameters.

    See Also
    --------
    :class:`BiMap` : Bilinear mapping layer without graph aggregation.
    :class:`BiMapIncreaseDim` : Bilinear mapping for dimension expansion.
    :class:`~spd_learn.models.GraphCSPNet` : Complete architecture using GraphBiMap.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import GraphBiMap
    >>> layer = GraphBiMap(n_nodes=4, in_features=6, out_features=3)
    >>> X = torch.randn(2, 4, 6, 6)
    >>> X = X @ X.mT + 0.1 * torch.eye(6)
    >>> Y = layer(X)
    >>> Y.shape
    torch.Size([2, 4, 3, 3])
    """

    def __init__(
        self,
        n_nodes: int,
        in_features: int,
        out_features: int,
        graph_matrix: Optional[torch.Tensor] = None,
        learnable_graph: bool = False,
        parametrized: bool = True,
        orthogonal_map: Optional[Literal["cayley", "matrix_exp", "householder"]] = None,
        orthogonal: bool = True,
        init_method: str = "kaiming_uniform",
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.learnable_graph = learnable_graph
        self.parametrized = parametrized
        self.orthogonal_map = orthogonal_map
        self.orthogonal = orthogonal
        self.init_method = init_method
        self.seed = seed

        if learnable_graph:
            if graph_matrix is not None:
                init_logits = graph_matrix.clone().detach().to(device=device, dtype=dtype)
            else:
                init_logits = torch.eye(n_nodes, device=device, dtype=dtype)
            self.graph_logits = nn.Parameter(init_logits)
        else:
            if graph_matrix is not None:
                P = graph_matrix.clone().detach().to(device=device, dtype=dtype)
            else:
                P = torch.eye(n_nodes, device=device, dtype=dtype)
            self.register_buffer("graph_matrix", P)

        self.bimap = BiMap(
            in_features=in_features,
            out_features=out_features,
            depthwise=n_nodes,
            parametrized=parametrized,
            orthogonal_map=orthogonal_map,
            orthogonal=orthogonal,
            init_method=init_method,
            seed=seed,
            device=device,
            dtype=dtype,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GraphBiMap layer.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD matrices with shape `(..., n_nodes, in_features, in_features)`.

        Returns
        -------
        torch.Tensor
            Transformed SPD matrices with shape `(..., n_nodes, out_features, out_features)`.
        """
        if X.ndim < 3:
            raise ValueError(
                f"Expected input with at least 3 dimensions (..., n_nodes, in_features, in_features), "
                f"got shape {X.shape}"
            )
        if X.shape[-3] != self.n_nodes:
            raise ValueError(
                f"Expected n_nodes={self.n_nodes} at dim -3, got {X.shape[-3]} in shape {X.shape}"
            )
        if X.shape[-2] != self.in_features or X.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected matrix size ({self.in_features}, {self.in_features}), "
                f"got ({X.shape[-2]}, {X.shape[-1]}) in shape {X.shape}"
            )

        if self.learnable_graph:
            if self.graph_logits.is_complex():
                P = torch.softmax(self.graph_logits.real, dim=-1).to(X.dtype)
            else:
                P = torch.softmax(self.graph_logits.to(X.dtype), dim=-1)
        else:
            if self.graph_matrix.is_complex():
                P = self.graph_matrix.real.to(X.dtype)
            else:
                P = self.graph_matrix.to(X.dtype)

        X_agg = torch.einsum("ij,...jkl->...ikl", P, X)

        return self.bimap(X_agg)
