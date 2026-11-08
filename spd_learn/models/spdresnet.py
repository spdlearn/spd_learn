# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spd_learn.functional import covariance, ensure_sym
from spd_learn.modules import (
    BiMap,
    CovLayer,
    LogEig,
    LogEuclideanResidual,
    ReEig,
    Shrinkage,
    TraceNorm,
    Vech,
)


class SPDResNet(nn.Module):
    r"""Riemannian Residual Network on the SPD Manifold (SPDResNet).

    This class implements the Riemannian Residual Network architecture on the
    Symmetric Positive Definite (SPD) manifold, integrating spectral 1D-CNN
    residual blocks :cite:p:`chen2024alem` with learnable Adaptive Log-Euclidean
    Metric (ALEM) deformation.

    The architecture consists of:
    1. **Preprocessing**: Covariance estimation via
       :class:`~spd_learn.modules.CovLayer`,
       trace normalization via :class:`~spd_learn.modules.TraceNorm`, and learnable
       shrinkage regularization via :class:`~spd_learn.modules.Shrinkage`.
    2. **Riemannian Residual Stages**: Dimension reduction via
       :class:`~spd_learn.modules.BiMap`
       and non-linear eigenvalue rectification via :class:`~spd_learn.modules.ReEig`,
       followed by :class:`SPDResBlock` residual blocks on the SPD manifold.
    3. **Riemannian-to-Euclidean Projection**: Eigenvalue deformation via
       :class:`AdaptiveLogEuclidean` followed by upper triangular vectorization
       via :class:`~spd_learn.modules.Vech` (or standard
       :class:`~spd_learn.modules.LogEig`
       when ``use_alem=False``).
    4. **Classification Head**: Dropout regularization and a fully connected
       linear layer.

    Parameters
    ----------
    n_chans : int, default=22
        Number of input channels (electrodes in EEG).
    n_outputs : int, default=2
        Number of output classes for classification.
    bimap_dims : tuple of int, default=(16, 12)
        Subspace dimensions for successive BiMap stages.
    num_res_blocks : int, default=2
        Number of :class:`SPDResBlock` modules in each stage.
    use_alem : bool, default=True
        Whether to use the Adaptive Log-Euclidean Metric (:class:`AdaptiveLogEuclidean`)
        deformation before vectorization. If ``False``, standard
        :class:`~spd_learn.modules.LogEig`
        is used.
    drop_prob : float, default=0.0
        Dropout probability applied before the linear classification layer.
    input_type : {"raw", "cov"}, default="raw"
        Format of input data ('raw' time series or precomputed 'cov' matrices).
    cov_method : callable, default=covariance
        Method to compute covariance when `input_type` is 'raw'.
    threshold : float, default=1e-4
        Eigenvalue rectification threshold for :class:`~spd_learn.modules.ReEig`.
    upper : bool, default=True
        If `True`, only upper-triangular elements are used in tangent projection.
    device : torch.device, optional
        Device on which module parameters are allocated. Default is None.
    dtype : torch.dtype, optional
        Data type of module parameters. Default is None.

    See Also
    --------
    :class:`SPDResBlock` : Riemannian residual block with spectral 1D-CNN.
    :class:`AdaptiveLogEuclidean` : Learnable eigenvalue deformation metric.
    :class:`GyroSPD` : Manifold neural network using gyrovector space operations.

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import SPDResNet
    >>> model = SPDResNet(n_chans=22, n_outputs=2)
    >>> x = torch.randn(2, 22, 1000)
    >>> out = model(x)
    >>> out.shape
    torch.Size([2, 2])
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        bimap_dims: tuple[int, ...] = (16, 12),
        num_res_blocks: int = 2,
        use_alem: bool = True,
        drop_prob: float = 0.0,
        input_type: str = "raw",
        cov_method=covariance,
        threshold: float = 1e-4,
        upper: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if input_type not in ("raw", "cov"):
            raise ValueError(
                f"Unknown input_type: '{input_type}'. Expected 'raw' or 'cov'."
            )
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.bimap_dims = tuple(bimap_dims)
        self.num_res_blocks = num_res_blocks
        self.use_alem = use_alem
        self.drop_prob = drop_prob
        self.input_type = input_type
        self.cov_method = cov_method
        self.threshold = threshold
        self.upper = upper

        self.cov: nn.Module
        if input_type == "raw":
            self.cov = CovLayer(method=cov_method, device=device, dtype=dtype)
        else:
            self.cov = nn.Identity()

        self.trace_norm = TraceNorm()
        self.shrinkage = Shrinkage(n_chans, device=device, dtype=dtype)

        stages: list[nn.Module] = []
        in_dim = n_chans
        for out_dim in self.bimap_dims:
            stages.append(BiMap(in_dim, out_dim, device=device, dtype=dtype))
            stages.append(ReEig(threshold=threshold, device=device, dtype=dtype))
            for _ in range(num_res_blocks):
                stages.append(
                    SPDResBlock(
                        num_features=out_dim,
                        eps=threshold,
                        device=device,
                        dtype=dtype,
                    )
                )
            in_dim = out_dim
        self.stages = nn.Sequential(*stages)

        final_dim = self.bimap_dims[-1] if len(self.bimap_dims) > 0 else n_chans
        if use_alem:
            self.alem = AdaptiveLogEuclidean(
                final_dim, return_spd=False, eps=threshold, device=device, dtype=dtype
            )
            self.vech = Vech(upper=upper, preserve_norm=True)
            n_features = final_dim * (final_dim + 1) // 2
        else:
            self.logeig = LogEig(upper=upper)
            n_features = final_dim * (final_dim + 1) // 2 if upper else final_dim**2

        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(n_features, n_outputs, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPDResNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. If raw signal, shape is `(batch_size, n_chans, n_times)`.
            If precomputed covariance, shape is `(batch_size, n_chans, n_chans)`.

        Returns
        -------
        torch.Tensor
            Class logits of shape `(batch_size, n_outputs)`.
        """
        if self.input_type == "raw":
            if x.ndim == 3 and x.shape[-1] != x.shape[-2]:
                x = self.cov(x)
        else:
            x = self.cov(x)
        x = self.trace_norm(x)
        x = self.shrinkage(x)
        x = self.stages(x)
        if self.use_alem:
            x = self.alem(x)
            x = self.vech(x)
        else:
            x = self.logeig(x)
        x = self.dropout(x)
        return self.classifier(x)


class SPDResBlock(nn.Module):
    r"""Spectral 1D-CNN Riemannian Residual Block on the SPD Manifold.

    This module implements a Riemannian residual block for SPD manifolds
    :cite:p:`chen2024alem`. Given an SPD matrix :math:`X \in \mathcal{S}_{++}^d`,
    it computes the spectral decomposition :math:`X = U \Lambda U^\top`, extracts
    log-eigenvalues :math:`\log(\text{clamp}(\Lambda, \varepsilon))`, maps them through
    a 1D spectral convolutional network, reconstructs the SPD residual
    :math:`F(X) = U \text{Diag}(\exp(s)) U^\top`, and combines it with the skip
    connection via the Log-Euclidean geodesic residual:

    .. math::

        Z = \exp(\log(X) + \log(F(X)))

    Parameters
    ----------
    num_features : int, optional
        Matrix dimension :math:`d`. Default is None.
    hidden_channels : int, default=16
        Number of hidden channels in the 1D spectral CNN.
    eps : float, default=1e-6
        Lower clamp threshold for eigenvalues to maintain strict positive definiteness.
    device : torch.device, optional
        Device on which module parameters are allocated. Default is None.
    dtype : torch.dtype, optional
        Data type of module parameters. Default is None.

    See Also
    --------
    :class:`SPDResNet` : Complete ResNet architecture using this block.
    :class:`~spd_learn.modules.LogEuclideanResidual` : Log-Euclidean skip connection.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import SPDResBlock
    >>> block = SPDResBlock(num_features=8)
    >>> X = torch.randn(2, 8, 8)
    >>> X = X @ X.mT + 0.1 * torch.eye(8)
    >>> out = block(X)
    >>> out.shape
    torch.Size([2, 8, 8])
    """

    def __init__(
        self,
        num_features: int | None = None,
        hidden_channels: int = 16,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.eps = eps

        self.cnn = nn.Sequential(
            nn.Conv1d(
                1,
                hidden_channels,
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channels,
                1,
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
        )
        self.residual = LogEuclideanResidual(device=device, dtype=dtype)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize residual spectral CNN near the identity map."""
        last_conv = self.cnn[2]
        if isinstance(last_conv, nn.Conv1d):
            nn.init.normal_(last_conv.weight, mean=0.0, std=0.01)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass computing the Riemannian residual transformation.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD tensor of shape `(..., d, d)`.

        Returns
        -------
        torch.Tensor
            Output SPD tensor of shape `(..., d, d)`.
        """
        eigvals, eigvecs = torch.linalg.eigh(X)
        clamped_eigvals = torch.clamp(eigvals, min=self.eps)
        log_eigvals = torch.log(clamped_eigvals)

        batch_shape = log_eigvals.shape[:-1]
        d = log_eigvals.shape[-1]
        log_flat = log_eigvals.reshape(-1, 1, d)
        s_flat = self.cnn(log_flat)
        s = 3.0 * torch.tanh(s_flat.view(*batch_shape, d) / 3.0)

        out_eigvals = torch.exp(log_eigvals + s)
        return ensure_sym(eigvecs @ torch.diag_embed(out_eigvals) @ eigvecs.mT)


class AdaptiveLogEuclidean(nn.Module):
    r"""Adaptive Log-Euclidean Metric (ALEM) Layer.

    This module implements the learnable eigenvalue deformation of the Adaptive
    Log-Euclidean Metric framework :cite:p:`chen2024alem`. Given an SPD matrix
    :math:`X = U \Lambda U^\top`, it deforms the eigenvalues via a learnable positive
    power :math:`p` and an affine scale/shift in the deformed log-eigenvalue space:

    .. math::

        p = \text{softplus}(p_{\text{raw}}) + 0.1

    .. math::

        \tilde{\lambda}_i = w_i \frac{\lambda_i^p - \lambda_i^{-p}}{2p} + b_i

    When ``return_spd=False``, the module returns the symmetric tangent matrix:

    .. math::

        S = U \text{Diag}(\tilde{\lambda}) U^\top \in \mathcal{S}_d

    When ``return_spd=True``, it projects back to the SPD manifold:

    .. math::

        Y = U \text{Diag}(\exp(\tilde{\lambda})) U^\top \in \mathcal{S}_{++}^d

    Parameters
    ----------
    num_features : int
        Dimension :math:`d` of the SPD matrices.
    return_spd : bool, default=False
        If ``True``, exponentiates the deformed eigenvalues and returns an SPD matrix.
        If ``False``, returns a symmetric tangent matrix.
    eps : float, default=1e-6
        Lower clamp threshold for eigenvalues.
    device : torch.device, optional
        Device on which module parameters are allocated. Default is None.
    dtype : torch.dtype, optional
        Data type of module parameters. Default is None.

    See Also
    --------
    :class:`SPDResNet` : Network architecture utilizing ALEM.
    :class:`~spd_learn.modules.LogEig` : Standard static matrix logarithm.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import AdaptiveLogEuclidean
    >>> alem = AdaptiveLogEuclidean(num_features=4, return_spd=False)
    >>> X = torch.randn(2, 4, 4)
    >>> X = X @ X.mT + 0.1 * torch.eye(4)
    >>> out = alem(X)
    >>> out.shape
    torch.Size([2, 4, 4])
    """

    def __init__(
        self,
        num_features: int,
        return_spd: bool = False,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.return_spd = return_spd
        self.eps = eps

        self.p_raw = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self.weight = nn.Parameter(torch.ones(num_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype))

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset learnable parameters to initial identity-like values."""
        nn.init.zeros_(self.p_raw)
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass applying adaptive eigenvalue deformation.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD tensor of shape `(..., d, d)`.

        Returns
        -------
        torch.Tensor
            Deformed tensor of shape `(..., d, d)`. Symmetric tangent matrix if
            ``return_spd=False``, SPD matrix if ``return_spd=True``.
        """
        eigvals, eigvecs = torch.linalg.eigh(X)
        eigvals = torch.clamp(eigvals, min=self.eps)

        p = F.softplus(self.p_raw) + 0.1
        deformed = (torch.pow(eigvals, p) - torch.pow(eigvals, -p)) / (2.0 * p)
        tilde_lambda = self.weight * deformed + self.bias

        if self.return_spd:
            lambda_spd = torch.exp(torch.clamp(tilde_lambda, min=-50.0, max=50.0))
            out = eigvecs @ torch.diag_embed(lambda_spd) @ eigvecs.mT
        else:
            out = eigvecs @ torch.diag_embed(tilde_lambda) @ eigvecs.mT

        return ensure_sym(out)
