# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch

from torch import nn

from ..functional.regularize import ledoit_wolf, trace_normalization


class Shrinkage(nn.Module):
    r"""Learnable Shrinkage Regularization for Covariance Matrices.

    This module applies a learnable shrinkage transformation inspired by the
    Ledoit-Wolf and Oracle Approximating Shrinkage (OAS) estimators
    :cite:p:`ledoit2004well` to regularize covariance matrices:

    .. math::

        \hat{C} = (1 - \alpha) C + \alpha \cdot \frac{\text{tr}(C)}{n} \cdot I_n

    where :math:`\alpha \in [0, 1]` is the shrinkage intensity and :math:`n` is
    the matrix dimension. This convex combination interpolates between the
    empirical covariance :math:`C` and a structured target (scaled identity).

    **Why Shrinkage is Necessary**

    Reliable covariance estimation in neuroimaging faces three fundamental
    challenges :cite:p:`varoquaux2010brain`:

    1. **Curse of dimensionality**: When the number of variables exceeds the
       number of samples (:math:`n_C > n_T`), empirical covariance estimates
       become ill-conditioned or rank-deficient.

    2. **Non-Gaussian artifacts**: Outliers such as eye blinks in EEG or head
       motion in fMRI violate distributional assumptions.

    3. **Temporal non-stationarity**: Brain signals evolve over time, breaking
       i.i.d. assumptions and causing estimation errors to propagate.

    Shrinkage estimators address these issues by trading variance for bias,
    pulling extreme eigenvalues toward a common mean, which improves
    conditioning and stability of downstream analyses. Automated model
    selection methods can determine the optimal shrinkage estimator for
    MEG and EEG applications :cite:p:`engemann2015automated`.

    **Typical Pipeline**

    Shrinkage is typically the final step in covariance regularization
    :cite:p:`aristimunha2025spdlearn`:

    1. **Normalize time series**: Zero mean and unit L2 norm per channel
    2. **Compute sample covariance**: :math:`C = XX^T`
    3. **Apply shrinkage**: :math:`\hat{C} = (1-\alpha)C + \alpha \cdot
       \text{tr}(C)/n \cdot I`

    **Limitations**

    While shrinkage estimators are theoretically well-founded, they rely on
    structured targets (e.g., scaled identity) that may not capture complex
    covariance structure in densely connected brain networks. This limitation
    motivates alternative approaches such as population-level shrinkage
    :cite:p:`rahim2017population`.

    Parameters
    ----------
    n_chans : int
        The size of the square matrices expected as input.
    init_shrinkage : float, default=0.0
        The initial value for the pre-sigmoid shrinkage parameter.
        After sigmoid: 0.0 → α ≈ 0.5, negative → less shrinkage, positive → more.
    learnable : bool, default=False
        If `True`, the shrinkage parameter is learned during training.

    Notes
    -----
    The optimal shrinkage intensity depends on the sample size, dimensionality,
    and signal properties. The Ledoit-Wolf estimator provides an analytical
    formula for the optimal :math:`\alpha`, while this module allows learning
    it from data when ``learnable=True``.

    See Also
    --------
    :class:`TraceNorm` : Normalizes by trace without shrinkage.
    :class:`CovLayer` : Computes covariance matrices from time series.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import Shrinkage
    >>> shrinkage = Shrinkage(n_chans=8, init_shrinkage=0.5, learnable=True)
    >>> X = torch.randn(4, 8, 8)
    >>> X = X @ X.mT  # Make SPD
    >>> Y = shrinkage(X)
    >>> Y.shape
    torch.Size([4, 8, 8])

    .. plot::
        :include-source:

        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from spd_learn.modules import Shrinkage, CovLayer

        torch.manual_seed(42)

        # Generate synthetic data and compute covariance
        n_channels = 8
        raw_signals = torch.randn(1, n_channels, 100)
        mixing = torch.randn(n_channels, n_channels)
        raw_signals = torch.einsum('ij,bjt->bit', mixing, raw_signals)

        cov_layer = CovLayer()
        covariances = cov_layer(raw_signals)

        # Apply shrinkage with different coefficients
        shrinkage_low = Shrinkage(n_chans=n_channels, init_shrinkage=-2.0)  # ~0.12
        shrinkage_high = Shrinkage(n_chans=n_channels, init_shrinkage=2.0)  # ~0.88

        cov_low = shrinkage_low(covariances)
        cov_high = shrinkage_high(covariances)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Original eigenvalues
        eigvals_orig = torch.linalg.eigvalsh(covariances[0]).numpy()
        eigvals_low = torch.linalg.eigvalsh(cov_low[0].detach()).numpy()
        eigvals_high = torch.linalg.eigvalsh(cov_high[0].detach()).numpy()

        for ax, eigv, title, color in zip(
            axes,
            [eigvals_orig, eigvals_low, eigvals_high],
            ['Original', r'Shrinkage $\alpha \approx 0.12$', r'Shrinkage $\alpha \approx 0.88$'],
            ['#3498db', '#e74c3c', '#2ecc71']
        ):
            ax.bar(range(n_channels), sorted(eigv, reverse=True), color=color, alpha=0.8)
            ax.set_xlabel('Eigenvalue index')
            ax.set_ylabel('Eigenvalue')
            ax.set_title(title, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=min(eigv), color='red', linestyle='--', alpha=0.5)
            cond = max(eigv) / min(eigv)
            ax.text(0.95, 0.95, f'Cond: {cond:.1f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.suptitle('Shrinkage: Eigenvalue Regularization', fontweight='bold')
        plt.tight_layout()
        plt.show()
    """

    identity_matrix: torch.Tensor  # Type annotation for registered buffer

    def __init__(
        self,
        n_chans: int,
        init_shrinkage: float = 0.0,
        learnable: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.learnable = learnable
        self.device = device
        self.dtype = dtype

        initial_logit = torch.tensor(float(init_shrinkage), dtype=dtype, device=device)
        self.shrinkage_logit = nn.Parameter(initial_logit, requires_grad=learnable)

        self.register_buffer("identity_matrix", torch.eye(self.n_chans))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Shrinkage layer.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape `(..., n_chans, n_chans)`.

        Returns
        -------
        torch.Tensor
            The regularized output tensor.
        """
        if X.shape[-2:] != (self.n_chans, self.n_chans):
            raise ValueError(
                f"Input tensor X must have last two dimensions "
                f"({self.n_chans}, {self.n_chans}), but got {X.shape}"
            )

        out = ledoit_wolf(X, self.shrinkage_logit, self.identity_matrix, self.n_chans)

        if self.device is not None or self.dtype is not None:
            return out.to(device=self.device, dtype=self.dtype)
        return out


class TraceNorm(nn.Module):
    r"""Trace Normalization Layer for Scale-Invariant Covariance.

    This module normalizes each covariance matrix by its trace, producing
    matrices with unit trace, and adds a small diagonal regularization:

    .. math::

        X_{\text{out}} = \frac{X}{\text{tr}(X)} + \epsilon I

    **Geometric Interpretation**

    Trace normalization removes scale information while preserving the
    correlation structure of the covariance matrix. This is particularly
    useful when comparing covariances across subjects or sessions with
    different overall signal amplitudes.

    For SPD matrices, this operation maps to a lower-dimensional submanifold
    of unit-trace SPD matrices, which can simplify geometric computations
    while retaining the relative eigenvalue structure.

    Parameters
    ----------
    epsilon : float, default=0.0
        A small value added to the diagonal for numerical stability.
        Ensures positive definiteness when trace normalization produces
        near-singular matrices.

    See Also
    --------
    :class:`Shrinkage` : Regularization via shrinkage toward identity.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import TraceNorm
    >>> trace_norm = TraceNorm(epsilon=1e-5)
    >>> X = torch.randn(4, 8, 8)
    >>> X = X @ X.mT  # Make SPD
    >>> Y = trace_norm(X)
    >>> Y.shape
    torch.Size([4, 8, 8])

    .. plot::
        :include-source:

        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from spd_learn.modules import TraceNorm, CovLayer

        torch.manual_seed(42)

        # Generate synthetic data with different scales
        n_channels = 6
        batch_size = 4

        # Create covariances with varying scales
        covariances = []
        for scale in [0.1, 1.0, 10.0, 100.0]:
            A = torch.randn(n_channels, n_channels) * scale
            cov = A @ A.T + 0.1 * torch.eye(n_channels)
            covariances.append(cov)
        covariances = torch.stack(covariances)

        # Apply TraceNorm
        trace_norm = TraceNorm(epsilon=1e-5)
        normalized = trace_norm(covariances)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Before normalization - traces vary widely
        ax1 = axes[0]
        traces_before = [torch.trace(covariances[i]).item() for i in range(batch_size)]
        ax1.bar(range(batch_size), traces_before, color='#3498db', alpha=0.8)
        ax1.set_xlabel('Sample index')
        ax1.set_ylabel('Trace')
        ax1.set_title('Before TraceNorm', fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # After normalization - traces are ~1
        ax2 = axes[1]
        traces_after = [torch.trace(normalized[i]).item() for i in range(batch_size)]
        ax2.bar(range(batch_size), traces_after, color='#2ecc71', alpha=0.8)
        ax2.set_xlabel('Sample index')
        ax2.set_ylabel('Trace')
        ax2.set_title('After TraceNorm', fontweight='bold')
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Target (trace=1)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.5)

        plt.suptitle('TraceNorm: Scale Normalization', fontweight='bold')
        plt.tight_layout()
        plt.show()
    """

    def __init__(self, epsilon: float = 0.0, device=None, dtype=None):
        super().__init__()
        self.epsilon = epsilon
        self.device = device
        self.dtype = dtype

    def forward(self, covariances: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TraceNorm layer.

        Parameters
        ----------
        covariances : torch.Tensor
            A batch of covariance matrices with shape `(..., n, n)`.

        Returns
        -------
        torch.Tensor
            The trace-normalized covariance matrices.
        """
        result = trace_normalization(covariances, epsilon=self.epsilon)
        if self.device is not None or self.dtype is not None:
            return result.to(device=self.device, dtype=self.dtype)
        return result
