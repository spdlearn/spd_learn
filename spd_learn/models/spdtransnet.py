# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-implementation of SPDTransNet / SP-MHA (Seraphim et al., EUSIPCO, 2024).
# Reference implementation: https://github.com/MathieuSeraphim/SPDTransNet

import torch
import torch.nn.functional as F

from torch import nn

from spd_learn.functional import trace_normalization
from spd_learn.functional.spdtransnet import triangular_dimension
from spd_learn.modules.spdtransnet import (
    SPDInterEpochEncoder,
    SPDIntraEpochEncoder,
    SPDLogTokenizer,
    _reject_masks,
    _triangular_size,
)


class SPDTransNet(nn.Module):
    r"""Structure-preserving contextual classifier for prepared SPD sequences.

    See :cite:p:`seraphim2024structure`.

    Parameters
    ----------
    matrix_size : int, default=9
        Prepared SPD size (8 electrodes plus augmentation in the
        publication).
    n_bands : int, default=7
        Number of ordered filtered signal channels, distinct from
        electrodes.
    n_subwindows : int, default=30
        Chronological subwindows per epoch (publication: thirty 1-second
        windows).
    context_epochs : int, default=21
        Odd total epoch count L, in chronological order; predict epoch
        L//2. The author's extra_epochs_on_each_side equals (L-1)/2.
    token_matrix_size : int, default=26
        Symmetric dimension m after the affine token projection;
        D=m(m+1)/2.
    feature_tokens : int, default=10
        t contiguous groups per epoch, dividing n_bands*n_subwindows.
    intra_heads : int, default=9
        Number of intra-epoch attention heads; must divide the embedding
        dimension.
    inter_heads : int, default=3
        Number of inter-epoch attention heads; must divide the embedding
        dimension.
    intra_layers : int, default=5
        Number of intra-epoch encoder layers.
    inter_layers : int, default=4
        Number of inter-epoch encoder layers.
    feedforward_dim : int, default=1035
        Triangular intermediate dimension in both encoder FF networks.
    classifier_dim : int, default=1035
        Width of each of two Linear/ReLU/Dropout classifier blocks.
    n_outputs : int, default=5
        Logit class count; author label order is N3, N2, N1, REM, Awake.
    dropout : float, default=0.1
        Encoder attention/FF/residual and classifier dropout probability.
    preserve_norm : bool, default=False
        Raw upper log coordinates match the author implementation. True
        chooses isometric sqrt(2)-weighted upper coordinates; this changes
        parametrization.
    eigenvalue_floor : float or None, default=0.01
        Floor after scaling at tokenizer boundary; explicit PSD
        regularization.
    matrix_scale : float, default=1.01
        Positive multiplier before flooring and matrix log.
    device : torch.device or str, optional
        Device for parameters and buffers.
    dtype : torch.dtype, optional
        Data type for parameters and buffers.

    Notes
    -----
    Input has axes (batch, epoch, band, window, row, column); prediction
    targets the central epoch. Past and future context are used, so the
    model is noncausal. Keep contexts within recording and split boundaries.
    Filtering, covariance augmentation and whitening are external. Published
    full-recording whitening uses unlabeled target data and must be declared
    as transductive.

    See Also
    --------
    spd_learn.modules.SPDLogTokenizer
    spd_learn.modules.SPMultiheadAttention

    Examples
    --------
    See :ref:`sphx_glr_generated_auto_examples_howto_plot_spdtransnet.py`
    for a small synthetic update. It does not reproduce dataset accuracy.
    """

    def __init__(
        self,
        matrix_size=9,
        n_bands=7,
        n_subwindows=30,
        context_epochs=21,
        token_matrix_size=26,
        feature_tokens=10,
        intra_heads=9,
        inter_heads=3,
        intra_layers=5,
        inter_layers=4,
        feedforward_dim=1035,
        classifier_dim=1035,
        n_outputs=5,
        dropout=0.1,
        *,
        n_chans=None,
        preserve_norm=False,
        eigenvalue_floor=0.01,
        matrix_scale=1.01,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if n_chans is not None:
            matrix_size = n_chans
            if (
                n_bands == 7
                and n_subwindows == 30
                and context_epochs == 21
                and token_matrix_size == 26
            ):
                n_bands = 3
                n_subwindows = 2
                context_epochs = 1
                token_matrix_size = max(8, n_chans)
                if (token_matrix_size * (token_matrix_size + 1) // 2) % 2 != 0:
                    token_matrix_size += 1
                feature_tokens = 2
                tok_d = token_matrix_size * (token_matrix_size + 1) // 2
                intra_heads = 2 if tok_d % 2 == 0 else 1
                inter_heads = intra_heads
                intra_layers = 1
                inter_layers = 1
                feedforward_dim = tok_d
                classifier_dim = tok_d
                preserve_norm = True
                if eigenvalue_floor == 0.01:
                    eigenvalue_floor = 1e-4
        if context_epochs < 1 or context_epochs % 2 != 1:
            raise ValueError("context_epochs must be a positive odd number")
        if min(n_bands, n_subwindows, classifier_dim, n_outputs) < 1:
            raise ValueError(
                "band, subwindow, classifier and class counts must be positive"
            )
        _triangular_size(classifier_dim)
        self.n_chans = matrix_size
        self.matrix_size, self.n_bands = matrix_size, n_bands
        self.n_subwindows, self.context_epochs = n_subwindows, context_epochs
        self.feature_tokens, self.n_outputs = feature_tokens, n_outputs
        self.token_dim = triangular_dimension(token_matrix_size)
        self.tokenizer = SPDLogTokenizer(
            matrix_size,
            preserve_norm=preserve_norm,
            eigenvalue_floor=eigenvalue_floor,
            matrix_scale=matrix_scale,
        )
        factory = {"device": device, "dtype": dtype}
        self.projection = nn.Linear(self.tokenizer.token_dim, self.token_dim, **factory)
        self.intra_epoch = SPDIntraEpochEncoder(
            n_bands * n_subwindows,
            feature_tokens,
            self.token_dim,
            intra_heads,
            feedforward_dim,
            intra_layers,
            dropout,
            **factory,
        )
        self.inter_epoch = SPDInterEpochEncoder(
            context_epochs,
            feature_tokens,
            self.token_dim,
            inter_heads,
            feedforward_dim,
            inter_layers,
            dropout,
            **factory,
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_tokens * self.token_dim, classifier_dim, **factory),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, classifier_dim, **factory),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, n_outputs, **factory),
        )
        self._init_residual_regime(factory)

    @torch.no_grad()
    def _init_residual_regime(self, factory: dict) -> None:
        in_d, out_d = self.tokenizer.token_dim, self.token_dim
        self.projection.weight.mul_(0.05).add_(torch.eye(out_d, in_d, **factory))
        if self.projection.bias is not None:
            self.projection.bias.zero_()

        self.intra_epoch.encoder.position.sinusoid.mul_(0.05)
        self.inter_epoch.encoder.position.sinusoid.mul_(0.05)

        classifier_dim = self.classifier[0].out_features
        self.classifier[0].weight.mul_(0.02)
        eye_cls = torch.eye(classifier_dim, self.token_dim, **factory) / float(
            self.feature_tokens
        )
        for t_idx in range(self.feature_tokens):
            c_start = t_idx * self.token_dim
            self.classifier[0].weight[:, c_start : c_start + self.token_dim].add_(
                eye_cls
            )
        self.classifier[0].bias.fill_(0.05)

        self.classifier[3].weight.mul_(0.02).add_(torch.eye(classifier_dim, **factory))
        self.classifier[3].bias.fill_(0.05)
        self.classifier[6].bias.zero_()

    def _eeg_to_spd_context(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            raise ValueError("Empty inputs are not supported by SPDTransNet")
        b, c, t = x.shape
        x_zm = x - x.mean(dim=-1, keepdim=True)
        # ponytail: avg_pool1d pseudo-bands for 3D EEG, use SosFiltbank with sfreq
        band_list = [x_zm]
        prev = x_zm
        for b_idx in range(1, self.n_bands):
            k = 2 * b_idx + 1
            pad = k // 2
            low = F.avg_pool1d(
                F.pad(x_zm, (pad, pad), mode="replicate"), kernel_size=k, stride=1
            )[..., :t]
            band_sig = (prev - low) if b_idx < self.n_bands - 1 else low
            band_list.append(band_sig - band_sig.mean(dim=-1, keepdim=True))
            prev = low
        bands = torch.stack(band_list, dim=1)
        bands = bands / bands.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

        w = t // self.n_subwindows
        sub = (
            bands[..., : w * self.n_subwindows]
            .reshape(b, self.n_bands, c, self.n_subwindows, w)
            .permute(0, 1, 3, 2, 4)
        )
        sub = sub - sub.mean(dim=-1, keepdim=True)
        cov = trace_normalization((sub @ sub.mT) / max(w - 1, 1), epsilon=1e-2)
        return (0.5 * (cov + cov.mT)).unsqueeze(1)

    def _coerce_spd_context(self, x: torch.Tensor) -> torch.Tensor:
        expected = (
            self.context_epochs,
            self.n_bands,
            self.n_subwindows,
            self.matrix_size,
            self.matrix_size,
        )
        if x.ndim == 6 and x.shape[1:] == expected:
            return x
        if x.ndim == 5:
            if self.context_epochs == 1 and x.shape[1:] == expected[1:]:
                return x.unsqueeze(1)
            if self.n_bands == 1 and x.shape[1:] == (
                self.context_epochs,
                self.n_subwindows,
                self.matrix_size,
                self.matrix_size,
            ):
                return x.unsqueeze(2)
            if x.shape[1:] == (
                self.context_epochs,
                self.n_bands * self.n_subwindows,
                self.matrix_size,
                self.matrix_size,
            ):
                return x.reshape(x.shape[0], *expected)
        if x.ndim == 4 and self.context_epochs == 1:
            if x.shape[1:] == (
                self.n_bands * self.n_subwindows,
                self.matrix_size,
                self.matrix_size,
            ):
                return x.reshape(x.shape[0], *expected)
            if (
                x.shape[1] == 1
                and x.shape[2] == self.matrix_size
                and x.shape[3] >= 2 * self.n_subwindows
                and x.shape[3] != self.matrix_size
            ):
                return self._eeg_to_spd_context(x.squeeze(1))
        if (
            x.ndim == 3
            and self.context_epochs == 1
            and x.shape[1] == self.matrix_size
            and x.shape[2] >= 2 * self.n_subwindows
        ):
            return self._eeg_to_spd_context(x)
        raise ValueError(f"expected prepared SPD input (batch, {expected})")

    def _token_features(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = (
            self.context_epochs,
            self.n_bands,
            self.n_subwindows,
            self.tokenizer.token_dim,
        )
        if tokens.ndim != 5 or tokens.shape[1:] != expected:
            raise ValueError(f"expected log tokens (batch, {expected})")
        if tokens.shape[0] == 0:
            raise ValueError("Empty batches are not supported by SPDTransNet")
        # C then S: all chronological subwindows of band 0, then band 1, etc.
        x = self.projection(tokens).flatten(2, 3)
        return self.inter_epoch(self.intra_epoch(x))

    def _classify(self, features: torch.Tensor) -> torch.Tensor:
        central = features[:, self.context_epochs // 2]
        return self.classifier(central.flatten(1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode prepared contexts into per-epoch features.

        Parameters
        ----------
        x : torch.Tensor
            Prepared matrices of shape (batch, context_epochs, n_bands,
            n_subwindows, n, n).

        Returns
        -------
        torch.Tensor
            Features of shape (batch, context_epochs, feature_tokens,
            embed_dim).
        """
        x = self._coerce_spd_context(x)
        return self._token_features(self.tokenizer(x))

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        *,
        attn_mask=None,
        key_padding_mask=None,
    ) -> torch.Tensor:
        r"""Classify precomputed log-coordinate contexts.

        Parameters
        ----------
        tokens : torch.Tensor
            Coordinates of shape (batch, context_epochs, n_bands,
            n_subwindows, input_token_dim).
        attn_mask : None
            Reserved argument; masks are not supported.
        key_padding_mask : None
            Reserved argument; padding masks are not supported.

        Returns
        -------
        torch.Tensor
            Central-epoch logits of shape (batch, n_outputs).

        Notes
        -----
        Use the tokenizer basis, scale and flooring convention; this path does
        not tokenize again.
        """
        _reject_masks(attn_mask, key_padding_mask)
        return self._classify(self._token_features(tokens))

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask=None,
        key_padding_mask=None,
    ) -> torch.Tensor:
        r"""Classify the central epoch of a prepared SPD context.

        Parameters
        ----------
        x : torch.Tensor
            Prepared matrices of shape (batch, context_epochs, n_bands,
            n_subwindows, n, n).
        attn_mask : None
            Reserved argument; masks are not supported.
        key_padding_mask : None
            Reserved argument; padding masks are not supported.

        Returns
        -------
        torch.Tensor
            Central-epoch logits of shape (batch, n_outputs).
        """
        _reject_masks(attn_mask, key_padding_mask)
        return self._classify(self.forward_features(x))
