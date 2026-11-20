# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from spd_learn.models import Green, PhaseSPDNet, SPDNet


def test_green_model_initialization_and_forward():
    """Test Green model initialization and forward pass."""
    model = Green(n_outputs=2, n_chans=21)
    x = torch.randn(2, 21, 1000)  # Batch of 2, 21 channels, 1000 time points
    output = model(x)
    assert output.shape == (2, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("eigenvalue", [0.3, 1.0])
def test_green_logeig_backward_at_repeated_eigenvalues(dtype, eigenvalue):
    """Green must use the finite matrix-log derivative at isotropic covariance."""
    model = Green(n_outputs=2, n_chans=3, n_freqs_init=2, dtype=dtype)
    identity = torch.eye(3, dtype=dtype)
    covariance = (eigenvalue * identity).expand(2, 2, 3, 3).clone().requires_grad_()
    model.proj[0](covariance).sum().backward()
    expected = (identity + (1 - identity) / 2**0.5) / eigenvalue
    torch.testing.assert_close(covariance.grad, expected.expand_as(covariance))


@pytest.mark.parametrize("random_f_init", [True, False])
@pytest.mark.parametrize("shrinkage_init", [None, 0.0, 2.5])
@pytest.mark.parametrize("logref", ["logeuclid", "identity"])
@pytest.mark.parametrize("dropout", [None, 0.5])
@pytest.mark.parametrize("bi_out", [None, [16], [8, 4]])
@pytest.mark.parametrize("hidden_dim", [(8,), (16, 8)])
def test_green_model_various_configs(
    random_f_init, shrinkage_init, logref, dropout, bi_out, hidden_dim
):
    """Test Green model with various configurations."""
    model = Green(
        n_outputs=2,
        n_chans=21,
        random_f_init=random_f_init,
        shrinkage_init=shrinkage_init,
        logref=logref,
        dropout=dropout,
        bi_out=bi_out,
        hidden_dim=hidden_dim,
    )
    x = torch.randn(2, 21, 1000)
    output = model(x)
    assert output.shape == (2, 2)


@pytest.mark.parametrize(
    "order",
    list(range(1, 5)),
)
@pytest.mark.parametrize(
    "lag",
    list(range(1, 5)),
)
def test_delay_order(order, lag):
    """Test if  input/output dimension combinations"""
    n_chans = 22
    n_outputs = 2

    layer = PhaseSPDNet(order=order, lag=lag, n_outputs=2, n_chans=22)
    x = torch.randn((2, n_chans, 1000))
    output = layer(x)

    # Verify output shape matches expected dimensions
    expected_shape = (2, n_outputs)
    assert output.shape == expected_shape, (
        f"Failed for {order}→{lag} with input {x.shape}"
    )


def test_spdnet_projection_and_diversity():
    """SPDNet exposes its projector, leverage scores and a Grassmann diversity."""

    def make(seed, k=3):
        torch.manual_seed(seed)
        return SPDNet(input_type="cov", n_chans=8, subspacedim=k, n_outputs=2)

    a, b = make(0), make(1)
    lev = a.leverage_scores()
    assert lev.shape == (8,) and not lev.requires_grad
    assert torch.allclose(lev, a.bimap.subspace_projector()[0].diagonal(), atol=1e-6)
    assert torch.allclose(lev.sum(), torch.tensor(3.0), atol=1e-5)
    assert torch.allclose(make(0, k=8).leverage_scores(), torch.ones(8), atol=1e-5)

    d = a.diversity(b)
    assert d.ndim == 0 and not d.requires_grad and d > 0
    assert torch.allclose(d, b.diversity(a), atol=1e-5)
    assert torch.allclose(a.diversity(a), torch.tensor(0.0), atol=1e-5)
    assert a.diversity(b, metric="chordal") <= 3**0.5 + 1e-5
    with pytest.raises(ValueError):
        a.diversity(make(2, k=4))
    with pytest.raises(ValueError):
        a.diversity(b, metric="airm")
    # padded models (subspacedim >= n_chans) over different channel sets
    wide = SPDNet(input_type="cov", n_chans=4, subspacedim=8, n_outputs=2)
    with pytest.raises(ValueError, match="n_chans"):
        wide.diversity(SPDNet(input_type="cov", n_chans=6, subspacedim=8, n_outputs=2))


def test_spdnet_analysis_does_not_freeze_bimap():
    """Analysis calls inside parametrize.cached() must not detach the BiMap."""
    from torch.nn.utils import parametrize

    torch.manual_seed(0)
    a = SPDNet(input_type="cov", n_chans=8, subspacedim=3, n_outputs=2)
    b = SPDNet(input_type="cov", n_chans=8, subspacedim=3, n_outputs=2)
    X = torch.randn(4, 8, 8)
    X = X @ X.mT + torch.eye(8)
    with parametrize.cached():
        a.leverage_scores()
        a.diversity(b)
        a(X).sum().backward()
    assert a.bimap.parametrizations.weight.original.grad is not None
