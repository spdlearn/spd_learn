from math import sqrt

import pytest
import torch

from spd_learn.functional import matrix_exp, matrix_inv_sqrt, matrix_log, vec_to_sym
from spd_learn.functional.batchnorm import karcher_mean_iteration
from spd_learn.modules.batchnorm import SPDBatchNormMean, SPDBatchNormMeanVar


@pytest.fixture()
def simulated_data():
    ndim = 4
    nobs = 128
    generator = torch.Generator().manual_seed(42)

    # define zero-mean tangent space (TS) features
    logz = vec_to_sym(torch.randn((nobs, ndim * (ndim + 1) // 2), generator=generator))
    logz = logz - logz.mean(dim=0, keepdim=True)
    # project them from tangent space at the identity matrix to the SPD
    # manifold
    z = matrix_exp.apply(logz)
    # define a linear mixing model
    eps = 0.1
    forward_model = (torch.rand((ndim, ndim), generator=generator) - 0.5) * (
        1 - eps
    ) + eps * torch.eye(ndim)
    # apply linear mixing model
    x = forward_model @ z @ forward_model.mT
    # due to invariance of the Fréchet mean we have an analytic solution:
    x_mean_expected = (forward_model @ forward_model.mT).unsqueeze(0)

    x_mean_invsq = matrix_inv_sqrt.apply(x_mean_expected)
    x_variance_expected = (
        matrix_log.apply(x_mean_invsq @ x @ x_mean_invsq)
        .square()
        .sum(dim=(-1, -2), keepdim=True)
        .squeeze(-1)
    ).mean(dim=0, keepdim=True)

    return x, x_mean_expected, x_variance_expected


def test_batchnormbrooks_running_mean(simulated_data):
    x, x_mean_expected, _ = simulated_data

    nobs = x.shape[0]
    ndim = x.shape[-1]

    # instantiate the batch norm layer
    layer = SPDBatchNormMean(
        num_features=ndim,
        momentum=1.0,
        rebias=True,
        n_iter=64,  # ensure that we have enough iterations to reach convergence
    )
    # estimate the (running) mean
    layer.train()
    layer(x)
    x_mean_estimated = layer.running_mean
    # check shapes
    assert x_mean_estimated.shape == x_mean_expected.shape
    # assert if we are close enough to the analytic solution
    atol = sqrt(1 / nobs)
    assert torch.allclose(x_mean_estimated, x_mean_expected, atol=atol, rtol=0.0)


def test_spdbatchnrom_running_stats_single_batch(simulated_data):
    x, x_mean_expected, x_variance_expected = simulated_data

    nobs = x.shape[0]
    ndim = x.shape[-1]

    # instantiate the batch norm layer
    layer = SPDBatchNormMeanVar(
        num_features=ndim,
        momentum=1.0,
        affine=False,
        n_iter=64,  # ensure that we have enough iterations to reach convergence
    )
    # estimate the (running) statistics
    layer.train()
    layer(x)
    x_mean_estimated = layer.running_mean
    x_variance_estimated = layer.running_var

    # check shapes
    assert x_mean_estimated.shape == x_mean_expected.shape
    assert x_variance_estimated.shape == x_variance_expected.shape

    # ensure that the running statistics do not require gradients
    assert x_mean_estimated.requires_grad is False
    assert x_variance_estimated.requires_grad is False

    # assert if we are close enough to the analytic solution
    atol = sqrt(1 / nobs)
    assert torch.allclose(x_mean_estimated, x_mean_expected, atol=atol, rtol=0.0)
    assert torch.allclose(
        x_variance_estimated, x_variance_expected, atol=atol, rtol=0.0
    )


def test_spdbatchnrom_running_stats_multi_batch(simulated_data):
    x, x_mean_expected, x_variance_expected = simulated_data

    nobs = x.shape[0]
    ndim = x.shape[-1]

    # instantiate the batch norm layer
    layer = SPDBatchNormMeanVar(num_features=ndim, affine=True, n_iter=1)

    # setup a dataset and dataloader
    ds = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(ds, batch_size=nobs // 4, drop_last=True)

    # estimate the (running) statistics over batches
    layer.train()
    n_epochs = 64 // len(loader) * 4  # more epochs for better convergence
    for epoch in range(n_epochs):
        layer.momentum = 1 / (epoch + 1)  # decay the momentum over batches
        for batch in loader:
            x = batch[0]
            layer(x)
    x_mean_estimated = layer.running_mean
    x_variance_estimated = layer.running_var

    # check shapes
    assert x_mean_estimated.shape == x_mean_expected.shape
    assert x_variance_estimated.shape == x_variance_expected.shape

    # assert if we are close enough to the analytic solution
    # variance estimation over batches is noisier, so use looser tolerance
    atol_mean = sqrt(1 / nobs)
    atol_var = 3 * sqrt(1 / nobs)
    assert torch.allclose(x_mean_estimated, x_mean_expected, atol=atol_mean, rtol=0.0)
    assert torch.allclose(
        x_variance_estimated, x_variance_expected, atol=atol_var, rtol=0.0
    )


def test_spdbatchnrom_requires_grad(simulated_data):
    x, _, _ = simulated_data

    # instantiate the batch norm layer
    layer = SPDBatchNormMeanVar(
        num_features=x.shape[-1],
    )

    # ensure that the running statistics do not require gradients
    assert layer.running_mean.requires_grad is False
    assert layer.running_var.requires_grad is False

    # estimate the (running) statistics
    layer.train()
    x.requires_grad_ = True
    output = layer(x)

    # ensure that the output requires gradients
    assert output.requires_grad is True

    # ensure that the running statistics do not require gradients
    assert layer.running_mean.requires_grad is False
    assert layer.running_var.requires_grad is False


def test_karcher_mean_iteration_gradient_flow():
    """Verify gradients flow through current_mean when detach=False."""
    ndim = 4
    nobs = 8

    # Test with detach=False: gradients should flow through current_mean
    generator = torch.Generator().manual_seed(42)
    logz = vec_to_sym(torch.randn((nobs, ndim * (ndim + 1) // 2), generator=generator))
    logz.requires_grad_(True)
    X = matrix_exp.apply(logz)

    log_mean = vec_to_sym(torch.randn((1, ndim * (ndim + 1) // 2), generator=generator))
    log_mean.requires_grad_(True)
    current_mean = matrix_exp.apply(log_mean)

    new_mean = karcher_mean_iteration(X, current_mean, detach=False)
    loss = new_mean.sum()
    loss.backward()
    assert log_mean.grad is not None
    assert log_mean.grad.abs().sum() > 0

    # Test with detach=True: gradients should NOT flow through current_mean
    # Create fresh tensors to avoid backward through freed graph
    generator = torch.Generator().manual_seed(42)
    logz = vec_to_sym(torch.randn((nobs, ndim * (ndim + 1) // 2), generator=generator))
    logz.requires_grad_(True)
    X = matrix_exp.apply(logz)

    log_mean = vec_to_sym(torch.randn((1, ndim * (ndim + 1) // 2), generator=generator))
    log_mean.requires_grad_(True)
    current_mean = matrix_exp.apply(log_mean)

    new_mean = karcher_mean_iteration(X, current_mean, detach=True)
    loss = new_mean.sum()
    loss.backward()
    # Gradient is None or zero because current_mean is detached
    assert log_mean.grad is None or log_mean.grad.abs().sum() == 0
    # But gradients should still flow through X
    assert logz.grad is not None


def test_karcher_mean_iteration_backward_compatibility():
    """Verify default behavior matches explicit detach=True."""
    generator = torch.Generator().manual_seed(42)
    ndim = 4
    nobs = 8

    # Create SPD matrices
    logz = vec_to_sym(torch.randn((nobs, ndim * (ndim + 1) // 2), generator=generator))
    X = matrix_exp.apply(logz)

    # Create initial mean
    log_mean = vec_to_sym(torch.randn((1, ndim * (ndim + 1) // 2), generator=generator))
    current_mean = matrix_exp.apply(log_mean)

    # Default behavior
    new_mean_default = karcher_mean_iteration(X, current_mean)

    # Explicit detach=True
    new_mean_explicit = karcher_mean_iteration(X, current_mean, detach=True)

    # Results should be identical
    assert torch.allclose(new_mean_default, new_mean_explicit)
