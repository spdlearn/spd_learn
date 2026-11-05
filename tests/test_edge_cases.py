# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for argument checks and code paths not reached by the other tests."""

import pytest
import torch

from torch.testing import assert_close

from spd_learn.functional import (
    exp_map_lem,
    log_euclidean_geodesic,
    log_map_lem,
    matrix_exp,
    matrix_log,
    spd_rpgd_attack,
)
from spd_learn.functional.metrics.affine_invariant import airm_distance
from spd_learn.functional.riemannian_pgd import _project_to_spd_ball
from spd_learn.models import SPDNet
from spd_learn.modules import PositiveDefiniteScalar, SymmetricPositiveDefinite


def _random_spd(*batch, n=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(*batch, n, n, dtype=torch.float64, generator=generator)
    return A @ A.mT + n * torch.eye(n, dtype=torch.float64)


# Manifold parametrizations
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 3), (2, 2, 3, 3)])
def test_dynamic_spd_rebase(dtype, shape):
    generator = torch.Generator().manual_seed(3)
    A = torch.randn(shape, generator=generator, dtype=dtype)
    P = A @ A.mT + torch.eye(3, dtype=dtype)
    S = torch.randn(shape, generator=generator, dtype=dtype)
    S = (S + S.mT) / 10
    param = SymmetricPositiveDefinite(use_trivialization=True)
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    assert_close(param.right_inverse(P), torch.zeros_like(P))
    assert_close(param.base_factor @ param.base_factor.mT, P, atol=tol, rtol=tol)
    expected = param.base_factor @ torch.linalg.matrix_exp(S) @ param.base_factor.mT
    assert_close(param(S), expected, atol=tol, rtol=tol)
    current = param(S).detach()
    assert_close(param(param.right_inverse(current)), current, atol=tol, rtol=tol)
    S.requires_grad_()
    param(S).square().sum().backward()
    assert torch.isfinite(S.grad).all()
    assert param.base_factor.grad_fn is None
    param.to(dtype=torch.float64)
    assert param.base_factor.dtype == torch.float64


def test_dynamic_spd_preserves_small_eigenvalues():
    P = torch.diag(torch.tensor([1e-9, 1.0]))
    param = SymmetricPositiveDefinite(use_trivialization=True)
    assert_close(param(param.right_inverse(P)), P, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        torch.ones(2, 3),
        torch.full((2, 2), float("nan")),
        torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
        torch.zeros(2, 2),
        torch.diag(torch.tensor([-1.0, 1.0])),
    ],
)
def test_dynamic_spd_rejects_invalid_assignment(bad):
    param = SymmetricPositiveDefinite(use_trivialization=True)
    param.right_inverse(torch.eye(2))
    before = param.base_factor.clone()
    with pytest.raises((ValueError, torch.linalg.LinAlgError)):
        param.right_inverse(bad)
    assert_close(param.base_factor, before)


def test_dynamic_spd_rejects_softplus():
    with pytest.raises(ValueError, match="exp"):
        SymmetricPositiveDefinite(mapping="softplus", use_trivialization=True)


def test_static_spd_checkpoint_stays_stateless():
    assert not SymmetricPositiveDefinite().state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_dynamic_spd_buffer_moves_to_cuda():
    param = SymmetricPositiveDefinite(use_trivialization=True)
    param.right_inverse(torch.eye(2))
    param.cuda()
    assert param(torch.zeros(2, 2, device="cuda")).is_cuda


@pytest.mark.parametrize("cls", [SymmetricPositiveDefinite, PositiveDefiniteScalar])
def test_parametrization_rejects_unknown_mapping(cls):
    with pytest.raises(ValueError, match="mapping must be"):
        cls(mapping="relu")


@pytest.mark.parametrize("mapping", ["exp", "softplus"])
def test_spd_parametrization_round_trip(mapping):
    param = SymmetricPositiveDefinite(mapping=mapping)
    P = _random_spd(2)
    S = param.right_inverse(P)
    assert_close(S, S.mT)
    assert_close(param(S), P)
    assert torch.linalg.eigvalsh(param(-P)).min() > 0


@pytest.mark.parametrize("mapping", ["exp", "softplus"])
def test_positive_scalar_parametrization_round_trip(mapping):
    param = PositiveDefiniteScalar(mapping=mapping)
    s = torch.tensor(2.5, dtype=torch.float64)
    assert_close(param(param.right_inverse(s)), s)
    assert param(torch.tensor(-10.0, dtype=torch.float64)) > 0


# Riemannian PGD attack
class _WrappedModel:
    """Mimics a skorch/Braindecode estimator, which exposes ``module_``."""

    def __init__(self, module):
        self.module_ = module


def test_rpgd_attack_accepts_wrapped_model_and_stays_in_ball():
    torch.manual_seed(0)
    model = SPDNet(input_type="cov", n_chans=3, n_outputs=2)
    x = (torch.eye(3) + 0.1 * torch.ones(3, 3)).unsqueeze(0)
    y = torch.tensor([0])
    eps = 0.05
    x_adv = spd_rpgd_attack(
        _WrappedModel(model),
        x,
        y,
        eps=eps,
        criterion=torch.nn.CrossEntropyLoss(),
        n_iterations=5,
        step_size=1.0,
    )
    assert x_adv.shape == x.shape
    assert airm_distance(x, x_adv).max() <= eps + 1e-4


def test_rpgd_attack_rejects_non_module():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        spd_rpgd_attack(
            object(),
            torch.eye(3)[None],
            torch.tensor([0]),
            eps=0.1,
            criterion=torch.nn.CrossEntropyLoss(),
        )


def test_project_to_spd_ball_single_matrix():
    x0 = _random_spd(seed=1)
    x = _random_spd(seed=2)
    dist = airm_distance(x0, x)
    assert dist.ndim == 0

    inside = _project_to_spd_ball(x0, x, eps=float(dist) + 1.0)
    assert_close(inside, x)

    eps = float(dist) / 2
    projected = _project_to_spd_ball(x0, x, eps=eps)
    assert_close(airm_distance(x0, projected), torch.tensor(eps, dtype=x.dtype))


def test_project_to_spd_ball_batch_projects_only_outside_points():
    x0 = _random_spd(2, seed=3)
    x = _random_spd(2, seed=4)
    dist = airm_distance(x0, x)
    eps = float(dist.min() + dist.max()) / 2  # one point inside, one outside
    projected = _project_to_spd_ball(x0, x, eps=eps)
    inside = dist <= eps
    assert_close(projected[inside], x[inside])
    assert_close(
        airm_distance(x0[~inside], projected[~inside]),
        torch.full((int((~inside).sum()),), eps, dtype=x.dtype),
    )


# Log-Euclidean metric
def test_log_euclidean_exp_and_log_maps_are_inverse():
    P = _random_spd(2, seed=5)
    Q = _random_spd(2, seed=6)
    V = log_map_lem(P, Q)
    assert_close(V, matrix_log.apply(Q) - matrix_log.apply(P))
    assert_close(exp_map_lem(P, V), Q)


def test_log_euclidean_geodesic_scalar_tensor_time_on_batch():
    A = _random_spd(4, seed=7)
    B = _random_spd(4, seed=8)
    t = torch.tensor(0.25, dtype=torch.float64)
    G = log_euclidean_geodesic(A, B, t)
    expected = matrix_exp.apply(0.75 * matrix_log.apply(A) + 0.25 * matrix_log.apply(B))
    assert_close(G, expected)
