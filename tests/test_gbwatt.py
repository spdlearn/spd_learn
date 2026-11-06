# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import copy
import math

import pytest
import torch

from spd_learn.functional import (
    bw_attention_mean,
    gbw_chart,
    gbw_distance,
    gbw_inverse_chart,
    gbw_mean,
)
from spd_learn.models import GBWAtt
from spd_learn.modules import GBWAttention, GBWGeometry, GBWPatchCovariance


def spd(*shape, dtype=torch.float64, device="cpu"):
    a = torch.randn(*shape, dtype=dtype, device=device)
    return a @ a.mT + 0.7 * torch.eye(shape[-1], dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chart_metric_identity_and_power(dtype):
    torch.manual_seed(12)
    x, y = spd(2, 3, 3, dtype=dtype), spd(2, 3, 3, dtype=dtype)
    metric = spd(3, 3, dtype=dtype)
    for power in [1.0, 1.5]:
        recovered = gbw_inverse_chart(gbw_chart(x, metric, power), metric, power)
        torch.testing.assert_close(recovered, x, rtol=4e-5, atol=2e-5)
        d = gbw_distance(x, y, metric, power)
        torch.testing.assert_close(
            d, gbw_distance(y, x, metric, power), rtol=4e-5, atol=2e-5
        )
        assert (d >= 0).all()
    # Scalar generalized metric gives a known factor 1/sqrt(m).
    eye = torch.eye(3, dtype=dtype)
    d = gbw_distance(4 * eye, 9 * eye, 4 * eye)
    torch.testing.assert_close(d, torch.tensor(3**0.5 / 2, dtype=dtype))
    # Independently known diagonal formula, including 1/power scaling.
    a, b = (
        torch.diag(torch.tensor([1.0, 4.0], dtype=dtype)),
        torch.diag(torch.tensor([9.0, 16.0], dtype=dtype)),
    )
    power = 1.5
    expected = (
        (a.diag() ** (power / 2) - b.diag() ** (power / 2)).square().sum()
    ).sqrt() / power
    torch.testing.assert_close(
        gbw_distance(a, b, torch.eye(2, dtype=dtype), power), expected
    )


def test_gradcheck_metric_power_repeated_spectra():
    eye = torch.eye(2, dtype=torch.float64)
    x = (2 * eye).requires_grad_()
    y = torch.tensor([[3.0, 0.2], [0.2, 4.0]], dtype=torch.float64)
    metric = eye.clone().requires_grad_()
    power = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

    # Symmetric perturbation is the scientific interface of spectral functions.
    def sym(a):
        return (a + a.mT) / 2

    assert torch.autograd.gradcheck(
        lambda a, m, p: gbw_distance(sym(a), y, sym(m), p, squared=True),
        (x, metric, power),
        atol=2e-5,
        rtol=2e-4,
    )
    weights = torch.tensor([[0.3, 0.7]], dtype=torch.float64)
    values = torch.stack([2 * eye, y]).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda v, m, p: gbw_mean(sym(v), weights, sym(m), p, n_iter=3),
        (values, metric, power),
        atol=3e-5,
        rtol=3e-4,
    )
    zero = gbw_distance(x, x, metric, power)
    zero.backward()
    assert zero == 0
    for p in [x, metric, power]:
        assert p.grad is not None and p.grad.isfinite().all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("trace", [1.0, 1e4, 1e6])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_coincidence_has_exact_zero_distance_and_gradients(dtype, trace, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    for seed in (0, 3, 13):
        a = torch.randn(
            6, 6, dtype=dtype, generator=torch.Generator().manual_seed(seed)
        )
        a = a.to(device)
        covariance = a @ a.mT + 0.1 * torch.eye(6, dtype=dtype, device=device)
        covariance = covariance * (trace / covariance.trace())
        for exponent in (1.0, 1.5):
            for squared in (False, True):
                x = covariance.clone().requires_grad_()
                y = covariance.clone().requires_grad_()
                metric = torch.eye(6, dtype=dtype, device=device).requires_grad_()
                power = torch.tensor(
                    exponent, dtype=dtype, device=device, requires_grad=True
                )
                distance = gbw_distance(x, y, metric, power, squared=squared)
                assert distance == 0
                distance.backward()
                for parameter in (x, y, metric, power):
                    assert parameter.grad is not None
                    assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nearby_large_commuting_distance_is_not_erased(dtype):
    x = 1e4 * torch.eye(6, dtype=dtype)
    y = (1e4 + 1) * torch.eye(6, dtype=dtype)
    expected = math.sqrt(6) * (math.sqrt(1e4 + 1) - math.sqrt(1e4))
    distance = gbw_distance(x, y, torch.eye(6, dtype=dtype))
    assert distance > 0
    tolerance = 2e-3 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(
        distance.double(),
        torch.tensor(expected, dtype=torch.float64),
        rtol=tolerance,
        atol=1e-12,
    )
    torch.testing.assert_close(
        gbw_distance(x, y, torch.eye(6, dtype=dtype), squared=True), distance.square()
    )


@pytest.mark.parametrize(
    "dtype,power_dtype",
    [(torch.float32, torch.float64), (torch.float64, torch.float32)],
)
@pytest.mark.parametrize("squared", [False, True])
def test_distance_mixed_power_dtype_is_independent_of_batch_shape(
    dtype, power_dtype, squared
):
    x, y = spd(3, 3, dtype=dtype), spd(3, 3, dtype=dtype)
    metric = torch.eye(3, dtype=dtype)
    power = torch.tensor(1.5, dtype=power_dtype, requires_grad=True)
    scalar = gbw_distance(x, y, metric, power, squared=squared)
    batch = gbw_distance(x[None], y[None], metric, power, squared=squared)
    assert scalar.dtype == batch.dtype == dtype
    torch.testing.assert_close(scalar, batch[0])
    (scalar + batch.sum()).backward()
    assert power.grad is not None and torch.isfinite(power.grad)
    assert power.grad.dtype == power_dtype


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_float32_fixed_point_trace_normalized_ill_conditioned_values(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(18, 18, generator=generator))
    spectrum = torch.logspace(-3, 0, 18)
    spectrum /= spectrum.sum()
    p = (q * spectrum) @ q.mT
    p = ((p + p.mT) / 2).to(device)
    values = gbw_chart(
        p[None, None].repeat(1, 3, 1, 1), torch.eye(18, device=device), 1.5
    ).requires_grad_()
    weights = (torch.ones(1, 1, 3, device=device) / 3).requires_grad_()
    result = bw_attention_mean(values, weights)
    assert result.dtype == torch.float32 and result.device == p.device
    assert torch.linalg.eigvalsh(result).min() > 0
    torch.linalg.cholesky(result)
    # Identical keys have that same matrix as their exact BW barycenter.
    torch.testing.assert_close(result[:, 0], values[:, 0], rtol=3e-4, atol=3e-7)
    reference = bw_attention_mean(values.double(), weights.double()).float()
    torch.testing.assert_close(result, reference)
    result.square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


@pytest.mark.parametrize("squared", [False, True])
def test_aligned_distance_broadcast_input_metric_and_power_gradients(squared):
    torch.manual_seed(94)
    x = spd(2, 1, 2, 2).requires_grad_()
    y = spd(1, 3, 2, 2).requires_grad_()
    metric = spd(2, 2).requires_grad_()
    power = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)

    def sym(a):
        return (a + a.mT) / 2

    assert torch.autograd.gradcheck(
        lambda a, b, m, p: gbw_distance(sym(a), sym(b), sym(m), p, squared=squared),
        (x, y, metric, power),
        atol=3e-5,
        rtol=3e-4,
    )


def wide_scale_values(condition, maximum):
    generator = torch.Generator().manual_seed(0)
    values = []
    for scale in torch.logspace(0, math.log10(maximum), 4).tolist():
        q, _ = torch.linalg.qr(torch.randn(18, 18, generator=generator))
        eigenvalues = torch.logspace(
            math.log10(scale / condition), math.log10(scale), 18
        )
        values.append((q * eigenvalues) @ q.mT)
    values = torch.stack(values)[None]
    values = (values + values.mT) / 2
    weights = torch.rand(1, 4, generator=generator)
    return values, weights / weights.sum(-1, keepdim=True)


@pytest.mark.parametrize("condition,maximum", [(1e4, 1e5), (1e6, 1e6)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_wide_scale_outputs_are_spd_or_raise_actionable_error(
    condition, maximum, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    values, weights = wide_scale_values(condition, maximum)
    values, weights = values.to(device), weights.to(device)
    # These are genuine SPD inputs; this checks representability, not bad input.
    assert torch.linalg.cholesky_ex(values).info.eq(0).all()
    metric = torch.eye(18, device=device)
    functions = [lambda: gbw_chart(values, metric, 1.5)]
    for method in ("fixed_point", "root"):
        functions.extend(
            [
                lambda method=method: bw_attention_mean(values, weights, method=method),
                lambda method=method: gbw_mean(
                    values, weights, metric, 1.5, method=method
                ),
            ]
        )
    for function in functions:
        try:
            output = function()
        except ValueError as error:
            assert "float64" in str(error) and "regularization" in str(error)
        else:
            assert torch.equal(output, output.mT)
            assert torch.isfinite(output).all()
            assert torch.linalg.eigvalsh(output).min() > 0
            assert torch.linalg.cholesky_ex(output).info.eq(0).all()


def test_wide_scale_float64_path_remains_usable():
    values, weights = wide_scale_values(1e4, 1e5)
    values, weights = values.double(), weights.double()
    output = gbw_mean(values, weights, torch.eye(18, dtype=torch.float64), 1.5)
    assert torch.isfinite(output).all() and torch.linalg.eigvalsh(output).min() > 0
    assert torch.linalg.cholesky_ex(output).info.eq(0).all()


@pytest.mark.parametrize("method", ["fixed_point", "root"])
def test_commuting_barycenter_reference(method):
    values = torch.diag_embed(
        torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float64)
    )
    weights = torch.tensor([[0.25, 0.75], [1.0, 0.0]], dtype=torch.float64)
    expected = torch.diag_embed(
        (weights @ values.diagonal(dim1=-2, dim2=-1).sqrt()).square()
    )
    torch.testing.assert_close(
        bw_attention_mean(values, weights, n_iter=4, method=method), expected
    )


def test_noncommuting_barycenter_stationarity():
    torch.manual_seed(5)
    values = spd(3, 2, 2)
    weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    mean = bw_attention_mean(values, weights, n_iter=30).detach().requires_grad_()
    loss = (
        gbw_distance(
            mean.unsqueeze(-3), values, torch.eye(2, dtype=torch.float64), squared=True
        )
        * weights
    ).sum()
    (grad,) = torch.autograd.grad(loss, mean)
    assert grad.norm() < 1e-7
    # The author approximation is demonstrably distinct on noncommuting values.
    assert not torch.allclose(
        mean, bw_attention_mean(values, weights, method="root"), atol=1e-5
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_covariance_rank_deficiency_remainders_identity(dtype):
    x = torch.arange(2 * 4 * 11, dtype=dtype).reshape(2, 4, 11)
    layer = GBWPatchCovariance(3)
    out = layer(x)
    assert out.shape == (2, 3, 4, 4)
    assert torch.linalg.eigvalsh(out).min() > 0
    torch.testing.assert_close(out[0], layer(x[:1])[0])
    constant = layer(torch.zeros_like(x))
    torch.testing.assert_close(
        constant, (1e-5 * torch.eye(4, dtype=dtype)).expand_as(constant)
    )
    # Uneven temporal sections are 4,4,3 and retain the last observation.
    changed = x.clone()
    changed[:, 0, -1] += 17
    assert not torch.allclose(out[:, -1], layer(changed)[:, -1])
    with pytest.raises(ValueError, match=">=2"):
        layer(x[..., :5])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_attention_batch_gradients_state(dtype):
    torch.manual_seed(11)
    module = GBWAttention(4, 3, learn_power=True, n_iter=4).to(dtype=dtype)
    x = spd(2, 3, 4, 4, dtype=dtype).requires_grad_()
    out, weights = module(x, return_weights=True)
    assert out.shape == (2, 3, 3, 3)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, dtype=dtype))
    assert torch.linalg.eigvalsh(out).min() > 0
    torch.testing.assert_close(out[:1], module(x[:1]), rtol=2e-4, atol=1e-4)
    changed = x.detach().clone()
    changed[1] *= 7
    torch.testing.assert_close(out[:1], module(changed)[:1], rtol=2e-4, atol=1e-4)
    # Token permutation equivariance.
    perm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(module(x[:, perm]), out[:, perm], rtol=2e-4, atol=1e-4)
    loss = (out * torch.randn_like(out)).sum()
    loss.backward()
    assert x.grad is not None and x.grad.isfinite().all()
    before = {n: p.detach().clone() for n, p in module.named_parameters()}
    for p in module.parameters():
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.norm() > 0
    torch.optim.Adam(module.parameters(), lr=0.0025).step()
    for n, p in module.named_parameters():
        assert not torch.equal(before[n], p)
    assert torch.linalg.eigvalsh(module.geometry.metric).min() > 0
    assert module.geometry.power > 0
    clone = GBWAttention(4, 3, learn_power=True, n_iter=4).to(dtype=dtype)
    clone.load_state_dict(copy.deepcopy(module.state_dict()))
    torch.testing.assert_close(clone(x), module(x))
    module.eval()
    torch.testing.assert_close(module(x), clone(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_repeated_and_ill_conditioned(dtype):
    module = GBWAttention(3, 2, learn_power=True, n_iter=3).to(dtype=dtype)
    x = (
        torch.diag_embed(
            torch.tensor([[1.0, 1.0, 1.0], [1e-5, 1e-3, 1.0]], dtype=dtype)
        )
        .unsqueeze(0)
        .requires_grad_()
    )
    out = module(x)
    out.sum().backward()
    assert out.isfinite().all() and torch.linalg.eigvalsh(out).min() > 0
    assert x.grad.isfinite().all()
    for p in module.parameters():
        assert p.grad is not None and p.grad.isfinite().all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_model_optimizer_and_eval_state(dtype, tmp_path):
    torch.manual_seed(19)
    kwargs = dict(
        n_chans=3,
        n_outputs=2,
        temporal_out_channels=4,
        attention_out_features=3,
        learn_power=True,
        n_iter=4,
    )
    model = GBWAtt(**kwargs).to(dtype=dtype)
    x = torch.randn(3, 3, 80, dtype=dtype)
    logits = model(x)
    assert logits.shape == (3, 2)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0]))
    loss.backward()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for p in model.parameters():
        assert p.grad is not None and p.grad.isfinite().all()
    torch.optim.Adam(model.parameters(), lr=0.0025).step()
    for n, p in model.named_parameters():
        if any(t in n for t in ["geometry", "_trans.parametrizations"]):
            assert not torch.equal(before[n], p), n
    model.eval()
    state = copy.deepcopy(model.state_dict())
    torch.testing.assert_close(model(x[:1]), model(x)[:1], rtol=1e-4, atol=1e-4)
    for n, value in model.state_dict().items():
        torch.testing.assert_close(value, state[n])
    torch.save(model.state_dict(), tmp_path / "gbw.pt")
    clone = GBWAtt(**kwargs).to(dtype=dtype).eval()
    clone.load_state_dict(torch.load(tmp_path / "gbw.pt", weights_only=True))
    torch.testing.assert_close(model(x), clone(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_parity_backward(dtype):
    torch.manual_seed(1)
    cpu = GBWAttention(3, 2, learn_power=True, n_iter=3).to(dtype=dtype)
    gpu = copy.deepcopy(cpu).cuda()
    x = spd(2, 3, 3, 3, dtype=dtype)
    y = cpu(x)
    z = gpu(x.cuda())
    torch.testing.assert_close(y, z.cpu(), rtol=2e-3, atol=3e-4)
    y.square().sum().backward()
    z.square().sum().backward()
    for p, q in zip(cpu.parameters(), gpu.parameters()):
        assert q.grad.isfinite().all()
        torch.testing.assert_close(p.grad, q.grad.cpu(), rtol=8e-3, atol=3e-3)


def test_invalid_interfaces():
    for power in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            GBWGeometry(2, power=power)
    with pytest.raises(ValueError):
        GBWAttention(2, 3)
    with pytest.raises(ValueError):
        GBWAttention(3, 2)(torch.empty(1, 0, 3, 3))
    values = torch.eye(2).expand(2, 2, 2)
    for weights in [
        torch.zeros(1, 2),
        torch.tensor([[-1.0, 2.0]]),
        torch.tensor([[float("nan"), 1.0]]),
    ]:
        with pytest.raises(ValueError):
            bw_attention_mean(values, weights)


def test_covariance_rank_deficient_backward():
    x = torch.randn(2, 6, 8, dtype=torch.float64, requires_grad=True)
    cov = GBWPatchCovariance(3)(x)
    cov.square().sum().backward()
    assert x.grad.isfinite().all() and x.grad.norm() > 0
    constant = torch.zeros(1, 6, 8, dtype=torch.float64, requires_grad=True)
    GBWPatchCovariance(3)(constant).sum().backward()
    assert constant.grad.isfinite().all()


def test_generalized_distance_direct_formula():
    # Independent equation 11, using native eigh only in the test reference.
    torch.manual_seed(21)
    x, y, metric = spd(3, 3), spd(3, 3), spd(3, 3)

    def sqrt(a):
        s, u = torch.linalg.eigh(a)
        return (u * s.sqrt()) @ u.mT

    inv = torch.linalg.inv(metric)
    root = sqrt(x)
    expected = torch.trace(inv @ x) + torch.trace(inv @ y)
    expected -= 2 * torch.trace(sqrt(root @ inv @ y @ inv @ root))
    torch.testing.assert_close(gbw_distance(x, y, metric, squared=True), expected)


def test_default_mi_architecture():
    import spd_learn

    from spd_learn.models import __all__ as model_registry

    assert spd_learn.GBWAtt is GBWAtt
    assert set(model_registry) <= set(spd_learn.__all__)
    model = GBWAtt().eval()
    with torch.no_grad():
        assert model(torch.randn(1, 22, 438)).shape == (1, 4)
    assert model.linear.in_features == 513
    assert not model.manifold_attention.geometry.parametrizations.power.original.requires_grad


@pytest.mark.parametrize("power", [1.0, 1.5])
@pytest.mark.parametrize("method", ["fixed_point", "root"])
def test_attention_independent_diagonal_reference(power, method):
    # In commuting coordinates every stage has a scalar closed form. This
    # checks distance scaling, query/key orientation and inverse deformation
    # without calling the functional geometry under test for the reference.
    module = GBWAttention(2, 2, power=power, aggregation=method).double()
    with torch.no_grad():
        for projection in [module.q_trans, module.k_trans, module.v_trans]:
            projection.weight = torch.eye(2, dtype=torch.float64).unsqueeze(0)
        module.geometry.metric = torch.diag(
            torch.tensor([2.0, 3.0], dtype=torch.float64)
        )
    diagonals = torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float64)
    metric_diagonal = torch.tensor([2.0, 3.0], dtype=torch.float64)
    roots = diagonals.pow(power / 2) / metric_diagonal.sqrt()
    distances = (roots[:, None] - roots[None, :]).norm(dim=-1) / power
    expected_weights = (1 / (1 + distances.log1p())).softmax(dim=-1)
    expected_diagonals = ((expected_weights @ roots).square() * metric_diagonal).pow(
        1 / power
    )
    out, weights = module(torch.diag_embed(diagonals)[None], return_weights=True)
    torch.testing.assert_close(weights[0], expected_weights, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(out[0], torch.diag_embed(expected_diagonals))


def test_geometry_broadcast_frozen_state_and_single_key():
    torch.manual_seed(31)
    geometry = GBWGeometry(2, power=1.5, learn_metric=False).double()
    assert all(not p.requires_grad for p in geometry.parameters())
    torch.testing.assert_close(geometry.metric, torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(geometry.power, torch.tensor(1.5, dtype=torch.float64))
    # Arbitrary leading dimensions broadcast, including independently chosen
    # per-sample metrics expanded over the key/query axes by the caller.
    values = spd(2, 3, 1, 2, 2)
    metric = spd(2, 1, 1, 2, 2)
    weights = torch.ones(2, 3, 2, 1, dtype=torch.float64)
    out = gbw_mean(values, weights, metric, 1.5, n_iter=3)
    torch.testing.assert_close(out, values.expand(2, 3, 2, 2, 2))
    for i in range(2):
        torch.testing.assert_close(
            out[i], gbw_mean(values[i], weights[i], metric[i], 1.5, n_iter=3)
        )
