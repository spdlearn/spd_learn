# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Behavior of moving parametrization charts and their training state."""

from copy import deepcopy

import pytest
import torch

from torch import nn
from torch.nn.utils import parametrizations, parametrize
from torch.testing import assert_close

from spd_learn import models
from spd_learn.models import TensorCSPNet
from spd_learn.modules import (
    BiMap,
    LogEig,
    PositiveDefiniteScalar,
    SPDBatchNormLie,
    SPDBatchNormMean,
    SPDBatchNormMeanVar,
    SymmetricPositiveDefinite,
    step_dynamic_trivialization,
)


OPTIMIZERS = [
    pytest.param(torch.optim.SGD, {"momentum": 0.9}, id="sgd-momentum"),
    pytest.param(torch.optim.AdamW, {"weight_decay": 0}, id="adamw"),
]


def _spd(n, dtype=torch.float64):
    a = torch.randn(4, n, n, generator=torch.Generator().manual_seed(2), dtype=dtype)
    return a @ a.mT / n + torch.eye(n, dtype=dtype)


def _model():
    return nn.Sequential(
        BiMap(6, 3, dtype=torch.float64),
        SPDBatchNormMeanVar(3, dtype=torch.float64),
        LogEig(),
        nn.Flatten(1),
        nn.Linear(6, 2, dtype=torch.float64),
    ).eval()


@pytest.mark.parametrize("enabled", [False, True], ids=["unconstrained", "orthogonal"])
def test_tensorcspnet_parametrization_opt_in(enabled):
    model = TensorCSPNet(n_chans=6, dims=(6, 8, 8, 3), parametrized=enabled)
    weights = [layer for layer in model.modules() if isinstance(layer, BiMap)]
    assert weights and all(
        parametrize.is_parametrized(layer, "weight") == enabled for layer in weights
    )


@pytest.mark.parametrize("name", models.__all__)
def test_all_models_dynamic_trivialization(name):
    options = dict(n_chans=22, n_outputs=4)
    if name == "TensorCSPNet":
        options["parametrized"] = True
    if name == "Green":
        options.update(sfreq=125, bi_out=[16])
    model = getattr(models, name)(**options).eval()
    shape = (2, 9, 22, 256) if name == "TensorCSPNet" else (2, 22, 256)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(1))
    model(x).square().mean().backward()
    original_ids = set(map(id, model.parameters()))
    before = model(x).detach()
    step_dynamic_trivialization(model)
    assert set(map(id, model.parameters())) == original_ids
    after = model(x)
    assert_close(after, before, atol=1e-5, rtol=1e-4)
    after.square().mean().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_repeated_float32_rebase():
    layer = SPDBatchNormMean(8).eval()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.05)
    x = _spd(8, torch.float32)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        layer(x).square().mean().backward()
        optimizer.step()
        step_dynamic_trivialization(layer, optimizer)
    assert torch.isfinite(layer.bias).all()


@pytest.mark.parametrize("shape", [(6, 3), (3, 3), (3, 6)])
@pytest.mark.parametrize("mapping", ["cayley", "matrix_exp", "householder"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dynamic_bimap_rebase(shape, mapping, dtype):
    n, k = shape
    layer = BiMap(n, k, depthwise=2, orthogonal_map=mapping, dtype=dtype)
    x = _spd(n, dtype).unsqueeze(1).expand(-1, 2, -1, -1)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-3)
    layer(x).square().mean().backward()
    optimizer.step()
    original = layer.parametrizations.weight.original
    weight, output = layer.weight.detach().clone(), layer(x).detach()
    step_dynamic_trivialization(layer, optimizer)
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    assert layer.parametrizations.weight.original is original
    assert_close(layer.weight, weight, atol=tol, rtol=tol)
    assert_close(layer(x), output, atol=tol, rtol=tol)
    assert_close(
        layer.weight.mT @ layer.weight,
        torch.eye(k, dtype=dtype).expand(2, -1, -1),
        atol=tol,
        rtol=tol,
    )
    assert torch.linalg.eigvalsh(layer(x)).min() > 0
    assert_close(original.tril(-1), torch.zeros_like(original))
    assert original.grad is None
    layer(x).square().mean().backward()
    assert torch.isfinite(original.grad).all() and original.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "layer_class, kwargs, scale",
    [
        pytest.param(SPDBatchNormMean, {}, None, id="mean"),
        pytest.param(SPDBatchNormMeanVar, {}, "weight", id="meanvar"),
        *[
            pytest.param(
                SPDBatchNormLie, {"metric": metric}, "shift", id=f"lie-{metric}"
            )
            for metric in ("AIM", "LEM", "LCM")
        ],
    ],
)
def test_dynamic_batchnorm_rebase(layer_class, kwargs, scale):
    layer = layer_class(3, dtype=torch.float64, **kwargs)
    layer.eval()
    with torch.no_grad():
        layer.bias = torch.tensor(
            [[[2.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 3.0]]], dtype=torch.float64
        )
    x = _spd(3)
    bias, output = layer.bias.detach(), layer(x).detach()
    before = {
        name: value.clone()
        for name, value in layer.state_dict().items()
        if "bias" not in name
    }
    original = layer.parametrizations.bias.original
    step_dynamic_trivialization(layer)
    assert layer.parametrizations.bias.original is original
    assert layer.parametrizations.bias[0].use_trivialization
    assert_close(layer.bias, bias)
    assert_close(layer(x), output)
    for name, value in before.items():
        assert_close(layer.state_dict()[name], value)
    layer(x).square().mean().backward()
    assert original.grad is not None and torch.isfinite(original.grad).all()
    layer.reset_parameters()
    assert_close(layer.bias, torch.eye(3, dtype=torch.float64).unsqueeze(0))
    if scale is not None:
        assert_close(
            getattr(layer, scale),
            torch.tensor(1.0, dtype=torch.float64),
        )


@pytest.mark.parametrize("optimizer_class, kwargs", OPTIMIZERS)
def test_rebase_selective_optimizer_state(optimizer_class, kwargs):
    model = _model()
    optimizer = optimizer_class(model.parameters(), lr=1e-3, **kwargs)
    model(_spd(6)).square().mean().backward()
    optimizer.step()
    ids = set(map(id, model.parameters()))
    changed = [
        model[0].parametrizations.weight.original,
        model[1].parametrizations.bias.original,
    ]
    unaffected = [model[1].parametrizations.weight.original, *model[-1].parameters()]
    states = [deepcopy(optimizer.state[p]) for p in unaffected]
    step_dynamic_trivialization(model, optimizer)
    assert set(map(id, model.parameters())) == ids
    for p in changed:
        assert p not in optimizer.state and p.grad is None
    for p, before in zip(unaffected, states):
        assert_close(optimizer.state[p], before)
    model(_spd(6)).square().mean().backward()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize(
    "n, dtype, device",
    [
        pytest.param(2, torch.float64, "cpu", id="dtype"),
        pytest.param(3, torch.float32, "cpu", id="shape"),
        pytest.param(
            2,
            torch.float32,
            "cuda",
            id="device",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_rejected_spd_assignment_preserves_chart(n, dtype, device):
    layer = SPDBatchNormMean(2, dtype=torch.float32).eval()
    step_dynamic_trivialization(layer)
    base = layer.parametrizations.bias[0].base_factor.clone()
    original = layer.parametrizations.bias.original.detach().clone()
    x = torch.eye(2).unsqueeze(0)
    before = layer(x).detach()
    value = 2 * torch.eye(n, dtype=dtype, device=device).unsqueeze(0)
    with pytest.raises(ValueError):
        layer.bias = value
    assert_close(layer.parametrizations.bias[0].base_factor, base)
    assert_close(layer.parametrizations.bias.original, original)
    assert_close(layer(x), before)
    layer.double()
    with torch.no_grad():
        layer.bias = 2 * torch.eye(2, dtype=torch.float64).unsqueeze(0)
    assert_close(layer.bias, 2 * torch.eye(2, dtype=torch.float64).unsqueeze(0))


def test_rebase_skips_unrelated_parameters():
    plain = BiMap(3, 2, parametrized=False, dtype=torch.float64)
    holder = nn.Module()
    holder.bias = nn.Parameter(torch.eye(3, dtype=torch.float64))
    holder.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    parametrize.register_parametrization(
        holder, "bias", SymmetricPositiveDefinite("softplus")
    )
    parametrize.register_parametrization(holder, "scale", PositiveDefiniteScalar())
    frozen = SPDBatchNormMeanVar(3, bias_requires_grad=False, dtype=torch.float64)
    model = nn.ModuleList(
        [
            plain,
            holder,
            frozen,
            SPDBatchNormMean(3, rebias=False),
            SPDBatchNormMeanVar(3, affine=False),
        ]
    )
    before = deepcopy(model.state_dict())
    step_dynamic_trivialization(model)
    for key, value in before.items():
        if key != "2.parametrizations.bias.original":
            assert_close(model.state_dict()[key], value)
    assert_close(frozen.bias, torch.eye(3, dtype=torch.float64).unsqueeze(0))
    assert not frozen.parametrizations.bias.original.requires_grad
    assert "2.parametrizations.bias.0.base_factor" in model.state_dict()
    assert not holder.parametrizations.bias[0].use_trivialization


@pytest.mark.parametrize("unsupported", ["chain", "no_base", "spd_chain"])
def test_rebase_rejects_unsupported_chain(unsupported):
    good = SPDBatchNormMean(3, dtype=torch.float64)
    if unsupported == "spd_chain":
        bad = SPDBatchNormMean(3, dtype=torch.float64)
        parametrize.register_parametrization(bad, "bias", nn.Identity())
    else:
        bad = BiMap(3, 2, parametrized=unsupported == "chain", dtype=torch.float64)
        if unsupported == "chain":
            parametrize.register_parametrization(bad, "weight", nn.Identity())
        else:
            parametrizations.orthogonal(
                bad, "weight", orthogonal_map="householder", use_trivialization=False
            )
    model = nn.ModuleList([good, bad])
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        step_dynamic_trivialization(model)
    assert_close(model.state_dict(), before)
    assert not good.parametrizations.bias[0].use_trivialization


@pytest.mark.parametrize("optimizer_class, kwargs", OPTIMIZERS)
def test_dynamic_checkpoint_round_trip(optimizer_class, kwargs):
    model = _model()
    static = deepcopy(model.state_dict())
    model.train()
    step_dynamic_trivialization(model)
    optimizer = optimizer_class(model.parameters(), lr=1e-3, **kwargs)
    x = _spd(6)
    model(x).square().mean().backward()
    optimizer.step()
    step_dynamic_trivialization(model, optimizer)
    model.eval()
    restored = _model()
    with torch.no_grad():
        restored.train()(x)
    restored.eval()
    restored[1].parametrizations.bias.original.requires_grad_(False)
    step_dynamic_trivialization(restored)
    assert model.state_dict().keys() == restored.state_dict().keys()
    restored.load_state_dict(deepcopy(model.state_dict()), strict=True)
    restored[1].parametrizations.bias.original.requires_grad_(True)
    other_optimizer = optimizer_class(restored.parameters(), lr=1e-3, **kwargs)
    other_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    assert_close(model(x), restored(x))
    for net, opt in [(model, optimizer), (restored, other_optimizer)]:
        opt.zero_grad(set_to_none=True)
        net(x).square().mean().backward()
        opt.step()
        with torch.random.fork_rng():
            torch.manual_seed(42)
            step_dynamic_trivialization(net, opt)
    assert_close(model(x), restored(x))
    legacy = _model()
    legacy.load_state_dict(static, strict=True)
    output = legacy(x).detach()
    step_dynamic_trivialization(legacy)
    assert_close(legacy(x), output)
