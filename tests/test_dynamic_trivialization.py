# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Behavior of moving parametrization charts and their training state."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

from torch import nn
from torch.nn.utils import parametrizations, parametrize
from torch.testing import assert_close

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


@pytest.mark.parametrize("kind", ["mean", "meanvar", "AIM", "LEM", "LCM"])
def test_dynamic_batchnorm_rebase(kind):
    if kind == "mean":
        layer = SPDBatchNormMean(3, dtype=torch.float64)
    elif kind == "meanvar":
        layer = SPDBatchNormMeanVar(3, dtype=torch.float64)
    else:
        layer = SPDBatchNormLie(3, metric=kind, dtype=torch.float64)
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
    if kind != "mean":
        assert_close(
            layer.weight if kind == "meanvar" else layer.shift,
            torch.tensor(1.0, dtype=torch.float64),
        )


def test_rebase_selective_optimizer_state():
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
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


def test_dynamic_checkpoint_round_trip():
    model = _model()
    static = deepcopy(model.state_dict())
    model.train()
    step_dynamic_trivialization(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
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
    other_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=0)
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


def _benchmark_module():
    path = (
        Path(__file__).parents[1]
        / "examples/applied_examples/benchmark_dynamic_trivialization.py"
    )
    spec = spec_from_file_location("dynamic_benchmark", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamic_benchmark_smoke():
    result = _benchmark_module().run_training(0, torch.float64, 1, steps=2, n=6, k=3)
    assert set(result["losses"]) == {0, 2}
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result["losses"].values())
    assert result["elapsed_seconds"] > 0
    assert result["orthogonality_error"] < 1e-8
    assert result["min_eigenvalue"] > 0


def test_repeated_float32_rebase():
    result = _benchmark_module().run_training(0, torch.float32, 1)
    assert result["min_eigenvalue"] > 0


def test_dynamic_benchmark_paired_inputs(monkeypatch):
    benchmark = _benchmark_module()
    records, targets = [], []
    forward, mse = BiMap.forward, torch.nn.functional.mse_loss

    def record_forward(layer, x):
        records.append((x.detach().clone(), layer.weight.detach().clone()))
        return forward(layer, x)

    def record_mse(output, target, *args, **kwargs):
        targets.append(target.detach().clone())
        return mse(output, target, *args, **kwargs)

    monkeypatch.setattr(BiMap, "forward", record_forward)
    monkeypatch.setattr(torch.nn.functional, "mse_loss", record_mse)
    benchmark.run_training(0, torch.float64, None, steps=2, n=6, k=3)
    first_records, first_targets = records[:], targets[:]
    records.clear()
    targets.clear()
    benchmark.run_training(0, torch.float64, 1, steps=2, n=6, k=3)
    assert len(records) == len(first_records) and len(targets) == len(first_targets)
    for (x, _), (previous_x, _) in zip(records, first_records):
        assert_close(x, previous_x, atol=0, rtol=0)
    assert_close(records[1][1], first_records[1][1], atol=0, rtol=0)
    assert_close(targets, first_targets, atol=0, rtol=0)
