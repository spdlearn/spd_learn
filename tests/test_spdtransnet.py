# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Equation-based tests; no restricted upstream implementation is imported."""

import copy
import io
import math

import pytest
import torch
import torch.nn.functional as F

from torch.autograd import gradcheck

from spd_learn.functional import log_tokens_to_spd, spd_to_log_tokens
from spd_learn.functional.numerical import get_epsilon
from spd_learn.models import SPDTransNet, __all__, __sequence_models__
from spd_learn.modules import (
    LearnedSinusoidalPosition,
    SPDLogTokenizer,
    SPDTransformerEncoderLayer,
    StructurePreservingMultiheadAttention,
)


def spd(shape, dtype=torch.float64, device="cpu"):
    a = torch.randn(*shape, dtype=dtype, device=device)
    return a @ a.mT / shape[-1] + 0.5 * torch.eye(shape[-1], dtype=dtype, device=device)


def small_model(dtype=torch.float64, device="cpu", dropout=0.0):
    return SPDTransNet(
        matrix_size=3,
        n_bands=2,
        n_subwindows=4,
        context_epochs=3,
        token_matrix_size=3,
        feature_tokens=2,
        intra_heads=2,
        inter_heads=3,
        intra_layers=2,
        inter_layers=2,
        feedforward_dim=10,
        classifier_dim=10,
        n_outputs=5,
        dropout=dropout,
        matrix_scale=1.0,
        eigenvalue_floor=None,
        dtype=dtype,
        device=device,
    )


def reference_attention(module, query, key, value):
    """Eq. 2, independently evaluated one batch/head at a time."""
    q = F.linear(query, module.query.weight, module.query.bias)
    k = F.linear(key, module.key.weight, module.key.bias)
    batches, maps = [], []
    for b in range(query.shape[0]):
        mixed = torch.zeros(
            query.shape[1], query.shape[1], dtype=query.dtype, device=query.device
        )
        for h in range(module.num_heads):
            sl = slice(h * module.head_dim, (h + 1) * module.head_dim)
            score = q[b, :, sl] @ k[b, :, sl].T / math.sqrt(module.head_dim)
            mixed = mixed + module.head_mix.weight[0, h] * score.softmax(-1)
        batches.append(mixed @ value[b])
        maps.append(mixed)
    return torch.stack(batches), torch.stack(maps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("preserve_norm", [False, True])
def test_token_roundtrip_basis_batching_and_norm(dtype, preserve_norm):
    x = spd((2, 3, 2, 4, 3, 3), dtype)
    tokenizer = SPDLogTokenizer(3, preserve_norm=preserve_norm)
    tokens = tokenizer(x)
    assert tokens.shape == (2, 3, 2, 4, 6)
    torch.testing.assert_close(tokenizer.inverse(tokens), x, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(tokens[1, 2], tokenizer(x[1, 2]))
    symmetric = torch.tensor([[1.0, 0.2], [0.2, -0.3]], dtype=dtype)
    x = torch.matrix_exp(symmetric)
    t = spd_to_log_tokens(x, preserve_norm=preserve_norm)
    expected = torch.tensor(
        [1.0, 0.2 * (math.sqrt(2) if preserve_norm else 1), -0.3], dtype=dtype
    )
    torch.testing.assert_close(t, expected)
    if preserve_norm:
        torch.testing.assert_close(t.square().sum(), symmetric.square().sum())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_repeated_spectrum_and_ill_conditioned_finite_backward(dtype):
    x = torch.eye(3, dtype=dtype).repeat(2, 1, 1).requires_grad_()
    z = spd_to_log_tokens(x, preserve_norm=True)
    torch.testing.assert_close(log_tokens_to_spd(z, preserve_norm=True), x)
    z.sum().backward()
    expected = torch.full((3, 3), 1 / math.sqrt(2), dtype=dtype)
    expected.diagonal().fill_(1)
    torch.testing.assert_close(x.grad, expected.expand_as(x))
    near = torch.diag(torch.tensor([1e-8, 0.2, 2], dtype=dtype)).requires_grad_()
    tokens = spd_to_log_tokens(near, eigenvalue_floor=0.01)
    reconstructed = log_tokens_to_spd(tokens)
    torch.testing.assert_close(
        reconstructed, torch.diag(torch.tensor([0.01, 0.2, 2], dtype=dtype))
    )
    tokens.square().sum().backward()
    assert torch.isfinite(near.grad).all()
    assert near.grad[0, 0] == 0


def test_gradient_checks_log_and_inverse_at_repeated_spectrum():
    t = torch.zeros(2, 6, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda v: log_tokens_to_spd(v, preserve_norm=True), (t,))
    # Factor parametrization keeps perturbed finite-difference inputs symmetric SPD.
    a = torch.eye(3, dtype=torch.float64).requires_grad_()
    assert gradcheck(lambda v: spd_to_log_tokens(v @ v.mT, preserve_norm=True), (a,))
    x = torch.diag(torch.tensor([1e-8, 0.5, 2.0], dtype=torch.float64))
    torch.testing.assert_close(log_tokens_to_spd(spd_to_log_tokens(x)), x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_psd_floor_and_scaling_order(dtype):
    x = torch.diag(torch.tensor([0.0, 0.1, 2.0], dtype=dtype)).requires_grad_()
    with pytest.raises(ValueError, match="SPD"):
        spd_to_log_tokens(x)
    module = SPDLogTokenizer(3, matrix_scale=2, eigenvalue_floor=0.3)
    reconstructed = module.inverse(module(x))
    torch.testing.assert_close(
        reconstructed, torch.diag(torch.tensor([0.15, 0.15, 2.0], dtype=dtype))
    )
    reconstructed.sum().backward()
    assert torch.isfinite(x.grad).all()
    tiny = torch.diag(torch.tensor([1e-8, 0.1, 2], dtype=dtype))
    result = log_tokens_to_spd(spd_to_log_tokens(tiny))
    torch.testing.assert_close(
        result[0, 0],
        torch.tensor(max(1e-8, get_epsilon(dtype, "eigval_log")), dtype=dtype),
    )


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(2, 3),
        torch.eye(3).to(torch.complex64),
        torch.eye(3).half(),
        torch.full((3, 3), float("nan")),
        -torch.eye(3),
    ],
)
def test_token_validation(bad):
    with pytest.raises((ValueError, TypeError)):
        spd_to_log_tokens(bad)


def test_invalid_coordinates_and_configuration():
    with pytest.raises(ValueError, match="symmetric"):
        spd_to_log_tokens(torch.tensor([[2.0, 1], [0, 2.0]]))
    for kwargs in (
        {"matrix_scale": 0},
        {"eigenvalue_floor": -1},
        {"matrix_scale": float("inf")},
    ):
        with pytest.raises(ValueError):
            spd_to_log_tokens(torch.eye(2), **kwargs)
    for d in (0, 2, 4, 8):
        with pytest.raises(ValueError):
            log_tokens_to_spd(torch.zeros(d))
    for kwargs in (
        {"context_epochs": 2},
        {"feature_tokens": 11},
        {"feedforward_dim": 12},
        {"classifier_dim": 12},
    ):
        with pytest.raises(ValueError):
            SPDTransNet(**kwargs)
    for kwargs in ({"embed_dim": 8}, {"num_heads": 4}, {"dropout": -0.1}):
        with pytest.raises(ValueError):
            StructurePreservingMultiheadAttention(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_map_and_model_inputs_raise_explicit_errors(dtype):
    with pytest.raises(ValueError, match="Empty inputs"):
        spd_to_log_tokens(torch.empty(0, 3, 3, dtype=dtype), eigenvalue_floor=0.01)
    with pytest.raises(ValueError, match="Empty inputs"):
        log_tokens_to_spd(torch.empty(0, 6, dtype=dtype))
    model = small_model(dtype)
    with pytest.raises(ValueError, match="Empty inputs"):
        model(torch.empty(0, 3, 2, 4, 3, 3, dtype=dtype))
    with pytest.raises(ValueError, match="Empty batches"):
        model.forward_tokens(torch.empty(0, 3, 2, 4, 6, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1e6])
def test_input_validation_uses_relative_per_matrix_scale(dtype, scale):
    asymmetric = scale * torch.tensor([[1.0, 0.01], [0.0, 1.0]], dtype=dtype)
    with pytest.raises(ValueError, match="symmetric"):
        spd_to_log_tokens(asymmetric, eigenvalue_floor=0.01)
    for spectrum in ([-1.0, -1.0], [-0.01, 1.0]):
        indefinite = scale * torch.diag(torch.tensor(spectrum, dtype=dtype))
        for batch in (
            indefinite[None],
            torch.stack((indefinite, 1e12 * scale * torch.eye(2, dtype=dtype))),
        ):
            with pytest.raises(ValueError, match="input must be SPD"):
                spd_to_log_tokens(batch, eigenvalue_floor=0.01)
    # Declared flooring still accepts genuine PSD data at the same tiny units.
    psd = scale * torch.diag(torch.tensor([0.0, 1.0], dtype=dtype))
    assert torch.isfinite(spd_to_log_tokens(psd, eigenvalue_floor=0.01)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_attention_equation_signed_mixing_unaltered_values(dtype):
    torch.manual_seed(21)
    module = StructurePreservingMultiheadAttention(6, 3, dtype=dtype)
    with torch.no_grad():
        module.head_mix.weight.copy_(torch.tensor([[-2.0, 0.1, 0.2]], dtype=dtype))
    q, k, v = [torch.randn(2, 4, 6, dtype=dtype) for _ in range(3)]
    output, weights = module(q, k, v, need_weights=True)
    reference, reference_weights = reference_attention(module, q, k, v)
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(weights, reference_weights)
    assert (weights < 0).any()
    torch.testing.assert_close(weights.sum(-1), torch.full((2, 4), -1.7, dtype=dtype))
    matrices = log_tokens_to_spd(output)
    assert torch.linalg.eigvalsh(matrices).min() > 0
    torch.testing.assert_close(module(q[:1], k[:1], v[:1])[0], output[:1])
    # A signed sum of raw SPD matrices can be negative definite; the log map is essential.
    with torch.no_grad():
        module.head_mix.weight.fill_(-1)
    identity_tokens = spd_to_log_tokens(2 * torch.eye(3, dtype=dtype)).expand(2, 4, 6)
    out, _ = module(identity_tokens)
    torch.testing.assert_close(
        log_tokens_to_spd(out), torch.eye(3, dtype=dtype).expand(2, 4, 3, 3) / 8
    )


def test_attention_gradcheck_parameters_and_values():
    torch.manual_seed(22)
    module = StructurePreservingMultiheadAttention(3, 3, dtype=torch.float64)
    x = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda t: module(t)[0], (x,))
    # Include the actual registered parameters in a numerical gradient check.
    names, parameters = zip(*module.named_parameters())

    def functional(*args):
        return torch.func.functional_call(
            module, dict(zip(names, args[1:])), (args[0],)
        )[0]

    assert gradcheck(functional, (x, *parameters))


def test_position_and_encoder_reference():
    torch.manual_seed(13)
    position = LearnedSinusoidalPosition(4, 3, dtype=torch.float64)
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    table = torch.empty(4, 3, dtype=torch.float64)
    for p in range(4):
        for c in range(3):
            angle = p / 10000 ** (2 * (c // 2) / 3)
            table[p, c] = math.sin(angle) if c % 2 == 0 else math.cos(angle)
    torch.testing.assert_close(position.sinusoid, table)
    expected = F.linear(
        F.gelu(
            F.linear(x + table, position.mapping[0].weight, position.mapping[0].bias)
        ),
        position.mapping[2].weight,
        position.mapping[2].bias,
    )
    torch.testing.assert_close(position(x), expected)
    encoder = SPDTransformerEncoderLayer(3, 3, 6, dropout=0, dtype=torch.float64)
    attention, _ = reference_attention(encoder.attention, x, x, x)
    y = F.layer_norm(
        x + attention, (3,), encoder.norm1.weight, encoder.norm1.bias, 1e-5
    )
    ff = F.linear(
        F.relu(F.linear(y, encoder.ff[0].weight, encoder.ff[0].bias)),
        encoder.ff[3].weight,
        encoder.ff[3].bias,
    )
    expected = F.layer_norm(
        y + ff, (3,), encoder.norm2.weight, encoder.norm2.bias, 1e-5
    )
    torch.testing.assert_close(encoder(x), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_shape_order_isolation_and_classifier_target(dtype):
    torch.manual_seed(42)
    model = small_model(dtype).eval()
    x = spd((2, 3, 2, 4, 3, 3), dtype)
    tokens = model.tokenizer(x)
    features = model.forward_features(x)
    assert features.shape == (2, 3, 2, 6)
    output = model(x)
    assert output.shape == (2, 5)
    torch.testing.assert_close(output, model.forward_tokens(tokens))
    torch.testing.assert_close(output, model.classifier(features[:, 1].flatten(1)))
    torch.testing.assert_close(model(x[:1]), output[:1])
    changed = x.clone()
    changed[1] *= 2
    torch.testing.assert_close(model(changed)[0], output[0])
    projected = model.projection(tokens).flatten(2, 3)
    # Verify band-major order before folding B*L into the independent epoch batch.
    torch.testing.assert_close(projected[:, :, 4:8], model.projection(tokens[:, :, 1]))
    intra = model.intra_epoch(projected)
    changed = projected.clone()
    changed[:, 0] += 4
    torch.testing.assert_close(model.intra_epoch(changed)[:, 1:], intra[:, 1:])
    # Independent per-epoch invocation equals the batched intra-epoch path.
    for epoch in range(3):
        manual = (
            model.intra_epoch.encoder(projected[:, epoch]).reshape(2, 2, 4, 6).mean(-2)
        )
        torch.testing.assert_close(intra[:, epoch], manual)
    # Future context really participates; this is not a causal encoder.
    changed_intra = intra.clone()
    changed_intra[:, 2] += torch.arange(6, dtype=dtype)
    assert not torch.allclose(model.inter_epoch(changed_intra)[:, 1], features[:, 1])
    # Identity/order is retained across inter-epoch flattening.
    manual_inter = model.inter_epoch.encoder(intra.flatten(1, 2)).reshape_as(features)
    torch.testing.assert_close(features, manual_inter)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_backward_optimizer_and_state(dtype):
    torch.manual_seed(17)
    model = small_model(dtype)
    x = spd((2, 3, 2, 4, 3, 3), dtype).requires_grad_()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, eps=1e-7, weight_decay=1e-5
    )
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    loss = F.cross_entropy(model(x), torch.tensor([1, 3]), label_smoothing=0.1)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    # Selected active paths must receive data gradients, not only weight decay.
    for fragment in (
        "projection.weight",
        "attention.query.weight",
        "attention.key.weight",
        "attention.head_mix.weight",
        "position.mapping.0.weight",
        "ff.0.weight",
        "classifier.0.weight",
    ):
        selected = [(n, p) for n, p in model.named_parameters() if fragment in n]
        assert selected
        for name, p in selected:
            assert p.grad.abs().max() > 0, name
    optimizer.step()
    for name, p in model.named_parameters():
        if name.endswith("weight"):
            assert not torch.equal(p, before[name]), name
    model.eval()
    buffer = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer
    )
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = small_model(dtype).eval()
    restored.load_state_dict(checkpoint["model"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
    assert len(restored_optimizer.state) == len(optimizer.state)
    # Continue both training paths from the same model and optimizer state.
    for net, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(net(x.detach()), torch.tensor([0, 4])).backward()
        opt.step()
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dropout_train_eval_and_layer_state(dtype):
    torch.manual_seed(12)
    model = small_model(dtype, dropout=0.5)
    x = spd((2, 3, 2, 4, 3, 3), dtype)
    assert not torch.equal(model(x), model(x))
    model.eval()
    torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
    # Dropout p=1 zeros all attention maps only in training.
    attention = StructurePreservingMultiheadAttention(6, 2, dropout=1, dtype=dtype)
    tokens = torch.randn(2, 4, 6, dtype=dtype)
    assert torch.count_nonzero(attention(tokens)[0]) == 0
    attention.eval()
    assert torch.count_nonzero(attention(tokens)[0]) > 0
    layers = model.intra_epoch.encoder.layers
    torch.testing.assert_close(layers[0].ff[0].weight, layers[1].ff[0].weight)
    assert layers[0].ff[0].weight.data_ptr() != layers[1].ff[0].weight.data_ptr()


def test_explicit_mask_shape_and_registry_contracts():
    import spd_learn.modules as modules

    module = StructurePreservingMultiheadAttention()
    tokens = torch.zeros(2, 4, 6)
    for kwargs in (
        {"attn_mask": torch.zeros(4, 4)},
        {"key_padding_mask": torch.ones(2, 4, dtype=torch.bool)},
        {"is_causal": True},
    ):
        with pytest.raises(ValueError, match="not supported"):
            module(tokens, **kwargs)
    with pytest.raises(ValueError):
        module(tokens, tokens[:, :2], tokens)
    with pytest.raises(ValueError):
        module(torch.zeros(2, 0, 6))
    model = small_model()
    with pytest.raises(ValueError, match="not supported"):
        model(torch.empty(2, 3, 2, 4, 3, 3), key_padding_mask=torch.ones(2, 3))
    with pytest.raises(ValueError, match="expected prepared SPD"):
        model(torch.empty(2, 3, 100))
    with pytest.raises(ValueError, match="expected log tokens"):
        model.forward_tokens(torch.empty(2, 2, 3, 4, 6))
    assert "SPDTransNet" in __sequence_models__
    assert "SPDTransNet" not in __all__
    sequence_modules = {
        "SPDLogTokenizer",
        "StructurePreservingMultiheadAttention",
        "SPDTransformerEncoderLayer",
        "LearnedSinusoidalPosition",
        "SPDTemporalEncoder",
        "SPDIntraEpochEncoder",
        "SPDInterEpochEncoder",
    }
    assert sequence_modules <= set(modules.__sequence_modules__)
    assert all(hasattr(modules, name) for name in sequence_modules)
    assert SPDTransNet().token_dim == 351


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_model_repeated_rank_deficient_and_central_classifier(dtype):
    """The full path accepts explicit PSD rectification without losing identity."""
    torch.manual_seed(83)
    model = small_model(dtype)
    model.tokenizer.eigenvalue_floor = 0.01
    # Rotated PSD matrices have a repeated zero eigenvalue, not merely a diagonal
    # repeated-spectrum fixture. The regularizer is explicit and intentionally lossy.
    direction = torch.tensor([1.0, 2.0, -1.0], dtype=dtype)
    psd = direction[:, None] @ direction[None, :]
    x = psd.expand(2, 3, 2, 4, 3, 3).clone().requires_grad_()
    features = model.forward_features(x)
    y = model(x)
    central = features[:, 1].flatten(1)
    for layer in model.classifier:
        central = layer(central)
    torch.testing.assert_close(y, central)
    F.cross_entropy(y, torch.tensor([1, 2])).backward()
    assert torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    # Changing one recording-local context cannot alter another batch item.
    changed = x.detach().clone()
    changed[0] = torch.eye(3, dtype=dtype)
    torch.testing.assert_close(model(changed)[1], y[1])


def test_pooling_agrees_with_published_equal_bin_aggregation():
    model = small_model()
    tokens = torch.randn(2, 3, 8, 6, dtype=torch.float64)
    encoded = model.intra_epoch.encoder(tokens.reshape(6, 8, 6))
    expected = F.adaptive_avg_pool2d(encoded, (2, 6)).reshape(2, 3, 2, 6)
    torch.testing.assert_close(model.intra_epoch(tokens), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_parity_and_full_backward(dtype):
    torch.manual_seed(14)
    cpu = small_model(dtype).eval()
    gpu = copy.deepcopy(cpu).cuda()
    x = spd((2, 3, 2, 4, 3, 3), dtype).requires_grad_()
    x_gpu = x.detach().cuda().requires_grad_()
    y_cpu, y_gpu = cpu(x), gpu(x_gpu)
    torch.testing.assert_close(y_gpu.cpu(), y_cpu, rtol=3e-4, atol=3e-5)
    y_cpu.square().sum().backward()
    y_gpu.square().sum().backward()
    torch.testing.assert_close(x_gpu.grad.cpu(), x.grad, rtol=1e-3, atol=1e-5)
    for (name, p), q in zip(cpu.named_parameters(), gpu.parameters()):
        assert torch.isfinite(q.grad).all(), name
        torch.testing.assert_close(q.grad.cpu(), p.grad, rtol=2e-3, atol=2e-5)
