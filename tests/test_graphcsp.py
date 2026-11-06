# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import copy
import io

from unittest.mock import patch

import pytest
import torch

from spd_learn.functional import normalize_spd_adjacency, spd_graph_aggregate
from spd_learn.models import GraphCSPNet
from spd_learn.modules import GraphBiMap, TimeFrequencyCovariances, TimeFrequencyGraph


def make_spd(shape, dtype=torch.float64, device="cpu"):
    x = torch.randn(*shape, dtype=dtype, device=device)
    return x @ x.mT + 0.2 * torch.eye(shape[-1], dtype=dtype, device=device)


def graph(dtype=torch.float64):
    return TimeFrequencyGraph(
        torch.tensor([[0.0, 6.0], [1.0, 6.0], [0.0, 10.0], [0.0, 14.0]], dtype=dtype),
        time_radius=1.0,
        frequency_radius=4.0,
        components=torch.tensor([0, 0, 0, 1]),
        bandwidth=2.0,
    )


def test_covariance_node_channel_and_trial_order():
    x = torch.arange(2 * 2 * 3 * 8, dtype=torch.float64).reshape(2, 2, 3, 8)
    segments = [(1, 4, 8), (0, 0, 2), (1, 0, 8)]
    layer = TimeFrequencyCovariances(segments, ridge=0.01)
    actual = layer(x)
    for trial in range(2):
        for node, (band, start, stop) in enumerate(segments):
            window = x[trial, band, :, start:stop]
            torch.testing.assert_close(
                actual[trial, node],
                window @ window.T / (stop - start) + 0.01 * torch.eye(3),
            )
    torch.testing.assert_close(
        layer(x[:, :, [2, 0, 1]]), actual[:, :, [2, 0, 1]][:, :, :, [2, 0, 1]]
    )
    assert torch.linalg.eigvalsh(actual).min() > 0
    centered = TimeFrequencyCovariances(segments, center=True, shrinkage=0.3, ridge=0.1)
    assert torch.isfinite(centered(x)).all()
    torch.testing.assert_close(
        centered(torch.zeros_like(x)),
        0.1 * torch.eye(3, dtype=x.dtype).expand(2, 3, 3, 3),
    )


def test_graph_topology_fit_reference_and_serialization():
    g = graph()
    assert not g.topology.diagonal().any()
    assert not g.topology[3].any()
    torch.testing.assert_close(
        g()[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    )
    # Diagonal matrices give analytic AIRM distance, independently of whitening.
    lattice = torch.diag_embed(
        torch.tensor(
            [[1.0, 2.0], [2.0, 3.0], [4.0, 1.0], [2.0, 2.0]], dtype=torch.float64
        )
    )
    data = lattice.unsqueeze(0).repeat(3, 1, 1, 1)
    g.fit(data)
    logs = lattice.diagonal(dim1=-2, dim2=-1).log()
    d2 = (logs[:, None] - logs[None, :]).square().sum(-1)
    weights = torch.exp(-d2 / 2) * g.topology
    expected = weights + torch.eye(4, dtype=torch.float64)
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(g(), expected)
    assert g.fitted
    fresh = graph()
    fresh.load_state_dict(g.state_dict())
    torch.testing.assert_close(fresh(), g())
    torch.testing.assert_close(fresh.lattice, lattice)
    before = g().clone()
    _ = spd_graph_aggregate(make_spd((5, 4, 2, 2)), g())
    torch.testing.assert_close(g(), before)
    g.fit(2 * data)
    torch.testing.assert_close(g.lattice, 2 * lattice)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("out_features", [2, 5])
def test_graph_bimap_reference_batching_and_optimizer(dtype, out_features):
    torch.manual_seed(14)
    x = make_spd((2, 3, 4, 3, 3), dtype=dtype).requires_grad_()
    p = normalize_spd_adjacency(torch.rand(2, 3, 4, 4, dtype=dtype))
    module = GraphBiMap(4, 3, out_features, dtype=dtype)
    y = module(x, p)
    expected = torch.empty_like(y)
    w = module.projection.weight
    for a in range(2):
        for b in range(3):
            for n in range(4):
                aggregate = sum(p[a, b, n, j] * x[a, b, j] for j in range(4))
                if out_features > 3:
                    aggregate = torch.block_diag(
                        aggregate, torch.eye(out_features - 3, dtype=dtype)
                    )
                expected[a, b, n] = w[n].T @ aggregate @ w[n]
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(module(x[0, 0], p[0, 0]), y[0, 0])
    optim = torch.optim.Adam(module.parameters(), lr=0.01)
    originals = [v.detach().clone() for v in module.parameters()]
    (y * torch.randn_like(y)).sum().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    optim.step()
    assert all(
        not torch.equal(old, new) for old, new in zip(originals, module.parameters())
    )
    assert torch.linalg.eigvalsh(module(x, p)).min() > 0
    torch.testing.assert_close(
        module.projection.weight.mT @ module.projection.weight,
        torch.eye(out_features, dtype=dtype).expand(4, -1, -1),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_model_gradients_state_and_eval_isolation(dtype):
    torch.manual_seed(23)
    model = GraphCSPNet(3, 2, n_freqs=2, n_patches=2, dtype=dtype)
    x = torch.randn(4, 2, 3, 32, dtype=dtype)
    old_mean = model.norm.running_mean.clone()
    model.fit_graph(x)
    torch.testing.assert_close(old_mean, model.norm.running_mean)
    p = model.graph().clone()
    before = {n: v.detach().clone() for n, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logits = model(x)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not torch.equal(before[name], parameter), name
    assert not torch.equal(old_mean, model.norm.running_mean)
    torch.testing.assert_close(p, model.graph())
    assert torch.linalg.eigvalsh(model.norm.bias).min() > 0
    model.eval()
    running = model.norm.running_mean.clone()
    torch.testing.assert_close(model(x[:1]), model(x)[:1], atol=3e-5, rtol=3e-5)
    assert torch.isfinite(model(x)).all()
    torch.testing.assert_close(running, model.norm.running_mean)
    fresh = GraphCSPNet(3, 2, n_freqs=2, n_patches=2, dtype=dtype).eval()
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    fresh.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(fresh(x), model(x))
    torch.testing.assert_close(
        model.forward_covariances(model.covariances(x)), model(x)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_repeated_and_ill_conditioned_backward(dtype):
    model = GraphCSPNet(
        3, 2, n_freqs=1, n_patches=2, dtype=dtype, require_fitted_graph=False
    ).eval()
    for spectrum in ([1.0, 1.0, 1.0], [1e-7, 1e-4, 1.0]):
        x = (
            torch.diag(torch.tensor(spectrum, dtype=dtype))
            .expand(2, 2, 3, 3)
            .clone()
            .requires_grad_()
        )
        model.zero_grad(set_to_none=True)
        model.forward_covariances(x).square().sum().backward()
        assert torch.isfinite(x.grad).all()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )


def test_gradcheck_graph_projection():
    module = GraphBiMap(2, 3, 2, dtype=torch.float64)
    x = make_spd((2, 2, 3, 3)).requires_grad_()
    p = normalize_spd_adjacency(torch.ones(2, 2, dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda z: module(z, p), (x,))


def test_nonuniform_default_and_explicit_segments():
    model = GraphCSPNet(3, 2)
    assert model.identity_graph.shape == (48, 48)
    x = torch.randn(2, 9, 3, 65)
    nodes = model.covariances(x)
    assert nodes.shape == (2, 48, 3, 3)
    for node, (band, count, window) in enumerate(
        (b, c, w) for b, c in enumerate(model.window_counts) for w in range(c)
    ):
        segment = TimeFrequencyCovariances(
            [(band, window * 65 // count, (window + 1) * 65 // count)],
            center=model.center,
            shrinkage=model.shrinkage,
        )
        torch.testing.assert_close(nodes[:, node], segment(x)[:, 0])
    g = graph(torch.float32)
    custom = GraphCSPNet(
        3,
        2,
        n_freqs=2,
        graph=g,
        segments=[(0, 0, 8), (0, 8, 16), (1, 0, 16), (1, 16, 32)],
        use_mlp=True,
        require_fitted_graph=False,
    )
    assert custom(torch.randn(2, 2, 3, 32)).shape == (2, 2)


def test_invalid_graphs_and_inputs():
    for bad in (
        torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
        torch.full((2, 2), float("nan")),
        torch.ones(2, 3),
    ):
        with pytest.raises(ValueError):
            normalize_spd_adjacency(bad)
    with pytest.raises(ValueError):
        spd_graph_aggregate(make_spd((2, 4, 2, 2)), torch.zeros(4, 4))
    with pytest.raises(ValueError):
        graph().fit(torch.zeros(2, 4, 2, 2, dtype=torch.float64))
    with pytest.raises(ValueError):
        TimeFrequencyCovariances([(0, 0, 100)])(torch.randn(2, 1, 3, 5))
    with pytest.raises(ValueError):
        GraphCSPNet(3, 2)(torch.randn(2, 3, 10))
    for kwargs in ({"ridge": float("inf")}, {"shrinkage": float("nan")}):
        with pytest.raises(ValueError):
            GraphCSPNet(**kwargs)
    with pytest.raises(ValueError):
        spd_graph_aggregate(make_spd((2, 4, 2, 2)), torch.tensor(1.0))
    with pytest.raises(ValueError):
        TimeFrequencyGraph(
            torch.zeros(2, 2), time_radius=float("nan"), frequency_radius=1
        )


def test_box_and_forward_neighborhood_difference():
    coordinates = torch.tensor([[0.0, 14.0], [1.0, 18.0], [0.0, 18.0], [1.0, 14.0]])
    box = TimeFrequencyGraph(coordinates, time_radius=1, frequency_radius=4)
    forward = TimeFrequencyGraph(
        coordinates, time_radius=1, frequency_radius=4, neighborhood="forward"
    )
    assert box.topology[2, 3] and not forward.topology[2, 3]
    assert forward.topology[0, 1] and forward.topology[1, 0]
    assert forward.topology[0, 2] and forward.topology[0, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_parity_backward(dtype):
    torch.manual_seed(3)
    cpu = GraphCSPNet(3, 2, n_freqs=2, n_patches=2, dtype=dtype).eval()
    x = torch.randn(3, 2, 3, 32, dtype=dtype)
    cpu.fit_graph(x)
    gpu = copy.deepcopy(cpu).cuda()
    actual = gpu(x.cuda())
    torch.testing.assert_close(actual.cpu(), cpu(x), atol=3e-5, rtol=3e-5)
    actual.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in gpu.parameters()
    )


def test_default_topology_matches_bcic_forward_steps():
    model = GraphCSPNet()
    assert model.dims == (22, 36, 22)
    assert model.graph.bandwidth == 50
    # Independently enumerate band/window indices, including the orientation
    # exclusion: rising frequency cannot connect to an earlier time before
    # reciprocal edges are added. No flattening wraps to the next band.
    nodes = [(b, w, c) for b, c in enumerate(model.window_counts) for w in range(c)]
    components = [0, 1, 2, 2, 2, 2, 3, 3, 3]
    expected = torch.zeros(48, 48, dtype=torch.bool)
    for i, (band_i, window_i, count_i) in enumerate(nodes):
        for j, (band_j, window_j, _) in enumerate(nodes):
            if components[band_i] != components[band_j] or i == j:
                continue
            if band_i <= band_j and 0 <= window_j - window_i <= count_i // 2:
                expected[i, j] = expected[j, i] = True
    torch.testing.assert_close(model.graph.topology, expected)
    assert not model.graph.topology[11, 12]  # beta wraps backward in time
    assert model.graph.topology[8, 20]  # three-band separation, same time
    assert model.graph.topology[24, 28]  # four gamma time steps


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_noncommuting_lgt_distance_and_fit_boundary(dtype):
    torch.manual_seed(19)
    g = graph(dtype)
    data = make_spd((5, 4, 3, 3), dtype=dtype)
    g.fit(data)
    lattice = data.mean(0)
    # Cholesky whitening is an independent AIRM distance oracle; eigenvalues
    # equal those of the symmetric inverse-square-root congruence.
    chol = torch.linalg.cholesky(lattice)
    inverse = torch.linalg.inv(chol)
    relative = inverse[:, None] @ lattice[None, :] @ inverse[:, None].mT
    d2 = torch.linalg.eigvalsh(relative).log().square().sum(-1)
    a = torch.exp(-d2 / 2) * g.topology
    expected = a + torch.eye(4, dtype=dtype)
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(g(), expected, atol=2e-6, rtol=2e-6)
    assert not g.lattice.requires_grad
    with pytest.raises(ValueError, match="dtype/device"):
        g.fit(data.to(torch.float64 if dtype == torch.float32 else torch.float32))
    torch.testing.assert_close(g.lattice, lattice)  # failed fit does not overwrite
    # A fresh fold has no fitted lattice, even though its topology is usable.
    fresh = graph(dtype)
    assert not fresh.fitted and fresh.lattice.numel() == 0
    # Loading into a chosen dtype must keep all graph buffers consistent.
    restored = graph(torch.float32)
    restored.load_state_dict(g.state_dict())
    assert restored.lattice.dtype == restored.propagation.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aggregation_isolation_permutation_and_extreme_weights(dtype):
    x = make_spd((2, 4, 3, 3), dtype=dtype).requires_grad_()
    a = torch.rand(2, 4, 4, dtype=dtype)
    p = normalize_spd_adjacency(a)
    output = spd_graph_aggregate(x, p)
    output[0].sum().backward()
    assert torch.count_nonzero(x.grad[1]) == 0
    perm = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(
        spd_graph_aggregate(x[:, perm], p[:, perm][:, :, perm]), output[:, perm]
    )
    assert torch.linalg.eigvalsh(output).min() > 0
    huge = normalize_spd_adjacency(
        torch.full((4, 4), torch.finfo(dtype).max, dtype=dtype)
    )
    assert torch.isfinite(huge).all()
    torch.testing.assert_close(huge.sum(-1), torch.ones(4, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rank_deficient_preprocessing_backward(dtype):
    signals = torch.ones(2, 1, 3, 4, dtype=dtype, requires_grad=True)
    preprocessing = TimeFrequencyCovariances(
        [(0, 0, 1), (0, 1, 4)], center=True, shrinkage=0.01, ridge=0.001
    )
    cov = preprocessing(signals)
    assert torch.linalg.eigvalsh(cov).min() > 0
    cov.sum().backward()
    assert torch.isfinite(signals.grad).all()


def test_model_requires_explicit_graph_fit_and_no_inference_refit():
    model = GraphCSPNet(3, 2, n_freqs=2, n_patches=2)
    x = torch.randn(3, 2, 3, 32)
    with pytest.raises(RuntimeError, match="Fit the LGT"):
        model(x)
    model.fit_graph(x)
    before = {name: value.clone() for name, value in model.graph.state_dict().items()}
    model.eval()
    model(3 * x)
    for name, value in model.graph.state_dict().items():
        torch.testing.assert_close(value, before[name])
    with pytest.raises(ValueError):
        model.forward_covariances(torch.ones(2, 3, 3))


def test_default_expansion_reduction_optimizer_preserves_spd():
    torch.manual_seed(17)
    model = GraphCSPNet(dtype=torch.float64)
    x = torch.randn(2, 9, 22, 64, dtype=torch.float64)
    model.fit_graph(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = [layer.projection.weight.detach().clone() for layer in model.layers]
    model(x).square().mean().backward()
    optimizer.step()
    nodes = model.covariances(x)
    for i, layer in enumerate(model.layers):
        nodes = layer(nodes, model.graph() if i == 0 else model.identity_graph)
        assert torch.linalg.eigvalsh(nodes).min() > 0
        assert not torch.equal(layer.projection.weight, before[i])
    assert nodes.shape == (2, 48, 22, 22)


def test_volt_scale_warning_and_explicit_unit_conversion():
    torch.manual_seed(0)
    model = GraphCSPNet(4, 2, n_freqs=2, n_patches=1, dims=(4, 4)).eval()
    volts = torch.randn(2, 2, 4, 64) * 5e-5
    with pytest.warns(UserWarning, match="Pre-scale signal units"):
        cov = model.covariances(volts)
    assert torch.linalg.eigvalsh(cov).max() < model.rectify.threshold_
    rectified = model.rectify(model.layers[0](cov, model.graph()))
    assert (rectified[:, 0] - rectified[:, 1]).abs().max() < 1e-9
    microvolts = model.covariances(volts * 1e6)
    assert torch.linalg.eigvalsh(microvolts).min() > 1
    preserved = model.rectify(model.layers[0](microvolts, model.graph()))
    assert (preserved[:, 0] - preserved[:, 1]).abs().max() > 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_float32_expansion_conditioning_has_explicit_postcondition(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(0)
    module = GraphBiMap(1, 22, 36, dtype=torch.float32).to(device)
    x = torch.diag(torch.logspace(0, 7, 22, device=device))[None]
    propagation = torch.ones(1, 1, device=device)
    # Precision boundaries vary by backend: return strict SPD or fail clearly.
    try:
        output = module(x, propagation)
    except ValueError as error:
        assert "float64" in str(error) and "regularization" in str(error)
    else:
        assert torch.equal(output, output.mT)
        assert torch.linalg.eigvalsh(output).min() > 0
        torch.linalg.cholesky(output)
    output = module.double()(x.double(), propagation.double())
    assert torch.linalg.eigvalsh(output).min() > 0
    torch.linalg.cholesky(output)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_custom_graph_construction_and_all_buffer_placement(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    g = TimeFrequencyGraph(
        torch.tensor([[0.5, 6.0]], dtype=torch.float64),
        time_radius=1.0,
        frequency_radius=20.0,
    )
    with pytest.raises(ValueError, match="construction dtype/device"):
        GraphCSPNet(4, 2, n_freqs=1, n_patches=1, graph=g, device=device)
    model = GraphCSPNet(
        4,
        2,
        n_freqs=1,
        n_patches=1,
        graph=g.to(device),
        device=device,
        dtype=torch.float64,
        require_fitted_graph=False,
    ).eval()
    assert model.graph is g
    expected_device = torch.empty(0, device=device).device
    assert all(value.device == expected_device for value in model.buffers())
    assert model.log.autograd_.device == expected_device
    assert model(
        torch.randn(2, 1, 4, 64, device=device, dtype=torch.float64)
    ).shape == (2, 2)


def test_failed_numerical_graph_fit_is_atomic():
    g = graph(torch.float32)
    data = make_spd((3, 4, 2, 2), dtype=torch.float32)
    g.fit(data)
    before = {name: value.clone() for name, value in g.state_dict().items()}
    enormous = (3e38 * torch.eye(2)).expand(2, 4, 2, 2)
    with pytest.raises(ValueError, match="lattice overflow"):
        g.fit(enormous)
    for name, value in g.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    with patch(
        "spd_learn.modules.graph.normalize_spd_adjacency",
        side_effect=ValueError("normalization failed"),
    ):
        with pytest.raises(ValueError, match="normalization failed"):
            g.fit(2 * data)
    for name, value in g.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_root_export_and_covariance_overflow():
    import spd_learn

    assert spd_learn.GraphCSPNet is GraphCSPNet
    assert "GraphCSPNet" in spd_learn.__all__
    with pytest.raises(ValueError, match="covariance overflow"):
        TimeFrequencyCovariances([(0, 0, 4)])(torch.full((2, 1, 2, 4), 3e38))


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("mismatch", ["nodes", "channels", "classifier"])
def test_failed_model_checkpoint_load_restores_values_and_ownership(assign, mismatch):
    source = GraphCSPNet(3, 2, n_freqs=2, n_patches=2, dtype=torch.float64)
    source.fit_graph(torch.randn(2, 2, 3, 32, dtype=torch.float64))
    if mismatch == "nodes":
        g = TimeFrequencyGraph(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64),
            time_radius=2.0,
            frequency_radius=0.0,
        )
        target = GraphCSPNet(
            3, 2, n_freqs=1, graph=g, segments=[(0, 0, 16)] * 3, dtype=torch.float64
        )
    else:
        target = GraphCSPNet(
            4 if mismatch == "channels" else 3,
            3 if mismatch == "classifier" else 2,
            n_freqs=2,
            n_patches=2,
            dtype=torch.float64,
        )
    before = copy.deepcopy(target.state_dict())
    parameters = list(target.parameters())
    buffers = list(target.buffers())
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    with pytest.raises(RuntimeError):
        target.load_state_dict(source.state_dict(), assign=assign)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert [id(p) for p in target.parameters()] == [id(p) for p in parameters]
    assert [id(b) for b in target.buffers()] == [id(b) for b in buffers]
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [
        id(p) for p in target.parameters()
    ]
    assert not target.graph.fitted
    assert target.graph.lattice.shape == (len(target.graph.coordinates), 0, 0)
    with pytest.raises(RuntimeError, match="Fit the LGT"):
        target(torch.randn(2, target.n_freqs, target.n_chans, 32, dtype=torch.float64))


def test_failed_graph_load_and_previously_fitted_model_are_atomic():
    target = GraphCSPNet(3, 2, n_freqs=2, n_patches=2, dtype=torch.float64).eval()
    x = torch.randn(2, 2, 3, 32, dtype=torch.float64)
    target.fit_graph(x)
    before = copy.deepcopy(target.state_dict())
    prediction = target(x)
    invalid = copy.deepcopy(before)
    invalid["classifier.weight"] = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        target.load_state_dict(invalid, assign=True)
    torch.testing.assert_close(target(x), prediction, atol=0, rtol=0)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    graph_before = copy.deepcopy(target.graph.state_dict())
    for invalid in (
        {"fitted": torch.tensor(True)},
        {**graph_before, "fitted": torch.tensor(False)},
        {**graph_before, "propagation": -torch.eye(4, dtype=torch.float64)},
    ):
        with pytest.raises(RuntimeError):
            target.graph.load_state_dict(invalid, strict=False, assign=True)
        for name, value in target.graph.state_dict().items():
            torch.testing.assert_close(value, graph_before[name], atol=0, rtol=0)


@pytest.mark.parametrize("assign", [False, True])
def test_valid_transactional_model_checkpoint_load(assign):
    source = GraphCSPNet(3, 2, n_freqs=1, n_patches=2, dtype=torch.float64).eval()
    x = torch.randn(2, 1, 3, 32, dtype=torch.float64)
    source.fit_graph(x)
    target = GraphCSPNet(3, 2, n_freqs=1, n_patches=2, dtype=torch.float64).eval()
    target.load_state_dict(source.state_dict(), assign=assign)
    torch.testing.assert_close(target(x), source(x), atol=0, rtol=0)
