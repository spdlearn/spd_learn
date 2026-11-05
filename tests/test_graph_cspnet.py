# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch

from spd_learn.models import GraphCSPNet
from spd_learn.modules import GraphBiMap


def _make_spd(shape, dtype=torch.float32, device="cpu"):
    """Helper to generate random symmetric positive definite matrices."""
    A = torch.randn(*shape, dtype=dtype, device=device)
    # A @ A.mT + eps * I ensures positive definiteness
    n = shape[-1]
    eye = torch.eye(n, dtype=dtype, device=device)
    return A @ A.mT + 0.1 * eye


# =============================================================================
# GraphBiMap Tests
# =============================================================================


def test_GraphBiMap_shape_reduction():
    """Test GraphBiMap dimension reduction: in_features > out_features."""
    n_nodes = 4
    in_features = 8
    out_features = 4
    batch_size = 3

    layer = GraphBiMap(n_nodes=n_nodes, in_features=in_features, out_features=out_features)
    X = _make_spd((batch_size, n_nodes, in_features, in_features))
    Y = layer(X)

    assert Y.shape == (batch_size, n_nodes, out_features, out_features)


def test_GraphBiMap_shape_expansion():
    """Test GraphBiMap dimension expansion: out_features > in_features."""
    n_nodes = 4
    in_features = 4
    out_features = 8
    batch_size = 2

    layer = GraphBiMap(n_nodes=n_nodes, in_features=in_features, out_features=out_features)
    X = _make_spd((batch_size, n_nodes, in_features, in_features))
    Y = layer(X)

    assert Y.shape == (batch_size, n_nodes, out_features, out_features)


def test_GraphBiMap_unbatched_3d():
    """Test GraphBiMap with 3D unbatched input: (n_nodes, in_feat, in_feat)."""
    n_nodes = 5
    in_features = 6
    out_features = 3

    layer = GraphBiMap(n_nodes=n_nodes, in_features=in_features, out_features=out_features)
    X = _make_spd((n_nodes, in_features, in_features))
    Y = layer(X)

    assert Y.shape == (n_nodes, out_features, out_features)


def test_GraphBiMap_multibatch_broadcast():
    """Test GraphBiMap with multiple leading batch dimensions."""
    n_nodes = 4
    in_features = 5
    out_features = 3
    batch_dims = (2, 3, 2)

    layer = GraphBiMap(n_nodes=n_nodes, in_features=in_features, out_features=out_features)
    X = _make_spd((*batch_dims, n_nodes, in_features, in_features))
    Y = layer(X)

    assert Y.shape == (*batch_dims, n_nodes, out_features, out_features)


def test_GraphBiMap_spd_preservation():
    """Verify that GraphBiMap strictly preserves the SPD property."""
    torch.manual_seed(42)
    n_nodes = 4
    in_features = 6
    out_features = 3

    for learnable in [False, True]:
        layer = GraphBiMap(
            n_nodes=n_nodes,
            in_features=in_features,
            out_features=out_features,
            learnable_graph=learnable,
        )
        X = _make_spd((5, n_nodes, in_features, in_features))
        Y = layer(X)

        # Symmetry check: Y == Y.mT
        sym_diff = torch.norm(Y - Y.mT, dim=(-2, -1)).max().item()
        assert sym_diff < 1e-5, f"Output not symmetric: diff={sym_diff}"

        # Positive definiteness check: all eigenvalues must be > 0
        eigvals = torch.linalg.eigvalsh(Y)
        assert (eigvals > 0).all(), f"Eigenvalues not all positive: min={eigvals.min()}"


def test_GraphBiMap_learnable_graph_gradient():
    """Verify gradients flow to learnable graph logits."""
    n_nodes = 3
    in_features = 4
    out_features = 4

    layer = GraphBiMap(
        n_nodes=n_nodes,
        in_features=in_features,
        out_features=out_features,
        learnable_graph=True,
    )
    assert hasattr(layer, "graph_logits")
    assert layer.graph_logits.requires_grad

    X = _make_spd((2, n_nodes, in_features, in_features))
    Y = layer(X)
    loss = Y.sum()
    loss.backward()

    assert layer.graph_logits.grad is not None
    assert torch.norm(layer.graph_logits.grad) > 0


def test_GraphBiMap_fixed_graph_buffer():
    """Verify fixed graph propagation matrix is stored as buffer."""
    n_nodes = 3
    in_features = 4
    out_features = 4

    custom_p = torch.tensor([
        [0.6, 0.2, 0.2],
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
    ])
    layer = GraphBiMap(
        n_nodes=n_nodes,
        in_features=in_features,
        out_features=out_features,
        graph_matrix=custom_p,
        learnable_graph=False,
    )
    assert "graph_matrix" in layer._buffers
    assert torch.allclose(layer.graph_matrix, custom_p)


def test_GraphBiMap_init_methods():
    """Test different initialization methods and seed reproducibility."""
    for init in ["kaiming_uniform", "orthogonal", "stiefel"]:
        m1 = GraphBiMap(n_nodes=3, in_features=4, out_features=2, init_method=init, seed=123)
        m2 = GraphBiMap(n_nodes=3, in_features=4, out_features=2, init_method=init, seed=123)
        assert torch.allclose(m1.bimap.weight, m2.bimap.weight)


def test_GraphBiMap_invalid_shapes():
    """Test that invalid input shapes raise informative ValueErrors."""
    layer = GraphBiMap(n_nodes=4, in_features=6, out_features=3)

    # 2D input
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        layer(torch.randn(6, 6))

    # Mismatched n_nodes
    with pytest.raises(ValueError, match="Expected n_nodes=4"):
        layer(torch.randn(2, 3, 6, 6))

    # Mismatched in_features
    with pytest.raises(ValueError, match="Expected matrix size"):
        layer(torch.randn(2, 4, 5, 5))


# =============================================================================
# GraphCSPNet Tests
# =============================================================================


def test_GraphCSPNet_3d_input():
    """Test GraphCSPNet with 3D raw EEG input (batch, n_chans, n_times)."""
    batch_size = 2
    n_chans = 22
    n_times = 500
    n_outputs = 4

    model = GraphCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_temp=3,
        n_freq=3,
    )
    x = torch.randn(batch_size, n_chans, n_times, requires_grad=True)
    out = model(x)

    assert out.shape == (batch_size, n_outputs)

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.norm(x.grad) > 0


def test_GraphCSPNet_4d_input():
    """Test GraphCSPNet with 4D filter-bank input (batch, n_freq, n_chans, n_times)."""
    batch_size = 2
    n_chans = 16
    n_freq = 4
    n_temp = 2
    n_times = 400
    n_outputs = 2

    model = GraphCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_temp=n_temp,
        n_freq=n_freq,
        bimap_dims=(20, 16),
    )
    x = torch.randn(batch_size, n_freq, n_chans, n_times, requires_grad=True)
    out = model(x)

    assert out.shape == (batch_size, n_outputs)

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.norm(x.grad) > 0


def test_GraphCSPNet_single_block():
    """Test GraphCSPNet with a single BiMap dimension."""
    model = GraphCSPNet(
        n_chans=10,
        n_outputs=2,
        n_temp=2,
        n_freq=2,
        bimap_dims=(8,),
    )
    x = torch.randn(2, 10, 300)
    out = model(x)
    assert out.shape == (2, 2)


def test_GraphCSPNet_eval_mode():
    """Test GraphCSPNet evaluation mode behaves deterministically."""
    model = GraphCSPNet(n_chans=8, n_outputs=2, n_temp=2, n_freq=2, drop_prob=0.5)

    # Run in train mode first to populate running mean in SPDBatchNormMean
    x = torch.randn(4, 8, 300)
    model.train()
    _ = model(x)

    # Run in eval mode
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2)


def test_GraphCSPNet_invalid_inputs():
    """Test GraphCSPNet invalid inputs raise ValueErrors."""
    model = GraphCSPNet(n_chans=8, n_outputs=2, n_freq=3)

    # 2D input
    with pytest.raises(ValueError, match="Expected 3D or 4D"):
        model(torch.randn(8, 300))

    # 5D input
    with pytest.raises(ValueError, match="Expected 3D or 4D"):
        model(torch.randn(2, 2, 3, 8, 300))

    # 4D input with wrong n_freq
    with pytest.raises(ValueError, match="Expected 4D input with n_freq=3"):
        model(torch.randn(2, 5, 8, 300))
