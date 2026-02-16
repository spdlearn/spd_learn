import pytest
import torch

from torch.testing import assert_close

from spd_learn.functional import sym_to_upper
from spd_learn.models.phase_spdnet import PhaseDelay
from spd_learn.modules.bilinear import BiMap, BiMapIncreaseDim
from spd_learn.modules.modeig import ExpEig, LogEig
from spd_learn.modules.residual import LogEuclideanResidual

from .constants import EXAMPLE_2X2_SPD_MATRIX


test_configs = [
    # User Examples + Variations
    (64, 32, "kaiming_uniform", True, None, "Kaiming (Parametrized)"),
    (64, 32, "orthogonal", True, None, "Orthogonal (Parametrized)"),
    (64, 32, "stiefel", False, 42, "Stiefel (Non-Param)"),
    # (64, 32, "orthogonal", False, None, "Orthogonal (Non-Param)"),
    # Additional cases
    # (16, 16, "stiefel", True, 123, "Square Stiefel (Parametrized)"),
    # (16, 16, "orthogonal", False, None, "Square Orthogonal (Non-Param)"),
    (10, 20, "kaiming_uniform", True, None, "Dim Increase Kaiming (Param)"),
    # (10, 20, "orthogonal", False, None, "Dim Increase Orthogonal (Non-Param)"),
]


@pytest.fixture(scope="module")
def device():
    """Fixture to provide the device (CPU or GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype():
    return torch.float32


@pytest.fixture
def input_data(device, dtype):
    """Fixture to create sample input data."""

    def _create_data(batch_size, n_features):
        X = torch.randn(batch_size, n_features, n_features, device=device, dtype=dtype)
        # Ensure symmetry (important for SPD matrix context)
        X = 0.5 * (X + X.mT)
        return X

    return _create_data


@pytest.mark.parametrize(
    "in_feat, out_feat, shape",
    [
        (3, 5, (2, 3, 3)),  # 3D input
        (5, 8, (4, 5, 5)),  # 3D with expansion
        (4, 6, (2, 3, 4, 4)),  # 4D with channels
        (2, 5, (5, 2, 2)),  # Edge case: minimal expansion
        (3, 3, (1, 3, 3)),  # No expansion (k = d)
    ],
)
def test_output_shapes(in_feat, out_feat, shape):
    """Test various input/output dimension combinations"""
    layer = BiMapIncreaseDim(in_feat, out_feat)
    x = torch.randn(*shape)
    output = layer(x)

    use_layer = BiMap(in_feat, out_feat)
    output_use = use_layer(x)

    # Verify output shape matches expected dimensions
    expected_shape = shape[:-2] + (
        out_feat,
        out_feat,
    )  # Adjusted for BiMap's dimensional behavior
    assert output.shape == expected_shape, (
        f"Failed for {in_feat}→{out_feat} with input {shape}"
    )
    assert output_use.shape == expected_shape, (
        f"Failed for {in_feat}→{out_feat} with input {shape}"
    )


@pytest.mark.parametrize(
    "in_feat, out_feat, shape",
    [
        (22, 36, (1, 3, 22, 22)),  # 4D with channels
        # Basic functionality
        (2, 4, (1, 1, 2, 2)),  # Minimal expansion (2→4), single channel
        (5, 5, (3, 2, 5, 5)),  # No expansion (5→5), multiple channels
        (3, 10, (2, 4, 3, 3)),  # Moderate expansion (3→10), multi-channel
        # Batch dimension edge cases
        (4, 6, (0, 3, 4, 4)),  # Empty batch dimension
        (2, 5, (100, 5, 2, 2)),  # Large batch size
        # depth dimension extremes
        (3, 7, (5, 1, 3, 3)),  # Single depht
        (4, 8, (2, 1024, 4, 4)),  # Extreme depth count
        # Feature dimension boundaries
        (1, 3, (4, 2, 1, 1)),  # Minimal input features (1→3)
        # dont make much sense, but go to test
        (256, 512, (2, 3, 256, 256)),  # Large feature dimensions
    ],
)
def test_output_shapes_depth(in_feat, out_feat, shape):
    """Test various input/output dimension combinations"""
    x = torch.randn(*shape)

    use_layer = BiMap(in_features=in_feat, out_features=out_feat)
    output_use = use_layer(x)

    # Verify output shape matches expected dimensions
    expected_shape = shape[:-2] + (out_feat, out_feat)
    assert output_use.shape == expected_shape, (
        f"Failed for {in_feat}→{out_feat} with input {shape}"
    )


def test_spd_property_preservation():
    """Verify output remains symmetric positive definite"""
    torch.manual_seed(42)
    in_feat, out_feat = 3, 5

    # Create valid SPD input (covariance matrix)
    x = torch.randn(10, 3, 3)
    x = x @ x.transpose(1, 2) + 1e-3 * torch.eye(3)  # Ensure positive definite

    layer = BiMapIncreaseDim(in_feat, out_feat)
    output = layer(x)

    # Test symmetry
    assert_close(output, output.transpose(-1, -2), msg="Output should be symmetric")

    # Test positive definiteness via Cholesky decomposition
    try:
        torch.linalg.cholesky(output)
    except RuntimeError as e:
        pytest.fail(f"Output not positive definite: {e}")


def test_structural_mathematical_properties():
    """Verify deep mathematical structure of the transformation"""
    in_feat, out_feat = 3, 5
    layer = BiMapIncreaseDim(in_feat, out_feat)
    W = layer.projection_matrix  # (5, 3)
    P = layer.add  # (5, 5)

    # Test case 1: X = identity matrix
    X_identity = torch.eye(in_feat).unsqueeze(0)  # (1, 3, 3)
    Y_identity = layer(X_identity)

    # Theoretical prediction: Y = P + W @ W.T
    WWT = W @ W.T  # (5,5)
    expected_Y_identity = P + WWT

    assert_close(
        Y_identity.squeeze(0),
        expected_Y_identity,
        rtol=1e-6,
        atol=1e-6,
        msg="Full matrix mismatch for X=identity",
    )

    # Verify upper-left block matches W@W.T and lower-right is identity
    upper_left = Y_identity[0, :in_feat, :in_feat]
    lower_right = Y_identity[0, in_feat:, in_feat:]
    off_diag = Y_identity[0, :in_feat, in_feat:]

    assert_close(
        upper_left, WWT[:in_feat, :in_feat], msg="Upper-left block mismatch for X=I"
    )
    assert_close(
        lower_right,
        torch.eye(out_feat - in_feat),
        msg="Lower-right block should be identity",
    )
    assert torch.allclose(off_diag, torch.zeros_like(off_diag)), (
        "Off-diagonal blocks should be zero"
    )

    # Test case 2: X = zero matrix
    X_zero = torch.zeros(1, in_feat, in_feat)
    Y_zero = layer(X_zero)
    assert_close(
        Y_zero, P.unsqueeze(0).float(), msg="Zero input should produce P matrix"
    )

    # Test case 3: General SPD matrix properties
    X = torch.randn(10, in_feat, in_feat)
    X = X @ X.transpose(1, 2)  # Make SPD
    Y = layer(X)

    # Direct computation of expected result
    W_batch = W.unsqueeze(0).expand(10, -1, -1)
    expected_Y = P.unsqueeze(0) + torch.bmm(
        W_batch, torch.bmm(X, W_batch.transpose(1, 2))
    )

    assert_close(Y, expected_Y, msg="Numerical mismatch in direct computation")

    # Test trace properties
    trace_X = X.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_Y = Y.diagonal(dim1=-2, dim2=-1).sum(-1)

    # Theoretical trace relationship: tr(Y) = tr(P) + tr(W@X@W.T)
    # Since W is semi-orthogonal: tr(W@X@W.T) = tr(X@W.T@W) = tr(X)
    expected_trace = (out_feat - in_feat) + trace_X

    assert_close(trace_Y, expected_trace, msg="Trace preservation property failed")


def test_semi_orthogonal_projection():
    """Verify W^T W = I (semi-orthogonal property)"""
    in_feat, out_feat = 4, 7
    layer = BiMapIncreaseDim(in_feat, out_feat)

    W = layer.projection_matrix
    identity = torch.eye(in_feat, dtype=W.dtype, device=W.device)

    # Verify W^T W = I
    WTW = W.T @ W
    assert_close(
        WTW,
        identity,
        atol=1e-6,
        rtol=1e-6,
        msg="Projection matrix should be semi-orthogonal",
    )


def test_padding_matrix_structure():
    """Verify correct construction of identity padding matrix."""
    in_feat, out_feat = 3, 6
    layer = BiMapIncreaseDim(in_feat, out_feat)

    # Extract the diagonal from the padding matrix.
    diag = torch.diag(layer.add)
    # Create expected diagonal tensor, and cast to the same device and dtype.
    expected = torch.cat([torch.zeros(in_feat), torch.ones(out_feat - in_feat)]).to(
        device=layer.add.device, dtype=layer.add.dtype
    )

    # Cast both diag and expected to float for comparison.
    diag_float = diag.float()
    expected_float = expected.float()

    assert_close(
        diag_float,
        expected_float,
        msg="Padding matrix should have 0s then 1s on its diagonal",
    )
    assert_close(
        layer.add.float(),
        torch.diag_embed(diag_float),
        msg="Padding matrix should be diagonal",
    )


def test_dimension_validation():
    """Test error is raised when out_features < in_features"""
    with pytest.raises(ValueError) as excinfo:
        BiMapIncreaseDim(5, 3)  # Invalid: 3 < 5

    assert "Output features must be >= input features" in str(excinfo.value), (
        "Should raise error for out_features < in_features"
    )


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("lag", [1, 2, 3])
def test_delay_order(order, lag):
    """Test if  input/output dimension combinations"""
    n_times = 1000
    n_chans = 22
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    layer = PhaseDelay(order=order, lag=lag)
    output = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    # Verify output shape matches expected dimensions
    if order == 1:
        expected_shape = (batch_size, n_chans, n_times)
    else:
        expected_shape = (batch_size, n_chans * order, n_times - (order * lag))
    assert output.shape == expected_shape, (
        f"Failed for {order}→{lag} with input {x.shape}"
    )


@pytest.mark.parametrize(
    "in_f, out_f, init, param, seed, desc",
    test_configs,
    ids=[c[-1] for c in test_configs],  # Use description as test ID
)
def test_bimap_instantiation_and_shape(
    in_f, out_f, init, param, seed, desc, input_data, device, dtype
):
    """Tests if BiMap layer instantiates and produces correct output shape."""
    layer = BiMap(
        in_features=in_f,
        out_features=out_f,
        init_method=init,
        parametrized=param,
        seed=seed,
        device=device,
        dtype=dtype,
    )

    batch_size = 4
    X = input_data(batch_size, in_f)  # Input uses original in_features

    # Forward pass
    Y = layer(X)

    # Check output shape: (batch_size, out_features, out_features)
    expected_shape = (batch_size, out_f, out_f)
    assert Y.shape == expected_shape, f"Failed shape check for {desc}"
    assert Y.device == device, f"Failed device check for {desc}"
    assert Y.dtype == dtype, f"Failed dtype check for {desc}"


@pytest.mark.parametrize(
    "in_f, out_f, init, param, seed, desc",
    test_configs,
    ids=[c[-1] for c in test_configs],
)
def test_bimap_orthogonality_property(
    in_f, out_f, init, param, seed, desc, device, dtype
):
    """Tests the orthogonality property (W^T W = I) of the weight matrix W.
    - If parametrized=True, the *effective* weight should be orthogonal.
    - If parametrized=False, the *raw* weight depends on the init_method.
    """
    atol = 1e-5  # Absolute tolerance for checking closeness to identity

    # Skip Stiefel for dimension increase (n < m), as it's not defined/tested here
    if init == "stiefel" and out_f > in_f:
        pytest.skip(
            f"Skipping Stiefel orthogonality check for dim increase n={in_f}, m={out_f}"
        )

    # --- Instantiate Layer ---
    layer = BiMap(
        in_features=in_f,
        out_features=out_f,
        init_method=init,
        parametrized=param,
        seed=seed,
        device=device,
        dtype=dtype,
    )

    # --- Get the relevant weight matrix W ---
    # W has shape (1, n_eff, m) where n_eff = max(in_f, out_f) if dim increased
    n_eff = layer._in_features  # Actual dimension used by W
    m = layer._out_features

    if param:
        # Access the weight *through* the parametrization context
        # Note: layer.weight directly gives the parametrized version
        effective_weight = layer.weight.squeeze(0)  # Shape (n_eff, m)
        W_to_check = effective_weight
        check_property = True  # Parametrization should always enforce orthogonality
        desc_detail = f"{desc} (effective W)"
    else:
        # Access the raw weight data directly
        raw_weight = layer.weight.data.squeeze(0)  # Shape (n_eff, m)
        W_to_check = raw_weight
        # Check property only if init method promises orthogonality
        check_property = init in ["orthogonal", "stiefel"]
        desc_detail = f"{desc} (raw W)"

    # --- Perform Check: W^T W = I ---
    identity_m = torch.eye(m, device=device, dtype=dtype)
    WtW = W_to_check.mT @ W_to_check  # Shape (m, m)

    is_close_to_identity = torch.allclose(WtW, identity_m, atol=atol)

    if check_property:
        assert is_close_to_identity, (
            f"Orthogonality (W^T W = I) check failed for {desc_detail}. Dist: {torch.dist(WtW, identity_m)}"
        )
    else:
        # For Kaiming init (non-parametrized), it should generally *not* be orthogonal
        if init == "kaiming_uniform":
            assert not is_close_to_identity, (
                f"Kaiming init (non-param) unexpectedly produced orthogonal matrix for {desc_detail}"
            )
        # Special case: Orthogonal init with n < m -> W W^T = I (orthonormal rows)
        elif init == "orthogonal" and n_eff < m:
            identity_n = torch.eye(n_eff, device=device, dtype=dtype)
            WWt = W_to_check @ W_to_check.mT  # Shape (n_eff, n_eff)
            is_WWt_identity = torch.allclose(WWt, identity_n, atol=atol)
            assert is_WWt_identity, (
                f"Orthogonality (W W^T = I) check failed for {desc_detail} (n<m case). Dist: {torch.dist(WWt, identity_n)}"
            )
            assert not is_close_to_identity, (
                f"Orthogonality (W^T W = I) check unexpectedly passed for {desc_detail} (n<m case)"
            )
        else:
            # Should not happen based on current logic, but good practice
            pytest.fail(
                f"Unexpected condition in orthogonality check for {desc_detail}"
            )


@pytest.mark.parametrize(
    "in_f, out_f, init, param, seed, desc",
    # Select a few representative cases for gradient check
    [
        (10, 5, "kaiming_uniform", True, None, "Grad Kaiming (Param)"),
        (10, 5, "stiefel", False, 42, "Grad Stiefel (Non-Param)"),
        (5, 10, "orthogonal", True, None, "Grad Dim Increase (Param)"),
    ],
    ids=[
        c[-1]
        for c in [  # Manually create IDs for this subset
            (10, 5, "kaiming_uniform", True, None, "Grad Kaiming (Param)"),
            (10, 5, "stiefel", False, 42, "Grad Stiefel (Non-Param)"),
            (5, 10, "orthogonal", True, None, "Grad Dim Increase (Param)"),
        ]
    ],
)
def test_bimap_gradient_flow(
    in_f, out_f, init, param, seed, desc, input_data, device, dtype
):
    """Tests if gradients flow back to the layer's weights."""
    layer = BiMap(
        in_features=in_f,
        out_features=out_f,
        init_method=init,
        parametrized=param,
        seed=seed,
        device=device,
        dtype=dtype,
    ).to(device)  # Ensure layer is on the correct device

    # Ensure weights require gradients (should be true by default)
    if param:
        # Check the original weight inside the parametrization
        assert layer.parametrizations.weight.original.requires_grad
    else:
        assert layer.weight.requires_grad

    X = input_data(2, in_f).requires_grad_(False)  # Input usually doesn't require grad

    # Forward pass
    Y = layer(X)

    # Dummy loss and backward pass
    loss = Y.mean()  # A scalar loss
    loss.backward()

    # Check if gradients exist
    grad = None
    if param:
        # Gradient attaches to the *original* parameter before parametrization
        param_obj = layer.parametrizations.weight
        if hasattr(param_obj, "original"):
            grad = param_obj.original.grad
        else:
            pytest.fail(
                f"Could not find 'original' parameter in parametrization for {desc}"
            )
    else:
        # Gradient attaches directly to the weight parameter
        grad = layer.weight.grad

    assert grad is not None, f"Gradient is None for {desc}"
    assert not torch.allclose(grad, torch.zeros_like(grad)), (
        f"Gradient is all zeros for {desc}"
    )
    assert grad.shape == layer.weight.shape, f"Gradient shape mismatch for {desc}"


def test_bimap_invalid_init_method(device, dtype):
    """Tests instantiation with an invalid init_method."""
    with pytest.raises(ValueError, match="Unknown init_method"):
        BiMap(10, 5, init_method="invalid_method", device=device, dtype=dtype)


def test_bimap_orthogonal_map_without_parametrization(device, dtype):
    """Tests error if orthogonal_map is given when parametrized=False."""
    with pytest.raises(
        ValueError, match="orthogonal_map is only used when parametrized is True"
    ):
        BiMap(
            10,
            5,
            parametrized=False,
            orthogonal_map="matrix_exp",
            device=device,
            dtype=dtype,
        )


def test_log_euclidean_residual():
    """Test Log-Euclidean residual calculation.

    Uses a 2x2 SPD matrix with known eigenvalues (5, 15) to verify
    the residual computation: exp(log(X) + log(Y)) where X = Y.
    """
    X = EXAMPLE_2X2_SPD_MATRIX
    Y = EXAMPLE_2X2_SPD_MATRIX

    residual = LogEuclideanResidual()
    actual = residual(X, Y)

    # For X = Y, the result should have eigenvalues that are
    # exp(2 * log(eigenvalues)) = eigenvalues^2
    # Original eigenvalues: 5, 15 -> Expected: 25, 225
    expected_eigenvalues = torch.tensor([25.0, 225.0])
    actual_eigenvalues, _ = torch.linalg.eigh(actual)

    assert_close(
        actual_eigenvalues,
        expected_eigenvalues,
        rtol=1e-5,
        atol=1e-5,
        msg="Log-Euclidean residual eigenvalue calculation failed",
    )


def test_log_euclidean_residual_preserves_spd():
    """Test that Log-Euclidean residual preserves SPD property."""
    torch.manual_seed(42)

    # Create random SPD matrices
    n = 5
    batch_size = 4

    A = torch.randn(batch_size, n, n)
    X = A @ A.mT + 0.1 * torch.eye(n)

    B = torch.randn(batch_size, n, n)
    Y = B @ B.mT + 0.1 * torch.eye(n)

    residual = LogEuclideanResidual()
    Z = residual(X, Y)

    # Check symmetry
    assert_close(Z, Z.mT, msg="Output is not symmetric")

    # Check positive definiteness via eigenvalues
    eigenvalues = torch.linalg.eigvalsh(Z)
    assert (eigenvalues > 0).all(), "Output has non-positive eigenvalues"


def test_log_euclidean_residual_gradient_flow():
    """Test that gradients flow through the Log-Euclidean residual."""
    torch.manual_seed(42)

    n = 4
    A = torch.randn(2, n, n)
    X = A @ A.mT + 0.1 * torch.eye(n)
    X.requires_grad_(True)

    B = torch.randn(2, n, n)
    Y = B @ B.mT + 0.1 * torch.eye(n)
    Y.requires_grad_(True)

    residual = LogEuclideanResidual()
    Z = residual(X, Y)

    loss = Z.mean()
    loss.backward()

    assert X.grad is not None, "Gradient not computed for X"
    assert Y.grad is not None, "Gradient not computed for Y"
    assert not torch.allclose(X.grad, torch.zeros_like(X.grad)), "X gradient is zero"
    assert not torch.allclose(Y.grad, torch.zeros_like(Y.grad)), "Y gradient is zero"


# =============================================================================
# LogEig / ExpEig shape and round-trip tests
# =============================================================================


@pytest.mark.parametrize(
    "upper,flatten,expected_ndim",
    [
        (True, True, 1),  # (batch, n(n+1)/2)
        (True, False, 1),  # upper takes precedence -> (batch, n(n+1)/2)
        (False, True, 1),  # (batch, n*n)
        (False, False, 2),  # (batch, n, n)
    ],
)
def test_logeig_output_shape(upper, flatten, expected_ndim, random_spd_matrix):
    """Verify LogEig output ndim for all upper/flatten combinations."""
    n = 5
    X = random_spd_matrix(batch_size=3, n_channels=n)
    layer = LogEig(upper=upper, flatten=flatten)
    Y = layer(X)

    # expected_ndim counts the non-batch dimensions
    assert Y.ndim - 1 == expected_ndim, (
        f"upper={upper}, flatten={flatten}: expected {expected_ndim} "
        f"non-batch dims, got {Y.ndim - 1}"
    )


@pytest.mark.parametrize(
    "upper,flatten",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_logeig_output_dimension(upper, flatten, random_spd_matrix):
    """Verify LogEig output vector length matches expected formula."""
    n = 5
    X = random_spd_matrix(batch_size=3, n_channels=n)
    layer = LogEig(upper=upper, flatten=flatten)
    Y = layer(X)

    if upper:
        expected_last = n * (n + 1) // 2
        assert Y.shape[-1] == expected_last
    elif flatten:
        expected_last = n * n
        assert Y.shape[-1] == expected_last
    else:
        assert Y.shape[-2:] == (n, n)


@pytest.mark.parametrize(
    "upper,flatten",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_expeig_output_shape(upper, flatten, random_spd_matrix):
    """Verify ExpEig always outputs (batch, n, n) SPD matrices."""
    n = 5
    X = random_spd_matrix(batch_size=3, n_channels=n)

    # Get LogEig output to feed into ExpEig
    log_layer = LogEig(upper=upper, flatten=flatten)
    X_log = log_layer(X)

    exp_layer = ExpEig(upper=upper, flatten=flatten)
    X_exp = exp_layer(X_log)

    assert X_exp.shape == (3, n, n), (
        f"upper={upper}, flatten={flatten}: "
        f"expected (3, {n}, {n}), got {X_exp.shape}"
    )


@pytest.mark.parametrize(
    "upper,flatten",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_logeig_expeig_roundtrip_shape(upper, flatten, random_spd_matrix):
    """Verify LogEig -> ExpEig round-trip recovers the original matrix."""
    n = 5
    X = random_spd_matrix(batch_size=3, n_channels=n)

    log_layer = LogEig(upper=upper, flatten=flatten)
    exp_layer = ExpEig(upper=upper, flatten=flatten)

    X_reconstructed = exp_layer(log_layer(X))

    assert X_reconstructed.shape == X.shape, (
        f"Round-trip shape mismatch: {X_reconstructed.shape} vs {X.shape}"
    )
    assert_close(
        X_reconstructed,
        X,
        atol=1e-4,
        rtol=1e-4,
        msg=f"Round-trip value mismatch for upper={upper}, flatten={flatten}",
    )


@pytest.mark.parametrize("n_channels", [3, 5, 10])
@pytest.mark.parametrize(
    "upper,flatten",
    [
        (True, True),
        (False, True),
        (False, False),
    ],
)
def test_logeig_expeig_shape_across_sizes(
    n_channels, upper, flatten, random_spd_matrix
):
    """Verify shape consistency across different matrix dimensions."""
    X = random_spd_matrix(batch_size=2, n_channels=n_channels)

    log_layer = LogEig(upper=upper, flatten=flatten)
    exp_layer = ExpEig(upper=upper, flatten=flatten)

    X_log = log_layer(X)
    X_roundtrip = exp_layer(X_log)

    assert X_roundtrip.shape == X.shape, (
        f"n={n_channels}, upper={upper}, flatten={flatten}: "
        f"roundtrip shape {X_roundtrip.shape} != input shape {X.shape}"
    )


def test_expeig_accepts_upper_vectorized_input(random_spd_matrix):
    """Verify ExpEig decodes upper-triangular vectors before exp."""
    X = random_spd_matrix(batch_size=2, n_channels=4)
    tangent = LogEig(upper=False, flatten=False)(X)

    expected = ExpEig(upper=False, flatten=False)(tangent)
    got = ExpEig(upper=True, flatten=False)(sym_to_upper(tangent))

    assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_expeig_accepts_flattened_input(random_spd_matrix):
    """Verify ExpEig decodes flattened matrices before exp."""
    X = random_spd_matrix(batch_size=2, n_channels=4)
    tangent = LogEig(upper=False, flatten=False)(X)

    expected = ExpEig(upper=False, flatten=False)(tangent)
    got = ExpEig(upper=False, flatten=True)(tangent.flatten(start_dim=-2))

    assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_expeig_raises_on_invalid_upper_length():
    """Upper-vectorized input length must be triangular."""
    layer = ExpEig(upper=True, flatten=False)
    bad = torch.randn(2, 8)  # 8 != n(n+1)/2 for any integer n
    with pytest.raises(ValueError, match="n\\(n\\+1\\)/2"):
        layer(bad)


def test_expeig_raises_on_invalid_flatten_length():
    """Flattened input length must be a perfect square."""
    layer = ExpEig(upper=False, flatten=True)
    bad = torch.randn(2, 10)  # 10 != n*n for any integer n
    with pytest.raises(ValueError, match="n\\*n"):
        layer(bad)


def test_expeig_raises_on_non_square_matrix():
    """Matrix input must be square when upper/flatten are disabled."""
    layer = ExpEig(upper=False, flatten=False)
    bad = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="shape \\(\\.\\.\\., n, n\\)"):
        layer(bad)
