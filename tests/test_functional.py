import torch

from torch.autograd import gradcheck


def test_matrix_log():
    from spd_learn.functional import ensure_sym, matrix_log

    ndim = 10
    threshold = 1e-3

    # generate random SPD matrix
    A = torch.randn((1, ndim, ndim), dtype=torch.double)
    U, s, _ = torch.linalg.svd(A)
    s = s.clamp(min=threshold)
    X = U @ torch.diag_embed(s) @ U.transpose(-1, -2)

    # test forward pass
    log_X = U @ torch.diag_embed(s.log()) @ U.transpose(-1, -2)
    assert matrix_log.apply(X).allclose(log_X)
    X.requires_grad_(True)

    # Numerical tests for backward pass
    def safe_matrix_log(matrix):
        return matrix_log.apply(ensure_sym(matrix))

    assert gradcheck(safe_matrix_log, X)


def test_matrix_exp():
    from spd_learn.functional import ensure_sym, matrix_exp

    ndim = 10

    # generate random SPD matrix
    A = torch.randn((1, ndim, ndim), dtype=torch.double)
    U, s, _ = torch.linalg.svd(A)
    X = U @ torch.diag_embed(s) @ U.transpose(-1, -2)

    # test forward pass
    exp_X = U @ torch.diag_embed(s.exp()) @ U.transpose(-1, -2)
    assert matrix_exp.apply(X).allclose(exp_X)
    X.requires_grad_(True)

    # numerical tests for backward pass
    def safe_matrix_exp(matrix):
        return matrix_exp.apply(ensure_sym(matrix))

    assert gradcheck(safe_matrix_exp, X)


def test_clamp_eigvals():
    from spd_learn.functional import clamp_eigvals, ensure_sym

    def safe_clamp_eigvals(matrix, threshold):
        return clamp_eigvals.apply(ensure_sym(matrix), threshold)

    ndim = 10
    nsamples = 5
    threshold = 1e-3

    # generate random base SPD matrix
    A = torch.randn((1, ndim, ndim), dtype=torch.double)
    U, s, _ = torch.linalg.svd(A)

    # generate batches
    # linear case (all eigenvalues are above the threshold)
    s = threshold * 1e1 + torch.rand((nsamples, ndim), dtype=torch.double) * threshold
    X = U @ torch.diag_embed(s) @ U.transpose(-1, -2)

    assert clamp_eigvals.apply(X, threshold).allclose(X)
    X.requires_grad_(True)
    assert gradcheck(safe_clamp_eigvals, (X, threshold))

    # non-linear case (some eigenvalues are below the threshold)
    s = torch.rand((nsamples, ndim), dtype=torch.double) * threshold
    s[::2] += threshold
    X = U @ torch.diag_embed(s) @ U.transpose(-1, -2)
    assert not clamp_eigvals.apply(X, threshold).allclose(X)
    X.requires_grad_(True)
    assert gradcheck(safe_clamp_eigvals, (X, threshold))


def test_abs_eigvals():
    from spd_learn.functional import abs_eigvals, ensure_sym

    def safe_abs_eigvals(matrix):
        return abs_eigvals.apply(ensure_sym(matrix))

    ndim = 5

    # generate random base SPD matrix
    U = torch.linalg.qr(torch.randn((ndim, ndim), dtype=torch.double)).Q

    # test all positive numbers
    # avoid 0 (i.e., undefined derivative)
    s = torch.arange(1, ndim + 1).to(U)
    X_pos = U @ torch.diag_embed(s) @ U.transpose(-1, -2)
    assert abs_eigvals.apply(X_pos).allclose(X_pos)
    X_pos.requires_grad_(True)
    assert gradcheck(safe_abs_eigvals, X_pos)

    # test all negative numbers
    X_neg = U @ torch.diag_embed(s * (-1)) @ U.transpose(-1, -2)
    assert abs_eigvals.apply(X_neg).allclose(X_pos)
    X_neg.requires_grad_(True)
    assert gradcheck(safe_abs_eigvals, X_neg)


def test_matrix_power():
    from spd_learn.functional import ensure_sym, matrix_power

    def safe_matrix_power(matrix, exponent):
        return matrix_power.apply(ensure_sym(matrix), exponent)

    ndim = 5

    # generate random base SPD matrix
    A = torch.randn((1, ndim, ndim), dtype=torch.double)
    U, s, _ = torch.linalg.svd(A)
    X = U @ torch.diag_embed(s) @ U.mT

    assert matrix_power.apply(X, 3.0).allclose(X @ X @ X)
    X.requires_grad_(True)
    assert gradcheck(safe_matrix_power, (X, 0.1))


def test_matrix_sqrt():
    from spd_learn.functional import ensure_sym, matrix_sqrt

    def safe_matrix_sqrt(matrix):
        return matrix_sqrt.apply(ensure_sym(matrix))

    ndim = 5

    # generate random base SPD matrix
    A = torch.randn((1, ndim, ndim), dtype=torch.double)
    U, s, _ = torch.linalg.svd(A)
    s.clamp_(min=1e-3)
    X = U @ torch.diag_embed(s) @ U.mT
    X_sqrt = U @ torch.diag_embed(s.sqrt()) @ U.mT
    assert matrix_sqrt.apply(X).allclose(X_sqrt)
    X.requires_grad_(True)
    assert gradcheck(safe_matrix_sqrt, X)


def test_matrix_inv_sqrt():
    from spd_learn.functional import ensure_sym, matrix_inv_sqrt

    def safe_matrix_inv_sqrt(matrix):
        return matrix_inv_sqrt.apply(ensure_sym(matrix))

    ndim = 5

    # generate random base SPD matrix
    A = torch.randn((1, ndim, ndim), dtype=torch.double)
    U, s, _ = torch.linalg.svd(A)
    s.clamp_(min=1e-2)
    X = U @ torch.diag_embed(s) @ U.mT
    X_invsqrt = U @ torch.diag_embed(s.rsqrt()) @ U.mT
    assert matrix_inv_sqrt.apply(X).allclose(X_invsqrt)
    X.requires_grad_(True)
    assert gradcheck(safe_matrix_inv_sqrt, X)


def test_matrix_sqrt_inv():
    from spd_learn.functional import ensure_sym, matrix_sqrt_inv

    def safe_matrix_sqrt_inv(matrix):
        return matrix_sqrt_inv.apply(ensure_sym(matrix))

    ndim = 5

    # generate random base SPD matrix
    A = torch.randn((1, ndim, ndim), dtype=torch.double)
    U, s, _ = torch.linalg.svd(A)
    s.clamp_(min=1e-2)
    X = U @ torch.diag_embed(s) @ U.mT
    X_sqrt = U @ torch.diag_embed(s.sqrt()) @ U.mT
    X_invsqrt = U @ torch.diag_embed(s.rsqrt()) @ U.mT

    X_sqrt_hat, X_invsqrt_hat = matrix_sqrt_inv.apply(X)
    assert X_sqrt_hat.allclose(X_sqrt)
    assert X_invsqrt_hat.allclose(X_invsqrt)
    X.requires_grad_(True)
    assert gradcheck(safe_matrix_sqrt_inv, X)


def test_log_euclidean_mean():
    from spd_learn.functional import log_euclidean_mean, matrix_exp, matrix_log

    torch.manual_seed(0)
    batch_size = 4
    ndim = 5

    A = torch.randn((batch_size, ndim, ndim), dtype=torch.double)
    V = A @ A.transpose(-1, -2) + 1e-3 * torch.eye(ndim, dtype=torch.double)

    # Test uniform weights
    weights = torch.full((1, batch_size), 1.0 / batch_size, dtype=torch.double)
    mean_uniform = log_euclidean_mean(weights, V).squeeze(0)

    # Verify against direct computation: exp(mean(log(V)))
    log_V = matrix_log.apply(V)
    expected_mean = matrix_exp.apply(log_V.mean(dim=0))
    assert mean_uniform.allclose(expected_mean, atol=1e-6, rtol=1e-5)

    # Test custom weights
    weights_custom = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.double)
    weights_custom = weights_custom / weights_custom.sum()
    mean_weighted = log_euclidean_mean(weights_custom, V)

    # Verify weighted mean has correct shape
    assert mean_weighted.shape == (1, ndim, ndim)


def test_spd_egrad2rgrad():
    from spd_learn.functional import ensure_sym, spd_egrad2rgrad

    ndim = 5

    # generate random base SPD matrix
    A = torch.randn((1, ndim, ndim), dtype=torch.double)
    U, s, _ = torch.linalg.svd(A)
    X = U @ torch.diag_embed(s) @ U.mT

    # generate random Euclidean gradient
    grad_euc = torch.randn((1, ndim, ndim), dtype=torch.double)

    grad_riem = spd_egrad2rgrad(X, grad_euc)

    # Symmetry: gradient must lie in the tangent space (symmetric matrices)
    assert ensure_sym(grad_riem).allclose(grad_riem)

    # Batched check: shapes and symmetry preserved
    batch = 4
    A = torch.randn((batch, ndim, ndim), dtype=torch.double)
    U, s, _ = torch.linalg.svd(A)
    XB = U @ torch.diag_embed(s.clamp(min=1e-3)) @ U.mT
    GB = torch.randn((batch, ndim, ndim), dtype=torch.double)
    GR = spd_egrad2rgrad(XB, GB)
    assert GR.shape == XB.shape
    assert ensure_sym(GR).allclose(GR)


def test_airm_distance_matches_metric_impl():
    from spd_learn.functional import airm_distance
    from spd_learn.functional.metrics.affine_invariant import (
        airm_distance as airm_distance_impl,
    )

    torch.manual_seed(0)
    ndim = 4

    A = torch.randn((ndim, ndim), dtype=torch.double)
    B = torch.randn((ndim, ndim), dtype=torch.double)
    A = A @ A.T + 1e-3 * torch.eye(ndim, dtype=torch.double)
    B = B @ B.T + 1e-3 * torch.eye(ndim, dtype=torch.double)

    d_exported = airm_distance(A, B)
    d_impl = airm_distance_impl(A, B)
    assert d_exported.allclose(d_impl, atol=1e-8, rtol=1e-6)


def test_dropout_spd_sets_dropped_diag_to_epsilon():
    from spd_learn.functional import dropout_spd

    torch.manual_seed(0)
    ndim = 5
    epsilon = 1e-4

    A = torch.randn((ndim, ndim), dtype=torch.double)
    X = A @ A.T + 1e-3 * torch.eye(ndim, dtype=torch.double)

    dropped = dropout_spd(X, p=1.0, use_scaling=True, epsilon=epsilon)
    expected = epsilon * torch.eye(ndim, dtype=torch.double)
    assert dropped.allclose(expected)


def test_bures_wasserstein_geodesic_tensor_t():
    from spd_learn.functional import bures_wasserstein_geodesic

    torch.manual_seed(0)
    batch = 3
    ndim = 4

    A = torch.randn((batch, ndim, ndim), dtype=torch.double)
    B = torch.randn((batch, ndim, ndim), dtype=torch.double)
    A = A @ A.transpose(-1, -2) + 1e-3 * torch.eye(ndim, dtype=torch.double)
    B = B @ B.transpose(-1, -2) + 1e-3 * torch.eye(ndim, dtype=torch.double)

    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.double)
    out = bures_wasserstein_geodesic(A, B, t)
    assert out[0].allclose(A[0])
    assert out[2].allclose(B[2])


def test_cross_covariance_batched_shape():
    from spd_learn.functional import cross_covariance

    torch.manual_seed(0)
    batch1, batch2 = 2, 3
    n_freqs, n_chans, n_times = 4, 5, 6
    X = torch.randn((batch1, batch2, n_freqs, n_chans, n_times), dtype=torch.double)

    cov = cross_covariance(X)
    assert cov.shape == (batch1, batch2, n_freqs * n_chans, n_freqs * n_chans)


def test_sym_to_upper_vec_to_sym_roundtrip():
    from spd_learn.functional import sym_to_upper, vec_to_sym

    torch.manual_seed(0)
    ndim = 5
    batch_size = 3

    # Create symmetric matrices
    A = torch.randn((batch_size, ndim, ndim), dtype=torch.double)
    X = (A + A.transpose(-1, -2)) / 2

    # Test with preserve_norm=True, upper=True (defaults)
    v = sym_to_upper(X)
    X_reconstructed = vec_to_sym(v)
    assert X_reconstructed.allclose(X)

    # Test with preserve_norm=False, upper=True
    v_raw = sym_to_upper(X, preserve_norm=False)
    X_reconstructed_raw = vec_to_sym(v_raw, preserve_norm=False)
    assert X_reconstructed_raw.allclose(X)

    # Test with preserve_norm=True, upper=False (lower triangular)
    v_lower = sym_to_upper(X, upper=False)
    X_reconstructed_lower = vec_to_sym(v_lower, upper=False)
    assert X_reconstructed_lower.allclose(X)

    # Test with preserve_norm=False, upper=False
    v_lower_raw = sym_to_upper(X, preserve_norm=False, upper=False)
    X_reconstructed_lower_raw = vec_to_sym(
        v_lower_raw, preserve_norm=False, upper=False
    )
    assert X_reconstructed_lower_raw.allclose(X)


def test_sym_to_upper_norm_preservation():
    import math

    from spd_learn.functional import sym_to_upper

    torch.manual_seed(0)
    ndim = 4

    # Create a symmetric matrix
    A = torch.randn((ndim, ndim), dtype=torch.double)
    X = (A + A.T) / 2

    # Frobenius norm of original matrix
    frob_norm = torch.linalg.norm(X, ord="fro")

    # With preserve_norm=True, vector norm should equal Frobenius norm
    v_preserved = sym_to_upper(X, preserve_norm=True)
    vec_norm_preserved = torch.linalg.norm(v_preserved)
    assert vec_norm_preserved.allclose(frob_norm, rtol=1e-6)

    # With preserve_norm=False, norms will differ (off-diagonals counted twice in Frobenius)
    # Raw vech doesn't preserve the norm - diagonal elements are the same
    # but off-diagonals are not scaled, so total norm differs

    # Verify the scaling factor for off-diagonals
    upper_indices = torch.triu_indices(ndim, ndim, offset=0)
    off_diag_mask = upper_indices[0] != upper_indices[1]

    # Check that off-diagonal elements are scaled by sqrt(2)
    v_scaled = sym_to_upper(X, preserve_norm=True)
    v_unscaled = sym_to_upper(X, preserve_norm=False)
    off_diag_ratio = v_scaled[off_diag_mask] / v_unscaled[off_diag_mask]
    expected_ratio = torch.full_like(off_diag_ratio, math.sqrt(2))
    assert off_diag_ratio.allclose(expected_ratio)
