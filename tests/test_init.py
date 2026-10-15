# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the initialization functions in :mod:`spd_learn.init`."""

import logging

import pytest
import torch

import spd_learn.init as spd_init


def _assert_orthonormal_columns(W):
    eye = torch.eye(W.shape[-1], dtype=W.dtype).expand(*W.shape[:-2], -1, -1)
    torch.testing.assert_close(W.mT @ W, eye, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "shape, seed", [((6, 3), 7), ((2, 6, 3), 7), ((1, 16, 16), 123)]
)
def test_stiefel_is_orthonormal_and_reproducible(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    W1 = spd_init.stiefel_(
        torch.empty(shape, dtype=torch.float32, device="cpu"), generator=generator
    )

    generator.manual_seed(seed)
    W2 = spd_init.stiefel_(
        torch.empty(shape, dtype=torch.float32, device="cpu"), generator=generator
    )
    _assert_orthonormal_columns(W1)
    torch.testing.assert_close(W1, W2)


def test_stiefel_uses_default_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = spd_init.stiefel_(torch.empty(16, 8))
        second = spd_init.stiefel_(torch.empty(16, 8))

        assert not torch.equal(first, second)

        torch.manual_seed(123)
        torch.testing.assert_close(spd_init.stiefel_(torch.empty(16, 8)), first)
        torch.testing.assert_close(spd_init.stiefel_(torch.empty(16, 8)), second)


@pytest.mark.parametrize("shape", [(6, 3), (2, 6, 3)])
def test_stiefel_falls_back_to_qr(monkeypatch, caplog, shape):
    def failing_polar_factor(_):
        raise torch.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(spd_init, "orthogonal_polar_factor", failing_polar_factor)
    with caplog.at_level(logging.WARNING, logger=spd_init.logger.name):
        W = spd_init.stiefel_(torch.empty(shape))
    assert "Falling back to QR" in caplog.text
    assert W.shape == shape
    _assert_orthonormal_columns(W)


@pytest.mark.parametrize("shape", [(4, 4), (3, 2, 5, 5)])
def test_spd_identity(shape):
    tensor = torch.randn(shape)
    out = spd_init.spd_identity_(tensor)
    assert out is tensor
    torch.testing.assert_close(out, torch.eye(shape[-1]).expand(shape))


def test_spd_identity_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        spd_init.spd_identity_(torch.empty(3, 4))
